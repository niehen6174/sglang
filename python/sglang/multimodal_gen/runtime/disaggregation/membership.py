# SPDX-License-Identifier: Apache-2.0
"""Minimal etcd-backed worker membership for diffusion disaggregation.

This module deliberately implements only the first control-plane boundary:
workers publish a READY descriptor under a lease, and the head takes one
membership snapshot before compiling its immutable ``ExecutionPlan``.  It
does not watch for changes or mutate a running worker pool.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

_KEY_ROOT = "/sglang/diffusion"


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _unb64(value: str) -> str:
    return base64.b64decode(value).decode()


def _prefix_end(prefix: str) -> str:
    raw = bytearray(prefix.encode())
    for index in range(len(raw) - 1, -1, -1):
        if raw[index] < 0xFF:
            raw[index] += 1
            return bytes(raw[: index + 1]).decode()
    return "\x00"


def _key_segment(value: str, name: str) -> str:
    value = str(value).strip()
    if not value or "/" in value:
        raise ValueError(f"{name} must be a non-empty path segment, got {value!r}")
    return value


def worker_prefix(cluster_id: str, node: str) -> str:
    cluster_id = _key_segment(cluster_id, "cluster_id")
    node = _key_segment(node, "node")
    return f"{_KEY_ROOT}/{cluster_id}/workers/{node}/"


def worker_key(cluster_id: str, node: str, instance_id: int) -> str:
    if instance_id < 0:
        raise ValueError(f"instance_id must be non-negative, got {instance_id}")
    return f"{worker_prefix(cluster_id, node)}{instance_id}"


@dataclass(frozen=True)
class EtcdKeyValue:
    key: str
    value: str
    lease: int


class EtcdClient(Protocol):
    def grant_lease(self, ttl: int) -> int: ...

    def keepalive(self, lease_id: int) -> int: ...

    def revoke_lease(self, lease_id: int) -> None: ...

    def put(self, key: str, value: str, lease_id: int | None = None) -> None: ...

    def get_prefix(self, prefix: str) -> list[EtcdKeyValue]: ...


class EtcdHttpClient:
    """Tiny client for etcd's v3 HTTP/JSON gateway.

    It keeps etcd optional: static DAG deployments never import an external
    etcd package. TLS, authentication, endpoint failover and watch resumption
    are intentionally outside this first version.
    """

    def __init__(self, endpoint: str, timeout_s: float = 3.0):
        if not endpoint:
            raise ValueError("etcd endpoint must not be empty")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.endpoint + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = response.read()
        return json.loads(raw or b"{}")

    def grant_lease(self, ttl: int) -> int:
        response = self._post("/v3/lease/grant", {"TTL": ttl})
        return int(response["ID"])

    def keepalive(self, lease_id: int) -> int:
        response = self._post("/v3/lease/keepalive", {"ID": lease_id})
        result = response.get("result", response)
        return int(result.get("TTL", 0))

    def revoke_lease(self, lease_id: int) -> None:
        self._post("/v3/lease/revoke", {"ID": lease_id})

    def put(self, key: str, value: str, lease_id: int | None = None) -> None:
        body: dict[str, str] = {"key": _b64(key), "value": _b64(value)}
        if lease_id is not None:
            body["lease"] = str(lease_id)
        self._post("/v3/kv/put", body)

    def get_prefix(self, prefix: str) -> list[EtcdKeyValue]:
        response = self._post(
            "/v3/kv/range",
            {"key": _b64(prefix), "range_end": _b64(_prefix_end(prefix))},
        )
        return [
            EtcdKeyValue(
                key=_unb64(item["key"]),
                value=_unb64(item["value"]),
                lease=int(item.get("lease", 0)),
            )
            for item in response.get("kvs", [])
        ]


@dataclass(frozen=True)
class WorkerRecord:
    instance_id: int
    boot_id: str
    node: str
    work_endpoint: str
    capacity: int
    state: str = "READY"
    schema_version: int = 1

    @classmethod
    def ready(
        cls, *, instance_id: int, node: str, work_endpoint: str, capacity: int
    ) -> "WorkerRecord":
        return cls(
            instance_id=instance_id,
            boot_id=str(uuid.uuid4()),
            node=node,
            work_endpoint=work_endpoint,
            capacity=capacity,
        )

    @classmethod
    def from_json(cls, raw: str) -> "WorkerRecord":
        data = json.loads(raw)
        record = cls(
            schema_version=int(data.get("schema_version", 1)),
            instance_id=int(data["instance_id"]),
            boot_id=str(data["boot_id"]),
            node=str(data["node"]),
            work_endpoint=str(data["work_endpoint"]),
            capacity=int(data["capacity"]),
            state=str(data.get("state", "READY")),
        )
        if record.schema_version != 1:
            raise ValueError(
                f"unsupported worker schema_version={record.schema_version}"
            )
        if record.instance_id < 0 or record.capacity < 1:
            raise ValueError("worker instance_id and capacity are invalid")
        if not record.work_endpoint:
            raise ValueError("worker work_endpoint must not be empty")
        return record

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def discover_workers(
    client: EtcdClient, cluster_id: str, node: str
) -> list[WorkerRecord]:
    """Return a deterministic READY snapshot for one DAG node."""
    records: list[WorkerRecord] = []
    for item in client.get_prefix(worker_prefix(cluster_id, node)):
        try:
            record = WorkerRecord.from_json(item.value)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid etcd worker record %s: %s", item.key, exc)
            continue
        if record.node == node and record.state == "READY":
            records.append(record)
    records.sort(key=lambda record: record.instance_id)
    return records


def wait_for_workers(
    client: EtcdClient,
    cluster_id: str,
    node: str,
    min_instances: int,
    timeout_s: float,
    *,
    poll_interval_s: float = 1.0,
) -> list[WorkerRecord]:
    if min_instances < 1:
        raise ValueError(f"min_instances must be at least 1, got {min_instances}")
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while True:
        try:
            records = discover_workers(client, cluster_id, node)
            last_error = None
            if len(records) >= min_instances:
                return records
        except Exception as exc:  # etcd may not be ready yet during startup
            records = []
            last_error = exc

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = f"; last etcd error: {last_error}" if last_error else ""
            raise TimeoutError(
                f"timed out waiting for {min_instances} worker(s) for node "
                f"{node!r}; found {len(records)}{detail}"
            )
        time.sleep(min(poll_interval_s, remaining))


class WorkerRegistrar:
    """Own a worker's etcd key and keep its lease alive."""

    def __init__(
        self,
        client: EtcdClient,
        cluster_id: str,
        record: WorkerRecord,
        ttl_s: int = 30,
    ):
        if ttl_s < 3:
            raise ValueError(f"etcd lease TTL must be at least 3 seconds, got {ttl_s}")
        self._client = client
        self._key = worker_key(cluster_id, record.node, record.instance_id)
        self._record = record
        self._ttl_s = ttl_s
        self._lease_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def key(self) -> str:
        return self._key

    def start(self) -> None:
        if self._thread is not None:
            return
        self._lease_id = self._client.grant_lease(self._ttl_s)
        self._client.put(self._key, self._record.to_json(), self._lease_id)
        self._thread = threading.Thread(
            target=self._keepalive_loop,
            name=f"etcd-keepalive-{self._record.node}-{self._record.instance_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Registered READY worker in etcd: key=%s endpoint=%s ttl=%ds",
            self._key,
            self._record.work_endpoint,
            self._ttl_s,
        )

    def _keepalive_loop(self) -> None:
        interval = max(1.0, self._ttl_s / 3)
        while not self._stop.wait(interval):
            try:
                remaining = self._client.keepalive(self._lease_id)
                if remaining <= 0:
                    # etcd may have restarted or been unreachable for longer
                    # than the TTL. Recreate the ephemeral membership key.
                    self._lease_id = self._client.grant_lease(self._ttl_s)
                    self._client.put(self._key, self._record.to_json(), self._lease_id)
                    logger.info("Re-registered expired etcd worker key %s", self._key)
            except Exception as exc:
                # Keep the worker process alive and retry on the next interval.
                logger.warning("etcd keepalive failed for %s: %s", self._key, exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._lease_id:
            try:
                self._client.revoke_lease(self._lease_id)
            except Exception as exc:
                logger.warning("Failed to revoke etcd lease for %s: %s", self._key, exc)
            self._lease_id = 0

    def __enter__(self) -> "WorkerRegistrar":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop()
