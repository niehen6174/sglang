# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the minimal etcd worker membership layer."""

import json
import unittest

from sglang.multimodal_gen.runtime.disaggregation.dag import DagSpec, ExecutionPlan
from sglang.multimodal_gen.runtime.disaggregation.membership import (
    EtcdKeyValue,
    WorkerRecord,
    WorkerRegistrar,
    discover_workers,
    wait_for_workers,
    worker_key,
)


class FakeEtcdClient:
    def __init__(self):
        self.values: dict[str, EtcdKeyValue] = {}
        self.next_lease = 100
        self.revoked: list[int] = []

    def grant_lease(self, _ttl: int) -> int:
        self.next_lease += 1
        return self.next_lease

    def keepalive(self, _lease_id: int) -> int:
        return 30

    def revoke_lease(self, lease_id: int) -> None:
        self.revoked.append(lease_id)

    def put(self, key: str, value: str, lease_id: int | None = None) -> None:
        self.values[key] = EtcdKeyValue(key, value, lease_id or 0)

    def get_prefix(self, prefix: str) -> list[EtcdKeyValue]:
        return [value for key, value in self.values.items() if key.startswith(prefix)]


class TestWorkerMembership(unittest.TestCase):
    def test_discovery_is_ready_only_and_sorted_by_instance_id(self):
        client = FakeEtcdClient()
        for instance_id in (10, 2):
            record = WorkerRecord.ready(
                instance_id=instance_id,
                node="denoiser",
                work_endpoint=f"tcp://worker-{instance_id}:35020",
                capacity=2,
            )
            client.put(worker_key("demo", "denoiser", instance_id), record.to_json(), 1)
        draining = WorkerRecord(
            instance_id=1,
            boot_id="old",
            node="denoiser",
            work_endpoint="tcp://old:35020",
            capacity=2,
            state="DRAINING",
        )
        client.put(worker_key("demo", "denoiser", 1), draining.to_json(), 1)

        records = discover_workers(client, "demo", "denoiser")

        self.assertEqual([record.instance_id for record in records], [2, 10])

    def test_wait_returns_startup_snapshot(self):
        client = FakeEtcdClient()
        record = WorkerRecord.ready(
            instance_id=0,
            node="encoder",
            work_endpoint="tcp://encoder:35010",
            capacity=4,
        )
        client.put(worker_key("demo", "encoder", 0), record.to_json(), 1)

        found = wait_for_workers(client, "demo", "encoder", 1, timeout_s=0)

        self.assertEqual(found, [record])

    def test_registrar_puts_lease_backed_key_and_revokes_it(self):
        client = FakeEtcdClient()
        record = WorkerRecord.ready(
            instance_id=3,
            node="decoder",
            work_endpoint="tcp://decoder:35030",
            capacity=4,
        )
        registrar = WorkerRegistrar(client, "demo", record, ttl_s=30)

        registrar.start()
        stored = client.values[registrar.key]
        registrar.stop()

        self.assertEqual(WorkerRecord.from_json(stored.value), record)
        self.assertEqual(stored.lease, 101)
        self.assertEqual(client.revoked, [101])

    def test_etcd_pool_can_compile_without_static_urls(self):
        spec = DagSpec.parse(
            {
                "source": "worker",
                "roles": [
                    {
                        "name": "worker",
                        "stages": ["Stage"],
                        "terminal": True,
                    }
                ],
                "pools": [
                    {
                        "role": "worker",
                        "discovery": "etcd",
                        "min_instances": 2,
                    }
                ],
            }
        )

        plan = ExecutionPlan.compile(spec)

        self.assertEqual(plan.node("worker").num_instances, 0)
        rendered = json.loads(json.dumps(spec.to_dict()))
        self.assertEqual(rendered["pools"][0]["discovery"], "etcd")
        self.assertEqual(rendered["pools"][0]["min_instances"], 2)


if __name__ == "__main__":
    unittest.main()
