# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime import ipc_array
from sglang.multimodal_gen.runtime.ipc_array import (
    NumpyArrayFileRef,
    TorchTensorFileRef,
    is_local_endpoint,
    materialize_file_refs,
    spill_large_arrays_to_file_refs,
)


def test_spill_large_arrays_round_trips_and_removes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs([array])

    assert isinstance(spilled[0], NumpyArrayFileRef)
    spilled_path = Path(spilled[0].path)
    assert spilled_path.exists()

    materialized = materialize_file_refs(spilled)

    assert np.array_equal(materialized[0], array)
    assert not spilled_path.exists()


def test_small_arrays_are_kept_inline():
    array = np.arange(16, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs((array,))

    assert spilled[0] is array


def test_large_arrays_are_kept_inline_without_shm(monkeypatch):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: None)
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs(array)

    assert spilled is array


def test_spill_removes_temp_file_when_save_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)
    created_paths = []

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    original_mkstemp = tempfile.mkstemp

    def tracked_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created_paths.append(Path(path))
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(np, "save", fail_save)

    # A failed spill falls back to sending the payload inline.
    assert spill_large_arrays_to_file_refs(array) is array

    assert created_paths
    assert not created_paths[0].exists()


def test_local_endpoint_detection():
    assert is_local_endpoint("tcp://127.0.0.1:30000")
    assert is_local_endpoint("tcp://localhost:30000")
    assert is_local_endpoint("ipc:///tmp/sgl.sock")
    assert is_local_endpoint("inproc://scheduler")
    assert not is_local_endpoint("tcp://10.0.0.2:30000")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.uint8])
def test_spill_large_tensors_round_trips(monkeypatch, tmp_path, dtype):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    elements = (
        ipc_array._MIN_FILE_REF_BYTES // torch.empty((), dtype=dtype).element_size()
    )
    tensor = torch.arange(elements, dtype=torch.int64).to(dtype).reshape(2, -1)

    spilled = spill_large_arrays_to_file_refs([tensor])

    assert isinstance(spilled[0], TorchTensorFileRef)
    spilled_path = Path(spilled[0].ref.path)
    assert spilled_path.exists()

    materialized = materialize_file_refs(spilled)[0]

    assert materialized.dtype == tensor.dtype
    assert materialized.shape == tensor.shape
    assert torch.equal(materialized, tensor)
    assert not spilled_path.exists()


def test_non_contiguous_tensor_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    elements = ipc_array._MIN_FILE_REF_BYTES // 2
    tensor = torch.arange(elements, dtype=torch.float32).reshape(2, -1).T

    materialized = materialize_file_refs(spill_large_arrays_to_file_refs(tensor))

    assert torch.equal(materialized, tensor)


def test_small_tensors_are_kept_inline():
    tensor = torch.zeros(16)

    spilled = spill_large_arrays_to_file_refs((tensor,))

    assert spilled[0] is tensor


def test_tensor_spill_falls_back_inline_when_shm_is_full(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    tensor = torch.zeros(ipc_array._MIN_FILE_REF_BYTES, dtype=torch.uint8)

    def fail_save(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(np, "save", fail_save)

    assert spill_large_arrays_to_file_refs(tensor) is tensor


@pytest.mark.skipif(os.name != "posix", reason="POSIX unlink semantics")
def test_materialized_array_uses_mapping_and_remains_writable_after_unlink(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ipc_array, "_can_map_spilled_array", lambda path: True)
    path = tmp_path / "array.npy"
    source = np.arange(1024, dtype=np.float32)
    np.save(path, source)
    result = NumpyArrayFileRef(str(path)).materialize()
    assert type(result) is np.ndarray
    assert isinstance(result.base, np.memmap)
    assert result.base.mode == "r+"
    assert not path.exists()
    assert result.flags.writeable
    np.testing.assert_array_equal(result, source)
    result[0] = -1
    assert result[0] == -1
    assert source[0] == 0


def test_materialized_tensor_keeps_storage_alive_after_file_ref_is_gone(tmp_path):
    path = tmp_path / "tensor.npy"
    source = torch.arange(1024, dtype=torch.bfloat16)
    np.save(path, source.view(torch.uint8).numpy())
    result = TorchTensorFileRef(
        NumpyArrayFileRef(str(path)), "bfloat16", (1024,)
    ).materialize()
    assert not path.exists()
    assert torch.equal(result, source)
    result[0] = -1
    assert result[0] == -1
    assert source[0] == 0


def test_empty_array_round_trip(tmp_path):
    path = tmp_path / "empty.npy"
    source = np.empty((0, 3), dtype=np.float32)
    np.save(path, source)
    result = NumpyArrayFileRef(str(path)).materialize()
    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert not path.exists()


def test_invalid_array_is_rejected_and_spill_removed(tmp_path):
    path = tmp_path / "broken.npy"
    path.write_bytes(b"invalid array")
    with pytest.raises(ValueError):
        NumpyArrayFileRef(str(path)).materialize()
    assert not path.exists()


def test_retained_results_do_not_map_when_spill_space_is_low(monkeypatch, tmp_path):
    path = tmp_path / "array.npy"
    source = np.zeros(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)
    np.save(path, source)
    from types import SimpleNamespace

    monkeypatch.setattr(
        ipc_array.os, "statvfs", lambda path: SimpleNamespace(f_bavail=1, f_frsize=4096)
    )
    result = NumpyArrayFileRef(str(path)).materialize()
    assert result.flags.owndata
    assert not isinstance(result.base, np.memmap)
    assert not path.exists()
    np.testing.assert_array_equal(result, source)


def test_read_only_mapping_failure_falls_back_to_eager_load(monkeypatch, tmp_path):
    path = tmp_path / "array.npy"
    source = np.arange(128, dtype=np.float32)
    np.save(path, source)
    monkeypatch.setattr(ipc_array, "_can_map_spilled_array", lambda path: True)
    load = np.load

    def read_only(filename, **kwargs):
        if kwargs.get("mmap_mode") == "r+":
            raise PermissionError("read-only spill")
        return load(filename, **kwargs)

    monkeypatch.setattr(ipc_array.np, "load", read_only)
    result = NumpyArrayFileRef(str(path)).materialize()
    assert result.flags.owndata
    assert not path.exists()
    np.testing.assert_array_equal(result, source)


def test_map_gate_requires_two_payloads_of_available_space(monkeypatch, tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "payload.npy"
    path.touch()
    size = ipc_array._MIN_FILE_REF_BYTES
    fake_os = SimpleNamespace(
        name="posix",
        stat=lambda path: SimpleNamespace(st_size=size),
        path=os.path,
        statvfs=lambda path: SimpleNamespace(f_bavail=2 * size - 1, f_frsize=1),
    )
    monkeypatch.setattr(ipc_array, "os", fake_os)
    monkeypatch.setattr(ipc_array, "_has_mapping_fd_headroom", lambda: True)
    assert not ipc_array._can_map_spilled_array(str(path))
    fake_os.statvfs = lambda path: SimpleNamespace(f_bavail=2 * size, f_frsize=1)
    assert ipc_array._can_map_spilled_array(str(path))


@pytest.mark.skipif(os.name != "posix", reason="POSIX FD limits")
def test_low_fd_limit_can_retain_many_results(tmp_path):
    import subprocess
    import sys

    code = """
import importlib.util, pathlib, resource, sys
import numpy as np
name = 'ipc_fd_limit_candidate'
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules[name] = m
spec.loader.exec_module(m)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64 if soft == resource.RLIM_INFINITY else min(64, soft), hard))
m._MIN_FILE_REF_BYTES = 0
arrays = []
for index in range(100):
    path = pathlib.Path(sys.argv[2]) / (str(index) + '.npy')
    np.save(path, np.arange(16, dtype=np.float32) + index)
    arrays.append(m.NumpyArrayFileRef(str(path)).materialize())
    assert not path.exists()
assert all(array[-1] == index + 15 for index, array in enumerate(arrays))
assert any(array.flags.owndata for array in arrays)
print('100 retained results with FD limit 64')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, ipc_array.__file__, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_small_arrays_do_not_consume_mapping_fds(tmp_path):
    path = tmp_path / "small.npy"
    np.save(path, np.arange(16, dtype=np.uint8))
    assert not ipc_array._can_map_spilled_array(str(path))


@pytest.mark.skipif(os.name != "posix", reason="POSIX unlink semantics")
def test_mapping_is_released_only_after_the_last_tensor_view(monkeypatch, tmp_path):
    import gc
    import weakref

    monkeypatch.setattr(ipc_array, "_can_map_spilled_array", lambda path: True)
    path = tmp_path / "tensor-view.npy"
    source = np.arange(128, dtype=np.float32)
    np.save(path, source)
    array = NumpyArrayFileRef(str(path)).materialize()
    mapping = weakref.ref(array.base)
    tensor = torch.from_numpy(array)
    view = tensor[1:]
    del array, tensor
    gc.collect()
    assert mapping() is not None
    assert view[0] == 1
    view[0] = -1
    assert source[1] == 1
    del view
    gc.collect()
    assert mapping() is None
