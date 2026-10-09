import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime import launch_server as ls
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.control_requests import ShutdownReq


class _FakeProcess:
    name = "fake-worker"

    def __init__(self, *, exit_on_join: bool = False):
        self.alive = True
        self.exit_on_join = exit_on_join
        self.join_timeouts = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self.exit_on_join:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False


class TestLaunchServerShutdown(unittest.TestCase):
    def test_monolithic_shutdown_requests_scheduler_then_forces_worker(self):
        process = _FakeProcess()
        server_args = SimpleNamespace(disagg_role=RoleType.MONOLITHIC)
        client = Mock()

        with patch.object(ls, "SchedulerClient", return_value=client):
            ls.shutdown_scheduler_processes(server_args, [process])

        client.initialize.assert_called_once_with(server_args)
        request = client.forward.call_args.args[0]
        self.assertIsInstance(request, ShutdownReq)
        self.assertEqual(client.forward.call_args.kwargs, {"timeout_ms": 5000})
        client.close.assert_called_once_with()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertAlmostEqual(process.join_timeouts[0], 10, delta=0.1)
        self.assertAlmostEqual(process.join_timeouts[1], 1, delta=0.1)
        self.assertAlmostEqual(process.join_timeouts[2], 1, delta=0.1)

    def test_scheduler_shutdown_error_still_forces_worker(self):
        process = _FakeProcess()
        server_args = SimpleNamespace(disagg_role=RoleType.MONOLITHIC)
        client = Mock()
        client.forward.side_effect = TimeoutError("blocked")

        with patch.object(ls, "SchedulerClient", return_value=client):
            ls.shutdown_scheduler_processes(server_args, [process])

        client.close.assert_called_once_with()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_disagg_role_does_not_send_monolithic_shutdown_request(self):
        process = _FakeProcess(exit_on_join=True)
        server_args = SimpleNamespace(disagg_role=RoleType.ENCODER)

        with patch.object(ls, "SchedulerClient") as scheduler_client:
            ls.shutdown_scheduler_processes(server_args, [process])

        scheduler_client.assert_not_called()
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)


if __name__ == "__main__":
    unittest.main()


class TestSchedulerInitFailure(unittest.TestCase):
    def test_init_error_reaches_launcher_through_ready_pipe(self):
        """A rank failing in Scheduler.__init__ used to close its pipe silently,
        so the launcher raised a bare EOFError without the reason."""
        import multiprocessing as mp

        from sglang.multimodal_gen.runtime.managers import gpu_worker

        platform = Mock()
        platform.is_cuda.return_value = False
        platform.is_musa.return_value = False
        reader, writer = mp.Pipe(duplex=False)
        with (
            patch.object(gpu_worker, "current_platform", platform),
            patch.object(gpu_worker, "initialize_current_platform"),
            patch.object(gpu_worker, "kill_itself_when_parent_died"),
            patch.object(gpu_worker, "configure_logger"),
            patch.object(gpu_worker, "globally_suppress_loggers"),
            patch.object(gpu_worker, "init_diffusion_tracing"),
            patch.object(gpu_worker, "_device_initialized", return_value=False),
            # The worker's exit path tears down an initialized process group;
            # keep the one other tests in this process created.
            patch.object(torch.distributed, "is_initialized", return_value=False),
            patch.object(gpu_worker.PortArgs, "from_server_args"),
            patch(
                "sglang.multimodal_gen.runtime.managers.scheduler.Scheduler",
                side_effect=ValueError("unsupported checkpoint"),
            ),
            self.assertRaisesRegex(ValueError, "unsupported checkpoint"),
        ):
            gpu_worker.run_scheduler_process(
                local_rank=0, rank=0, server_args=Mock(), pipe_writer=writer
            )
        writer.close()
        self.assertTrue(reader.poll(1))
        self.assertEqual(
            reader.recv(),
            {"status": "error", "error": "ValueError: unsupported checkpoint"},
        )
