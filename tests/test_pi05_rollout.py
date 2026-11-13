"""Rollout regressions using a mock RTDE controller, fake serial I/O and localhost policy."""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('UR5_REPO', str(ROOT)))
sys.path.insert(0, os.environ.get('PI05_DEPLOY_ROOT', str(ROOT)))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_rollout import RolloutConfig, UR5Robot, at_home, check_state, control_loop, move_home, resolve_config
from rollout_io import ActionChunk, PolicyWorker, validate_actions, TrajectoryExecutor, canonicalize_policy_state
from ur5_teleop.backend import RobotState, BackendError
from ur5_teleop.home import HOME_JOINTS_DEG, build_movej_program
from ur5_teleop.ur_client import RTDEBackend
from ur5_teleop.gripper import Gripper, GripperError
from mock_rtde_server import MockRTDEServer
from test_gripper import FakeTransport, status_response


class FakeRobot:
    executing = True
    gripper = None
    args = SimpleNamespace(mode='record')

    def __init__(self):
        self.sent = []
        self.held = False
        self.delay = 0
        self.trajectory = SimpleNamespace(action=np.zeros(7), complete=True)

    def get_state(self):
        return RobotState((0., 0., .4, 0., 3.14, 0.), (0.,) * 6, 7, 1, time.monotonic(), 1., 2)

    def get_observation(self):
        return {'test': np.zeros(1)}, time.monotonic()

    def send_action(self, action, state, dt):
        self.trajectory.action = action
        self.sent.append(time.monotonic())
        time.sleep(self.delay)
        return action.tolist()

    def hold(self):
        self.held = True


class RolloutTest(unittest.TestCase):
    def test_policy_state_preserves_rotation_across_equivalent_branches(self):
        from ur5_teleop.frames import rotvec_to_matrix
        for rotation in ([0, -np.pi, 0], [.416751, 2.976688, .045675],
                         [-.45432866, -3.2450902, -.04979343], [0, 0, 0]):
            state = np.array([.1, .2, .3, *rotation, *range(6), 1.], dtype=np.float32)
            original = state.copy()
            result = canonicalize_policy_state(state)
            self.assertGreaterEqual(result[4], 0)
            np.testing.assert_allclose(rotvec_to_matrix(result[3:6]), rotvec_to_matrix(state[3:6]), atol=1e-6)
            np.testing.assert_equal(result[:3], state[:3])
            np.testing.assert_equal(result[6:], state[6:])
            np.testing.assert_equal(state, original)
            np.testing.assert_allclose(canonicalize_policy_state(result), result, atol=1e-6)
        with self.assertRaises(ValueError):
            canonicalize_policy_state(np.full(13, np.nan))

    def test_waypoint_commits_pose_and_gripper_on_one_timeline(self):
        from ur5_teleop.safety import pose_error
        executor = TrajectoryExecutor()
        start = [0, 0, .4, 0, np.deg2rad(179), 0]
        close = [.003, 0, .4, 0, np.deg2rad(-179), 0, 0]
        first = executor.step(close, start, .7, .001, np.deg2rad(1))
        self.assertAlmostEqual(first[0], .001)
        self.assertEqual(first[6], .7)
        self.assertFalse(executor.complete)
        self.assertLess(pose_error(first[:6], start)[1], np.deg2rad(1))
        # New inference must not replace a partly executed pose/gripper pair.
        later = [.1, 0, .4, 0, 0, 0, 1]
        second = executor.step(later, start, .7, .001, np.deg2rad(1))
        self.assertAlmostEqual(second[0], .002)
        self.assertEqual(second[6], .7)
        final = executor.step(later, start, .7, .001, np.deg2rad(1))
        self.assertFalse(executor.complete)
        self.assertEqual(final[6], 0)
        self.assertLess(pose_error(final[:6], close[:6])[1], 1e-6)
        executor.step(later, start, .7, .001, np.deg2rad(1), gripper_done=False)
        self.assertFalse(executor.complete)
        executor.step(later, start, .7, .001, np.deg2rad(1), gripper_done=True)
        self.assertTrue(executor.complete)
        # Once complete, adopt the latest candidate, preserving the prior command pose.
        fourth = executor.step(later, start, .7, .001, np.deg2rad(1))
        self.assertLessEqual(np.linalg.norm(fourth[:3] - final[:3]), .001 + 1e-12)
        self.assertEqual(fourth[6], 0)

    def test_waypoint_handles_stationary_gripper_and_does_not_restart_from_feedback(self):
        executor = TrajectoryExecutor()
        start = [0, 0, .4, 0, 0, 0]
        same = [*start, 0]
        self.assertEqual(executor.step(same, start, 1, .001, .01)[6], 0)
        executor.step(same, start, 0, .001, .01, gripper_done=True)
        move = [.1, 0, .4, 0, 0, 0, 1]
        for _ in range(10):
            target = executor.step(move, start, 0, .001, .01)
        self.assertAlmostEqual(target[0], .01)
        self.assertEqual(target[6], 0)

    def test_gripper_completion_requires_delivery_and_new_feedback(self):
        grip = Gripper('fake', transport=FakeTransport())
        grip.healthy = True
        grip.grip_state = 1
        grip.last_read_at = 10
        self.assertFalse(grip.motion_complete(0))
        grip._last_delivered = 0
        self.assertFalse(grip.motion_complete(0))
        grip.last_read_at = 12
        grip.grip_state = 0
        grip._motion_status = (0, 0)
        self.assertFalse(grip.motion_complete(0))
        for status in (1, 2, 3):
            grip.grip_state = status
            grip._motion_status = (0, status)
            self.assertTrue(grip.motion_complete(0))
        self.assertFalse(grip.motion_complete(1000))

    def test_robot_observation_normalizes_only_policy_orientation(self):
        robot = object.__new__(UR5Robot)
        state = RobotState((0., 0., .4, 0., -3.14, 0.), (0.,) * 6, 7, 1, time.monotonic(), 1., 2)
        stamp = time.monotonic()
        robot.get_state = lambda: state
        robot.cameras = [SimpleNamespace(snapshot=lambda _: (np.zeros((2, 2, 3), dtype=np.uint8), stamp))] * 2
        robot.gripper = SimpleNamespace(position_raw=500)
        robot.cfg = RolloutConfig()
        robot.args = SimpleNamespace(prompt='test')
        obs, _ = robot.get_observation()
        np.testing.assert_allclose(obs['observation/state'][6:12], state.joints)
        self.assertAlmostEqual(obs['observation/state'][12], .5)
        self.assertGreater(obs['observation/state'][4], 0)
        self.assertEqual(state.tcp_pose, (0., 0., .4, 0., -3.14, 0.))

    def test_task_config_defaults_and_cli_overrides(self):
        args = SimpleNamespace(**{key: None for key in RolloutConfig.__dataclass_fields__})
        cfg = resolve_config(args, {'inference': {'max_step_mm': .5, 'duration': 30}})
        self.assertEqual(cfg.max_step_mm, .5)
        self.assertEqual(cfg.duration, 30)
        args.duration = 60
        self.assertEqual(resolve_config(args, {'inference': {'duration': 30}}).duration, 60)
        with self.assertRaises(ValueError):
            resolve_config(args, {'inference': {'typo': 1}})

    def test_run_entry_orchestrates_without_hardware(self):
        with tempfile.TemporaryDirectory(prefix='pi05 run ') as directory:
            root = Path(directory)
            (root/'start_inference.sh').write_text('exit 0\n')
            (root/'run.sh').write_text('printf "%s\\n" "$@"\n')
            entry = Path(os.environ.get('PI05_DEPLOY_ROOT', ROOT))/'run_robot.sh'
            result = subprocess.run(['bash', str(entry), 'run', '--duration', '7'],
                                    env={**os.environ, 'PI05_DEPLOY_ROOT': directory},
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(root/'policy_rollout.py'), '--mode',
                             'record', '--home-at-start', '--execute', '--i-understand', '--duration', '7'])

    def test_entry_help_does_not_start_model_or_move_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'start_inference.sh').write_text('exit 99\n')
            (root/'run.sh').write_text('printf "%s\\n" "$@"\n')
            entry = Path(os.environ.get('PI05_DEPLOY_ROOT', ROOT))/'run_robot.sh'
            result = subprocess.run(['bash', str(entry), 'run', '--help'],
                                    env={**os.environ, 'PI05_DEPLOY_ROOT': directory},
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(root/'policy_rollout.py'), '--help'])

    def test_home_is_joint_degrees_and_preserves_turns(self):
        self.assertEqual(HOME_JOINTS_DEG, (-180, -90, -90, -90, 90, -90))
        q = np.deg2rad(HOME_JOINTS_DEG)
        self.assertTrue(at_home(q, HOME_JOINTS_DEG, 1))
        q[0] += 2 * np.pi
        self.assertFalse(at_home(q, HOME_JOINTS_DEG, 1))
        self.assertIn('movej([-3.14159265, -1.57079633', build_movej_program(HOME_JOINTS_DEG))

    def test_latency_skips_old_actions_and_expiry_fails(self):
        chunk = ActionChunk(np.arange(350).reshape(50, 7), 10., 500., 1)
        action, index = chunk.action_at(10.5, 30)
        self.assertEqual(index, 15)
        np.testing.assert_equal(action, chunk.actions[15])
        with self.assertRaises(RuntimeError):
            chunk.action_at(12, 30)

    def test_invalid_model_outputs_rejected(self):
        for action in [np.zeros((7, 50)), np.zeros(350), np.full((50, 7), np.nan)]:
            with self.assertRaises(ValueError):
                validate_actions(action)
        with self.assertRaises(ValueError):
            RolloutConfig(fps=0)
        state = FakeRobot().get_state()
        with self.assertRaises(RuntimeError):
            check_state(RobotState(state.tcp_pose, state.joints, 7, 3, time.monotonic(), 1, 2), executing=True)

    def test_overrun_skips_ticks_instead_of_catching_up(self):
        robot = FakeRobot()
        robot.delay = .06
        chunk = ActionChunk(np.zeros((50, 7)), time.monotonic(), 0, 1)
        policy = SimpleNamespace(latest=lambda: chunk)
        summary = control_loop(robot, policy, RolloutConfig(duration=.36), threading.Event(), io.StringIO())
        self.assertTrue(robot.held)
        self.assertGreater(summary['overruns'], 0)
        self.assertTrue(all(b - a > .08 for a, b in zip(robot.sent, robot.sent[1:])))

    def test_policy_failure_holds_robot(self):
        def fail():
            raise TimeoutError('policy timed out')
        robot = FakeRobot()
        with self.assertRaises(TimeoutError):
            control_loop(robot, SimpleNamespace(latest=fail), RolloutConfig(), threading.Event(), io.StringIO())
        self.assertTrue(robot.held)

    def test_action_limits_and_workspace_rejection_use_real_adapter(self):
        robot = object.__new__(UR5Robot)
        robot.cfg = RolloutConfig()
        robot.executing = False
        robot.trajectory = TrajectoryExecutor()
        robot.gripper = SimpleNamespace(position_raw=1000)
        robot.hardware = {'safety': {'workspace_min': [-.8, -.8, 0], 'workspace_max': [.8, .8, 1.2]}}
        state = FakeRobot().get_state()
        target = robot.send_action([.5, 0, .4, 0, 0, 0, .9], state, 1 / 30)
        self.assertAlmostEqual(target[0], .002)
        from ur5_teleop.safety import pose_error
        self.assertLessEqual(pose_error(target[:6], state.tcp_pose)[1], np.deg2rad(2) + 1e-6)
        self.assertEqual(target[-1], 1)
        with self.assertRaises(ValueError):
            robot.send_action([2, 0, .4, 0, 0, 0, 1], state, 1 / 30)

    def test_read_only_rtde_does_not_claim_or_write_input(self):
        server = MockRTDEServer(host='127.0.0.1', port=0).start()
        try:
            backend = RTDEBackend('127.0.0.1', port=server.port, read_only=True)
            try:
                backend.connect()
                deadline = time.monotonic() + 1
                while True:
                    try:
                        backend.read_state()
                        break
                    except BackendError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.01)
                time.sleep(.15)
                with self.assertRaises(BackendError):
                    backend.send_sets((0,) * 6, True)
            finally:
                backend.close()
            self.assertEqual(server.input_recipes, {})
            self.assertEqual(server.input_count, 0)
        finally:
            server.stop()

    def test_read_only_gripper_and_acknowledged_stop(self):
        transport = FakeTransport([status_response(position=250)] * 100)
        gripper = Gripper(transport=transport, read_only=True)
        try:
            gripper.connect()
            with self.assertRaises(GripperError):
                gripper.set_position(1000)
        finally:
            gripper.close()
        self.assertTrue(all(frame[1] == 3 for frame in transport.written))
        transport = FakeTransport([status_response(position=250)] * 100)
        gripper = Gripper(transport=transport)
        try:
            gripper.connect()
            gripper.set_position(1000)
            time.sleep(.06)
            gripper.stop_motion()
            writes = [frame for frame in transport.written if frame[1] == 6 and frame[2:4] == b'\x01\x03']
            self.assertEqual(int.from_bytes(writes[-1][4:6], 'big'), 250)
        finally:
            gripper.close()

    def test_stale_state_does_not_block_idle_or_gripper_stop(self):
        calls = []
        robot = object.__new__(UR5Robot)
        robot.executing = True
        robot.backend = SimpleNamespace(send_sets=lambda pose, servo: calls.append(('idle', servo)))
        robot.gripper = SimpleNamespace(stop_motion=lambda: calls.append(('gripper',)))
        robot.hold()
        self.assertEqual(calls, [('idle', False), ('gripper',)])

    def test_home_interruption_stops_movej_program(self):
        stop = threading.Event()
        stop.set()
        dashboard = MagicMock()
        dashboard.command.return_value = 'NORMAL'
        with patch('ur5_teleop.ur_client.RTDEBackend') as backend_factory, \
                patch('ur5_teleop.ur_client.DashboardClient', return_value=dashboard), \
                patch('ur5_teleop.ur_client.open_script_channel'), \
                patch('ur5_teleop.ur_client.send_program') as send:
            with self.assertRaises(InterruptedError):
                move_home(SimpleNamespace(robot_ip='127.0.0.1'),
                          {'home': {'joints_deg': HOME_JOINTS_DEG}}, RolloutConfig(), stop)
            self.assertNotIn('read_only', backend_factory.call_args.kwargs)
            self.assertIn('movej([-3.14159265', send.call_args.args[1])
        self.assertEqual(sum(call.args == ('stop',) for call in dashboard.command.call_args_list), 2)

    def test_home_reuses_rollout_connection_without_releasing_registers(self):
        stop = threading.Event()
        stop.set()
        backend = MagicMock()
        dashboard = MagicMock()
        dashboard.command.return_value = 'NORMAL'
        with patch('ur5_teleop.ur_client.RTDEBackend') as factory, \
                patch('ur5_teleop.ur_client.DashboardClient', return_value=dashboard), \
                patch('ur5_teleop.ur_client.open_script_channel'), \
                patch('ur5_teleop.ur_client.send_program'):
            with self.assertRaises(InterruptedError):
                move_home(SimpleNamespace(robot_ip='127.0.0.1'),
                          {'home': {'joints_deg': HOME_JOINTS_DEG}}, RolloutConfig(), stop, backend=backend)
        factory.assert_not_called()
        backend.connect.assert_not_called()
        backend.close.assert_not_called()


try:
    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed
    from openpi_client import msgpack_numpy
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False


@unittest.skipUnless(WS_AVAILABLE, 'Run WebSocket integration tests in the AGX inference environment')
class WebsocketRolloutTest(unittest.TestCase):
    def run_server(self, delay, timeout, duration, invalid=False):
        def handler(connection):
            packer = msgpack_numpy.Packer()
            try:
                connection.send(packer.pack({'action_horizon': 50, 'num_steps': 10}))
                while True:
                    connection.recv()
                    time.sleep(delay)
                    actions = np.full((50, 7), np.nan if invalid else 0.)
                    connection.send(packer.pack({'actions': actions}))
            except ConnectionClosed:
                # Test clients deliberately close while the mock server is sleeping.
                return
        server = serve(handler, '127.0.0.1', 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        robot = FakeRobot()
        worker = PolicyWorker(robot, '127.0.0.1', server.socket.getsockname()[1], timeout, .4)
        worker.start()
        try:
            summary = control_loop(robot, worker, RolloutConfig(duration=duration, timeout=timeout),
                                   threading.Event(), io.StringIO())
            return robot, summary
        finally:
            worker.close()
            server.shutdown()
            thread.join(timeout=1)
            self.assertTrue(robot.held)
            self.assertFalse(worker.thread.is_alive())

    def test_half_second_inference_keeps_control_at_30hz(self):
        robot, summary = self.run_server(.48, 2, 1.3)
        self.assertGreaterEqual(summary['ticks'], 33)
        self.assertGreater(len(robot.sent), 15)
        gaps = [b - a for a, b in zip(robot.sent, robot.sent[1:])]
        self.assertGreater(min(gaps), .02)
        self.assertLess(max(gaps), .1)

    def test_unresponsive_or_invalid_policy_fails_and_holds(self):
        for delay, invalid in [(1, False), (.01, True)]:
            with self.subTest(delay=delay, invalid=invalid), self.assertRaises((RuntimeError, TimeoutError)):
                self.run_server(delay, .2, 1, invalid)


if __name__ == '__main__':
    unittest.main()
