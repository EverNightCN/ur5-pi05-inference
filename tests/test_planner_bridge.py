"""Exercise planner cancellation through the production control loop with fake hardware."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner_bridge import Pi05Session, open_session
from policy_rollout import RolloutConfig
from rollout_io import ActionChunk
from test_pi05_rollout import FakeRobot


class Robot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(mode='record', prompt='original task', host='127.0.0.1', port=8000)
        self.gripper = SimpleNamespace(position_raw=1000, grip_state=1)
        self.sent_event = threading.Event()
        self.closed = False

    def get_observation(self):
        return {'observation/image': np.zeros((8, 8, 3), dtype=np.uint8),
                'observation/wrist_image': np.ones((8, 8, 3), dtype=np.uint8),
                'observation/state': np.zeros(13, dtype=np.float32),
                'prompt': self.args.prompt}, time.monotonic()

    def send_action(self, *arguments):
        result = super().send_action(*arguments)
        self.sent_event.set()
        return result


class Worker:
    instances = []

    def __init__(self, robot, *arguments):
        self.robot = robot
        self.closed = False
        self.close_started = threading.Event()
        self.close_allowed = threading.Event()
        self.close_allowed.set()
        self.instances.append(self)

    def start(self):
        self.observation, _ = self.robot.get_observation()
        self.trajectory = self.robot.trajectory

    def latest(self):
        return ActionChunk(np.zeros((50, 7)), time.monotonic(), 1, 1)

    def close(self):
        self.close_started.set()
        if not self.close_allowed.wait(2):
            raise TimeoutError('Test failed to release worker shutdown')
        self.closed = True


class PlannerBridgeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.robot = Robot()
        self.session = Pi05Session(self.robot, RolloutConfig(fps=100), self.directory.name)
        Worker.instances = []
        self.worker_patch = patch('planner_bridge.PolicyWorker', Worker)
        self.worker_patch.start()
        self.addCleanup(self.worker_patch.stop)

    def test_observation_contains_images_and_measured_feedback(self):
        result = self.session.observe()
        self.assertEqual(result['observation']['observation/state'].shape, (13,))
        self.assertEqual(result['observation']['observation/image'].dtype, np.uint8)
        self.assertEqual(result['feedback']['gripper_position_raw'], 1000)
        self.assertEqual(result['feedback']['runtime_state'], 2)

    def test_action_uses_instruction_and_reports_interval_completion(self):
        result = self.session.act('push the cardboard box aside', .04)
        self.assertEqual(result['execution_status'], 'completed')
        self.assertTrue(result['execution_done'])
        self.assertIsNone(result['task_success'])
        self.assertTrue(self.robot.held)
        self.assertTrue(Worker.instances[0].closed)
        self.assertEqual(Worker.instances[0].observation['prompt'], 'push the cardboard box aside')
        self.assertEqual(self.robot.args.prompt, 'original task')
        self.assertTrue(Path(result['summary']['log_path']).is_file())

    def test_pause_acknowledges_hold_and_resumes_with_fresh_worker_and_trajectory(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.session.act, 'first instruction', 10)
            self.assertTrue(self.robot.sent_event.wait(1))
            self.session.pause()
            state = self.session.wait(2)
            result = future.result(2)
        self.assertEqual(state['execution_status'], 'paused')
        self.assertEqual(result['execution_status'], 'paused')
        count = len(self.robot.sent)
        time.sleep(.03)
        self.assertEqual(len(self.robot.sent), count)
        self.assertTrue(self.robot.held)
        self.session.act('second instruction', .03)
        self.assertEqual(Worker.instances[1].observation['prompt'], 'second instruction')
        self.assertIsNot(Worker.instances[0].trajectory, Worker.instances[1].trajectory)

    def test_wait_does_not_acknowledge_before_worker_shutdown(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.session.act, 'instruction', 10)
            self.assertTrue(self.robot.sent_event.wait(1))
            worker = Worker.instances[0]
            worker.close_allowed.clear()
            self.session.pause()
            try:
                self.assertTrue(worker.close_started.wait(1))
                with self.assertRaises(TimeoutError):
                    self.session.wait(.02)
            finally:
                worker.close_allowed.set()
            self.assertEqual(future.result(2)['execution_status'], 'paused')

    def test_stop_is_terminal_and_dominates_later_pause(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.session.act, 'instruction', 10)
            self.assertTrue(self.robot.sent_event.wait(1))
            self.session.stop()
            self.session.pause()
            self.assertEqual(future.result(2)['execution_status'], 'stopped')
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            self.session.act('new instruction', .03)

    def test_only_one_execution_owner_is_allowed(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.session.act, 'instruction', 10)
            self.assertTrue(self.robot.sent_event.wait(1))
            try:
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    self.session.act('another instruction', .03)
            finally:
                self.session.stop()
            future.result(2)

    def test_inference_errors_propagate_and_hold(self):
        with patch.object(Worker, 'latest', side_effect=RuntimeError('policy transport failed')):
            with self.assertRaisesRegex(RuntimeError, 'policy transport failed'):
                self.session.act('instruction', .04)
        state = self.session.wait(1)
        self.assertEqual(state['execution_status'], 'failed')
        self.assertIn('policy transport failed', state['error'])
        self.assertTrue(self.robot.held)
        self.assertTrue(Worker.instances[0].closed)
        self.assertEqual(self.robot.args.prompt, 'original task')

    def test_pause_and_stop_preserve_a_failed_execution_result(self):
        with patch.object(Worker, 'latest', side_effect=RuntimeError('serial or policy failure')):
            with self.assertRaises(RuntimeError):
                self.session.act('instruction', .04)
        self.session.pause()
        self.session.stop()
        state = self.session.wait(1)
        self.assertEqual(state['execution_status'], 'failed')
        self.assertIn('serial or policy failure', state['error'])
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            self.session.act('new instruction', .03)

    def test_invalid_action_arguments_do_not_start_a_worker(self):
        for instruction, seconds in [('', 1), ('task', 0), ('task', float('nan'))]:
            with self.subTest(instruction=instruction, seconds=seconds), self.assertRaises(ValueError):
                self.session.act(instruction, seconds)
        self.assertEqual(Worker.instances, [])
        self.assertTrue(self.session.status()['execution_done'])

    def test_session_context_uses_one_hardware_owner_and_closes_it(self):
        args = SimpleNamespace(mode='record', log_dir=self.directory.name)
        cfg = self.session.cfg
        def connect(arguments, config, hardware, stack):
            stack.callback(lambda: setattr(self.robot, 'closed', True))
            return self.robot
        with patch('planner_bridge.load_rollout_config', return_value=({}, cfg)), \
                patch('planner_bridge.UR5Robot', side_effect=connect) as factory, \
                patch('planner_bridge.prepare_execution') as prepare:
            with open_session(args) as session:
                session.act('instruction', .03)
                self.assertFalse(self.robot.closed)
            self.assertTrue(self.robot.closed)
            self.assertEqual(session.status()['execution_status'], 'closed')
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                session.observe()
        factory.assert_called_once()
        prepare.assert_called_once()

try:
    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed
    from openpi_client import msgpack_numpy
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False


@unittest.skipUnless(WS_AVAILABLE, 'WebSocket protocol dependencies are installed in the inference environment')
class PlannerProtocolTest(unittest.TestCase):
    def test_real_worker_sends_language_images_and_state_and_holds_after_segment(self):
        received = []
        def handler(connection):
            packer = msgpack_numpy.Packer()
            connection.send(packer.pack({'action_horizon': 50, 'num_steps': 10}))
            try:
                while True:
                    received.append(msgpack_numpy.unpackb(connection.recv()))
                    connection.send(packer.pack({'actions': np.zeros((50, 7), dtype=np.float32)}))
            except ConnectionClosed:
                return
        server = serve(handler, '127.0.0.1', 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        robot = Robot()
        robot.args.port = server.socket.getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as temporary:
                session = Pi05Session(robot, RolloutConfig(fps=100), temporary)
                result = session.act('language supplied by the planner', .2)
            self.assertEqual(result['execution_status'], 'completed')
            self.assertGreater(result['summary']['sent'], 0)
            self.assertTrue(robot.held)
            self.assertGreater(len(received), 0)
            self.assertEqual(received[0]['prompt'], 'language supplied by the planner')
            self.assertEqual(received[0]['observation/image'].dtype, np.uint8)
            self.assertEqual(received[0]['observation/wrist_image'].shape, (8, 8, 3))
            self.assertEqual(received[0]['observation/state'].shape, (13,))
        finally:
            server.shutdown()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
