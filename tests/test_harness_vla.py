"""Harness integration checks use fake hardware and the real RPent agent loop."""
from __future__ import annotations

from importlib.util import find_spec
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(os.environ.get('UR5_REPO', str(Path(__file__).resolve().parents[1])))
DEPLOY = Path(os.environ.get('PI05_DEPLOY_ROOT', str(ROOT)))
sys.path[:0] = [str(ROOT), str(DEPLOY)]
from policy_rollout import RolloutConfig, UR5Robot, prepare_execution
from rollout_io import ActionChunk, TrajectoryExecutor
from ur5_teleop.backend import RobotState

HAS_RPENT = find_spec('rpent') is not None
if HAS_RPENT:
    from harness_tools import UR5HarnessToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager


class FakeRobot(UR5Robot):
    def __init__(self, args=None, cfg=None, hardware=None, stack=None):
        self.args = args or SimpleNamespace(mode='dry-run', prompt='trained task', host='127.0.0.1', port=8000)
        self.cfg = cfg or RolloutConfig(fps=100, max_step_mm=1, max_rot_step_deg=1)
        self.executing = self.args.mode == 'record'
        self.hardware = {'safety': {'workspace_min': [-1, -1, 0], 'workspace_max': [1, 1, 1]}}
        self.gripper = SimpleNamespace(position_raw=650, grip_state=1, motion_complete=MagicMock(return_value=True),
                                       set_position=MagicMock(), stop_motion=MagicMock())
        self.backend = SimpleNamespace(send_sets=MagicMock())
        self.trajectory = TrajectoryExecutor()
        self.holds = []

    def get_state(self):
        return RobotState((0, 0, .4, 0, 3.14, 0), (0,) * 6, 7, 1, time.monotonic(), 1, 2)

    def get_observation(self):
        physical = self.get_state()
        return {'observation/state': np.asarray([*physical.tcp_pose, *physical.joints, .65], np.float32),
                'observation/image': np.full((48, 64, 3), 80, dtype=np.uint8),
                'observation/wrist_image': np.full((48, 64, 3), 120, dtype=np.uint8),
                'prompt': self.args.prompt}, time.monotonic()

    def hold(self, *, stop_gripper=True):
        self.holds.append(stop_gripper)
        super().hold(stop_gripper=stop_gripper)


@unittest.skipUnless(HAS_RPENT, 'RPent is required in the dedicated Harness environment')
class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.robot = FakeRobot()
        self.toolkit = UR5HarnessToolkit(robot=self.robot, cfg=self.robot.cfg, output_dir=self.root,
                                         memory=MemoryManager(self.root / 'memory'),
                                         dashboard_events=NullDashboardEventSink())
        self.addCleanup(self.toolkit.close)

    def test_private_provider_config_preserves_explicit_environment(self):
        import harness_vla
        env_file = self.root / 'provider.env'
        env_file.write_text('OPENAI_API_KEY=file-test-token\n'
                            'HARNESSVLA_MODEL=openai-responses:gpt-6.1-sol\n'
                            'HARNESSVLA_BASE_URL=https://api.openai.com/v1\n'
                            'HARNESSVLA_REASONING_EFFORT=low\n')
        environment = {'HARNESSVLA_ENV_FILE': str(env_file), 'OPENAI_API_KEY': 'shell-test-token'}
        output = io.StringIO()
        with patch.dict(os.environ, environment, clear=True), \
             patch.object(sys, 'argv', ['harness_vla.py', '--check', '--ur5-repo', str(ROOT), '--config', str(ROOT / 'config/ur5.example.toml')]), \
             redirect_stdout(output):
            self.assertEqual(harness_vla.main(), 0)
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'shell-test-token')
        report = json.loads(output.getvalue())
        self.assertEqual(report['model'], 'openai-responses:gpt-6.1-sol')
        self.assertEqual(report['reasoning_effort'], 'low')
        self.assertNotIn('test-token', output.getvalue())

    def test_shared_startup_reuses_the_connected_rtde_owner(self):
        home = [-180., -90., -90., -90., 90., -90.]
        state = RobotState((0, 0, .4, 0, 3.14, 0), tuple(np.deg2rad(home)),
                           7, 1, time.monotonic(), 1, 2)
        robot = SimpleNamespace(executing=True, backend=MagicMock())
        robot.backend.read_state.return_value = state
        args = SimpleNamespace(home_at_start=True)
        hardware = {'home': {'joints_deg': home}}
        stop = threading.Event()
        with patch('policy_rollout.move_home') as home_move, \
             patch('policy_rollout.install_servo_program') as servo:
            prepare_execution(robot, args, hardware, self.robot.cfg, stop)
        home_move.assert_called_once_with(args, hardware, self.robot.cfg, stop, backend=robot.backend)
        servo.assert_called_once_with(args, hardware, stop)

    def test_parallel_observation_records_remain_consistent(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: self.toolkit.execute_tool('observe', {}), range(6)))
        self.assertTrue(all(not result.is_error and len(result.images) == 2 for result in results))
        self.assertEqual(len(json.loads((self.root / 'states.json').read_text())['steps']), 7)

    def test_pose_tools_preserve_a_partially_closed_gripper(self):
        self.robot.executing = True
        result = self.toolkit.execute_tool('move_delta', dict(dx_mm=3., dy_mm=0., dz_mm=0.))
        self.assertFalse(result.is_error, result.data)
        np.testing.assert_allclose(self.robot.trajectory.target[:3], [.003, 0, .4], atol=1e-7)
        self.assertAlmostEqual(self.robot.trajectory.target[6], .65)
        self.robot.gripper.set_position.assert_not_called()
        self.robot.gripper.stop_motion.assert_not_called()
        self.assertEqual(len(result.images), 2)
        result = self.toolkit.execute_tool('rotate_delta', dict(rx_deg=0., ry_deg=0., rz_deg=2.))
        self.assertFalse(result.is_error, result.data)
        self.robot.gripper.set_position.assert_not_called()

    def test_gripper_tool_uses_real_feedback_progress(self):
        self.robot.executing = True
        result = self.toolkit.execute_tool('set_gripper', {'position': 'closed'})
        self.assertFalse(result.is_error, result.data)
        self.robot.gripper.set_position.assert_called_with(0)
        self.robot.gripper.motion_complete.assert_called_with(0)
        self.assertTrue(self.robot.trajectory.complete)

    def test_cancel_drains_an_active_motion(self):
        self.robot.executing = True
        first_send = threading.Event()
        self.robot.backend.send_sets.side_effect = lambda *a, **k: first_send.set()
        output = []
        worker = threading.Thread(target=lambda: output.append(self.toolkit.execute_tool(
            'move_delta', dict(dx_mm=100., dy_mm=0., dz_mm=0.))))
        worker.start()
        self.assertTrue(first_send.wait(1))
        self.toolkit.cancel_active_and_wait()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(output[0].is_error)
        self.assertIn('interrupted', str(output[0].data))
        self.assertFalse(self.robot.backend.send_sets.call_args.kwargs['servo'])

    def test_invalid_motion_is_reported_without_commands(self):
        self.robot.executing = True
        for args in (dict(dx_mm=float('nan'), dy_mm=0., dz_mm=0.),
                     dict(dx_mm=2000., dy_mm=0., dz_mm=0.)):
            result = self.toolkit.execute_tool('move_delta', args)
            self.assertTrue(result.is_error)
        self.robot.backend.send_sets.assert_called_with((0.,) * 6, servo=False)
        self.assertFalse(any(call.kwargs['servo'] for call in self.robot.backend.send_sets.call_args_list))

    def test_native_planner_entry_executes_tools_images_and_finish(self):
        import harness_vla
        from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
        from pydantic_ai.models.function import DeltaToolCall, FunctionModel
        from rpent.planner.api_loop import ApiAgentLoop

        calls = iter([('observe', {}), ('pi05_act', {'seconds': .08, 'instruction': 'custom prompt'}),
                      ('move_delta', dict(dx_mm=2., dy_mm=0., dz_mm=0.)),
                      ('set_gripper', {'position': 'closed'}),
                      ('finish', {'status': 'success', 'summary': 'Mock tool chain completed'})])
        image_counts = []

        def model(messages, info):
            images = [content for msg in messages if isinstance(msg, ModelRequest)
                      for part in msg.parts if isinstance(part, ToolReturnPart)
                      for content in part.files if isinstance(content, BinaryContent)]
            image_counts.append(len(images))
            name, arguments = next(calls)
            return ModelResponse(parts=[ToolCallPart(name, arguments)])

        async def stream(messages, info):
            response = model(messages, info)
            part = response.parts[0]
            yield {0: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args))}

        planner = ApiAgentLoop(model=FunctionModel(model, stream_function=stream), dashboard_events=NullDashboardEventSink(),
                               max_tokens=512, timeout_s=float('inf'))
        robot = FakeRobot()
        prompts = []

        class FakePolicyWorker:
            def __init__(self, attached_robot, *args):
                self.robot = attached_robot
            def start(self):
                prompts.append(self.robot.args.prompt)
            def latest(self):
                actions = np.tile([0, 0, .4, 0, 3.14, 0, 1], (50, 1))
                return ActionChunk(actions, time.monotonic(), 1, 1)
            def close(self):
                prompts.append('closed')

        output = self.root / 'entry-run'
        argv = ['harness_vla.py', '--model', 'openai-chat:test', '--ur5-repo', str(ROOT), '--config', str(ROOT / 'config/ur5.example.toml'),
                '--output-dir', str(output), '--memory-dir', str(self.root / 'memory')]
        with patch.object(sys, 'argv', argv), patch('rpent.planner.base.build_planner', return_value=planner), \
             patch('harness_vla.load_dotenv'), \
             patch('policy_rollout.UR5Robot', return_value=robot), \
             patch('harness_vla.subprocess.run') as service, \
             patch('harness_tools.PolicyWorker', FakePolicyWorker):
            self.assertEqual(harness_vla.main(), 0)
        self.assertGreater(max(image_counts), 0)
        self.assertEqual(prompts, ['custom prompt', 'closed'])
        self.assertEqual(robot.args.prompt, 'trained task')
        robot.backend.send_sets.assert_not_called()
        report = json.loads((output / 'result.json').read_text())
        self.assertEqual(report['finish']['status'], 'success')
        self.assertIsNone(report['physical_success'])
        transcript = json.loads((output / 'transcript.json').read_text())
        for entry in transcript:
            if entry['role'] == 'tool':
                data = json.loads(entry['content'])
                self.assertNotIn('error', data, data)
        actions = [json.loads(line) for line in (output / 'policy-actions.jsonl').read_text().splitlines()]
        self.assertTrue(any(row['would_send'] for row in actions))
        self.assertFalse(any(row['sent'] for row in actions))
        self.assertEqual(len(json.loads((output / 'states.json').read_text())['steps']), 5)
        service.assert_called_once()


if __name__ == '__main__':
    unittest.main()
