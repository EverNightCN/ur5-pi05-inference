"""RPent tool adapter for the existing UR5 controller and pi0.5 service."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from typing import Literal

import numpy as np

from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools, tool
from rpent.tools.toolkit import Toolkit

from policy_rollout import control_loop
from rollout_io import PolicyWorker, TrajectoryExecutor


class UR5HarnessToolkit(Toolkit):
    """Keep one RTDE/camera owner across planner tools and policy invocations."""

    def __init__(self, *, robot, cfg, output_dir, memory, dashboard_events):
        state = EnvState(Path(output_dir))
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self.robot = robot
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.capture_lock = threading.Lock()
        self.add_tools(iter_tools(self))
        self.observe()

    def capture(self, *, command=None, result=None, elapsed_s=None):
        with self.capture_lock:
            observation, observed_at = self.robot.get_observation()
            physical = self.robot.get_state()
            state = dict(tcp_pose=list(physical.tcp_pose), joints_rad=list(physical.joints),
                         gripper=float(observation['observation/state'][12]),
                         observed_at=observed_at, mode=self.robot.args.mode)
            with self.state.record_step(state=state, command=command, result=result, elapsed_s=elapsed_s):
                self.state.save('global.png', observation['observation/image'])
                self.state.save('wrist.png', observation['observation/wrist_image'])
            return ToolResult(data=self.state.latest_record().to_blob(), images=[
                self.state.load_bytes('global.png'), self.state.load_bytes('wrist.png')])

    def get_env_state(self, *, command, result, elapsed_s):
        return self.capture(command=command, result=result, elapsed_s=elapsed_s)

    @tool(readonly=True)
    def observe(self) -> ToolResult:
        """Capture fresh global and wrist RGB views, TCP pose, joints and gripper state.

        Images are returned in global, wrist order. TCP uses base-frame metres
        and axis-angle radians. Joint values use radians.
        """
        return self.capture(command={'action': 'observe'})

    @tool
    def pi05_act(self, seconds: float = 4.0, instruction: str = '') -> ToolResult:
        """Run the frozen pi0.5 policy for a finite interval, then hold and observe.

        The checkpoint was trained on the full red-cylinder/blue-cube task.
        An empty instruction uses the configured training instruction. Custom
        instructions are passed to the policy as supplied.

        Args:
            seconds: Duration of this policy invocation in seconds.
            instruction: Language instruction for pi0.5, or empty for the default.
        """
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('seconds must be positive and finite')
        self.raise_if_cancelled()
        previous_prompt = self.robot.args.prompt
        self.robot.args.prompt = instruction or previous_prompt
        self.robot.trajectory = TrajectoryExecutor()
        policy = PolicyWorker(self.robot, self.robot.args.host, self.robot.args.port,
                              self.cfg.timeout, self.cfg.infer_period)
        try:
            policy.start()
            with (self.output_dir / 'policy-actions.jsonl').open('a') as log:
                summary = control_loop(self.robot, policy, replace(self.cfg, duration=seconds),
                                       threading.Event(), log, check_cancelled=self.raise_if_cancelled)
        finally:
            try:
                policy.close()
            finally:
                self.robot.hold()
                self.robot.args.prompt = previous_prompt
        return ToolResult(data=dict(**summary, dry_run=not self.robot.executing))

    def execute_target(self, target, *, control_gripper=False):
        """Apply an analytic target using the shared pose/gripper progress logic."""
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('Target must contain seven finite values')
        self.robot.trajectory = TrajectoryExecutor()
        tick = 1 / self.cfg.fps
        previous = time.monotonic() - tick
        next_tick = previous + tick
        ticks = 0
        try:
            with (self.output_dir / 'analytic-actions.jsonl').open('a') as log:
                while True:
                    self.raise_if_cancelled()
                    time.sleep(max(0, next_tick - time.monotonic()))
                    self.raise_if_cancelled()
                    now = time.monotonic()
                    physical = self.robot.get_state()
                    command = self.robot.send_action(target, physical, now - previous,
                                                     control_gripper=control_gripper)
                    ticks += 1
                    log.write(json.dumps(dict(t=now, target=command, requested=target.tolist(),
                                              sent=self.robot.executing,
                                              complete=self.robot.trajectory.complete)) + '\n')
                    if self.robot.trajectory.complete:
                        return ToolResult(data=dict(ticks=ticks, target=command,
                                                    dry_run=not self.robot.executing))
                    previous = now
                    next_tick = max(next_tick + tick, time.monotonic())
        finally:
            self.robot.hold(stop_gripper=control_gripper)

    def current_target(self):
        observation, _ = self.robot.get_observation()
        return np.array([*observation['observation/state'][:6],
                         observation['observation/state'][12]], dtype=float)

    @tool
    def move_delta(self, dx_mm: float, dy_mm: float, dz_mm: float) -> ToolResult:
        """Translate the TCP relative to its measured position in the robot base frame.

        Args:
            dx_mm: Base-frame X displacement in millimetres.
            dy_mm: Base-frame Y displacement in millimetres.
            dz_mm: Base-frame Z displacement in millimetres.
        """
        target = self.current_target()
        target[:3] += np.asarray([dx_mm, dy_mm, dz_mm]) / 1000
        return self.execute_target(target)

    @tool
    def rotate_delta(self, rx_deg: float, ry_deg: float, rz_deg: float) -> ToolResult:
        """Rotate the TCP about a base-frame axis-angle vector, preserving position.

        Args:
            rx_deg: X component of the rotation vector in degrees.
            ry_deg: Y component of the rotation vector in degrees.
            rz_deg: Z component of the rotation vector in degrees.
        """
        from ur5_teleop import frames
        rotation = np.asarray([rx_deg, ry_deg, rz_deg], dtype=float)
        if not np.isfinite(rotation).all():
            raise ValueError('Rotation must contain finite values')
        target = self.current_target()
        delta = frames.rotvec_to_matrix(np.deg2rad(rotation))
        current = frames.rotvec_to_matrix(target[3:6])
        target[3:6] = frames.matrix_to_rotvec(frames.mat_mul(delta, current))
        return self.execute_target(target)

    @tool
    def set_gripper(self, position: Literal['open', 'closed']) -> ToolResult:
        """Open or close the gripper while holding the measured TCP pose.

        Args:
            position: 'open' for fully open, 'closed' for closed.
        """
        target = self.current_target()
        target[6] = float(position == 'open')
        return self.execute_target(target, control_gripper=True)
