"""UR5 π0.5 rollout with LeRobot-style robot/policy adapters and a fixed-rate loop.

Reference: huggingface/lerobot e6ffde60312365c5452c46189a26872483943154,
lerobot/scripts/lerobot_rollout.py and lerobot/rollout/controller.py.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from rollout_io import LatestCamera, PolicyWorker, TrajectoryExecutor, canonicalize_policy_state

PROMPT = ("Put the red cylinder and the blue cube into the storage box. "
          "Leave all other objects outside the box.")
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutConfig:
    fps: float = 30.0
    action_fps: float = 30.0
    duration: float = 15.0
    infer_period: float = 0.4
    timeout: float = 3.0
    max_step_mm: float = 2.0
    max_rot_step_deg: float = 2.0
    camera_max_age: float = 0.25
    camera_max_skew: float = 0.1
    home_tolerance_deg: float = 1.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "duration":
                # duration = 0 -> 不限时（Ctrl-C 停止）
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and >= 0")
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def check_state(state, *, executing=False):
    values = np.asarray([*state.tcp_pose, *state.joints], dtype=float)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise RuntimeError("Robot state must contain 12 finite TCP/joint values")
    if time.monotonic() - state.timestamp > 0.2:
        raise RuntimeError("Robot state expired")
    if executing and (state.safety_status != 1 or state.robot_mode != 7):
        raise RuntimeError(f"Robot state rejected: safety={state.safety_status}, mode={state.robot_mode}")
    if executing and state.runtime_state != 2:
        raise RuntimeError(f"Servo program is not running: runtime_state={state.runtime_state}")


def at_home(joints, home_degrees, tolerance_degrees):
    # Compare actual joint turns, preserving the configured cable/wrist winding.
    return bool(np.all(np.abs(np.rad2deg(joints) - home_degrees) <= tolerance_degrees))


class UR5Robot:
    def __init__(self, args, cfg, hardware, stack):
        from ur5_teleop.backend import BackendError
        from ur5_teleop.ur_client import RTDEBackend
        self.args, self.cfg, self.hardware = args, cfg, hardware
        self.executing = args.mode == "record"
        self.backend = RTDEBackend(args.robot_ip or hardware['robot']['ip'],
                                   read_only=not self.executing,
                                   frequency_hz=hardware['robot']['rtde_frequency_hz'])
        self.backend.connect()
        stack.callback(self.backend.close)
        deadline = time.monotonic() + cfg.timeout
        while True:
            try:
                self.backend.read_state()
                break
            except BackendError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        self.gripper = None
        if not args.no_gripper_read:
            from ur5_teleop.gripper import Gripper
            grip = hardware['gripper']
            self.gripper = Gripper(args.gripper_dev or grip['port'], baud=grip['baud'],
                                   slave_id=grip['slave_id'], speed=grip['speed'], force=grip['force'],
                                   read_only=not self.executing)
            stack.callback(self.gripper.close)
            self.gripper.connect()
        self.cameras = []
        collect = hardware['collect']
        devices = (args.global_dev or collect['camera_indices'][0],
                   args.wrist_dev or collect['camera_indices'][1])
        for device in devices:
            camera = LatestCamera(device, collect['camera_width'], collect['camera_height'],
                                  collect['camera_fps'])
            self.cameras.append(camera)
            stack.callback(camera.close)
        deadline = time.monotonic() + cfg.timeout
        while any(camera.frame is None for camera in self.cameras):
            for camera in self.cameras:
                if camera.error is not None:
                    raise RuntimeError(f"Camera failed: {camera.device}") from camera.error
            if time.monotonic() >= deadline:
                raise TimeoutError("Camera startup timed out")
            time.sleep(0.02)
        self.trajectory = TrajectoryExecutor()

    def get_state(self):
        state = self.backend.read_state()
        check_state(state, executing=self.executing)
        if self.gripper is not None and not self.gripper.healthy:
            raise RuntimeError(f"Gripper failed: {self.gripper.last_error}")
        if self.gripper is not None and time.monotonic() - self.gripper.last_read_at > 0.3:
            raise RuntimeError("Gripper feedback expired")
        for camera in self.cameras:
            camera.check_fresh(self.cfg.camera_max_age)
        return state

    def get_observation(self):
        state = self.get_state()
        pairs = [camera.snapshot(self.cfg.camera_max_age) for camera in self.cameras]
        stamps = [stamp for _, stamp in pairs] + [state.timestamp]
        if max(stamps) - min(stamps) > self.cfg.camera_max_skew:
            raise RuntimeError("Camera/robot observation timestamps are too far apart")
        if self.gripper is None:
            raise RuntimeError("Policy observations require a measured gripper position")
        grip = self.gripper.position_raw
        if grip is None:
            raise RuntimeError("Gripper position is unavailable")
        obs = {"observation/image": pairs[0][0], "observation/wrist_image": pairs[1][0],
               "observation/state": canonicalize_policy_state([*state.tcp_pose, *state.joints, grip / 1000]),
               "prompt": self.args.prompt}
        return obs, min(stamps)

    def send_action(self, action, state, dt, *, control_gripper=True):
        desired = np.asarray(action, dtype=float)
        if desired.shape != (7,) or not np.isfinite(desired).all():
            raise ValueError("Action must contain seven finite values")
        safety = self.hardware['safety']
        if (np.any(desired[:3] < safety['workspace_min']) or
                np.any(desired[:3] > safety['workspace_max'])):
            raise ValueError(f"Policy target outside configured workspace: {desired[:3]}")
        step = self.cfg.max_step_mm / 1000 * min(dt * self.cfg.fps, 1.0)
        rotation = np.deg2rad(self.cfg.max_rot_step_deg) * min(dt * self.cfg.fps, 1.0)
        gripper_done = (not control_gripper or not self.executing or self.trajectory.target is None or
                        self.gripper.motion_complete(round(self.trajectory.target[6] * 1000)))
        target = self.trajectory.step(desired, state.tcp_pose, self.gripper.position_raw / 1000,
                                      step, rotation, gripper_done, control_gripper=control_gripper)
        grip = round(target[6] * 1000)
        if self.executing:
            self.backend.send_sets(target[:6], servo=True)
            if control_gripper:
                self.gripper.set_position(grip)
        return target.tolist()

    def hold(self, *, stop_gripper=True):
        if self.executing:
            # cmd=0 ignores the pose payload, so stopping also works with stale feedback.
            try:
                self.backend.send_sets((0.0,) * 6, servo=False)
            finally:
                if stop_gripper and self.gripper is not None:
                    self.gripper.stop_motion()


def move_home(args, hardware, cfg, stop, *, backend=None):
    from ur5_teleop.home import build_movej_program
    from ur5_teleop.ur_client import DashboardClient, RTDEBackend, open_script_channel, send_program
    ip = args.robot_ip or hardware['robot']['ip']
    target = hardware['home']['joints_deg']
    LOG.info('Moving HOME joints_deg=%s, acceleration=0.3 rad/s^2, speed=0.2 rad/s', target)
    with ExitStack() as stack:
        if backend is None:
            backend = RTDEBackend(ip)
            backend.connect()
            stack.callback(backend.close)
        dashboard = DashboardClient(ip)
        dashboard.connect()
        stack.callback(dashboard.close)
        if 'NORMAL' not in dashboard.command('safetymode'):
            raise RuntimeError("HOME requires NORMAL safety mode")
        try:
            dashboard.command('stop')
            with open_script_channel(ip, 30001) as channel:
                send_program(channel, build_movej_program(target, a=0.3, v=0.2))
                stop.wait(0.8)
            deadline = time.monotonic() + 60
            startup_deadline = time.monotonic() + 3
            program_started = False
            while not stop.wait(0.05):
                state = backend.read_state()
                check_state(state)
                if state.safety_status != 1 or state.robot_mode != 7:
                    raise RuntimeError("Robot became unavailable during HOME")
                if at_home(state.joints, target, cfg.home_tolerance_deg):
                    LOG.info("HOME reached, joints_deg=%s", target)
                    return
                program_started = program_started or state.runtime_state == 2
                if state.runtime_state == 1 and (program_started or time.monotonic() > startup_deadline):
                    raise RuntimeError("HOME program stopped before reaching the target")
                if time.monotonic() >= deadline:
                    raise TimeoutError("HOME move exceeded 60 seconds")
            raise InterruptedError("HOME interrupted")
        finally:
            # Stop the movej program on success, timeout, communication loss and Ctrl-C.
            dashboard.command('stop')


def install_servo_program(args, hardware, stop):
    from ur5_teleop.ur_client import open_script_channel, send_program
    program = (Path(args.ur5_repo) / 'urscript/ur5_teleop.script').read_text()
    with open_script_channel(args.robot_ip or hardware['robot']['ip'], 30001) as channel:
        send_program(channel, program)
        if stop.wait(0.8):
            raise InterruptedError("Servo setup interrupted")


def prepare_execution(robot, args, hardware, cfg, stop):
    """Initialize HOME and the servo program using the connected RTDE owner."""
    if not robot.executing:
        return
    if args.home_at_start:
        move_home(args, hardware, cfg, stop, backend=robot.backend)
    if not at_home(robot.backend.read_state().joints, hardware['home']['joints_deg'], cfg.home_tolerance_deg):
        raise RuntimeError('Robot is outside HOME tolerance; use --mode home or --home-at-start')
    install_servo_program(args, hardware, stop)
    deadline = time.monotonic() + cfg.timeout
    while robot.backend.read_state().runtime_state != 2:
        if stop.wait(0.02) or time.monotonic() >= deadline:
            raise TimeoutError('Servo program startup interrupted or timed out')
    LOG.info('Servo program ready; policy control enabled')


def control_loop(robot, policy, cfg, stop, log, *, check_cancelled=None):
    started = time.monotonic()
    next_tick = started
    last_tick = started - 1 / cfg.fps
    ticks = sent = missed = 0
    try:
        while not stop.is_set():
            if check_cancelled is not None:
                check_cancelled()
            if stop.wait(max(0, next_tick - time.monotonic())):
                break
            now = time.monotonic()
            if cfg.duration > 0 and now - started >= cfg.duration:
                break
            state = robot.get_state()
            chunk = policy.latest() if policy is not None else None
            action = target = index = None
            if chunk is not None:
                action, index = chunk.action_at(now, cfg.action_fps)
                target = robot.send_action(action, state, now - last_tick)
                sent += int(robot.executing)
            elif policy is not None and now - started > cfg.timeout:
                raise TimeoutError("No initial policy chunk within timeout")
            elif policy is None:
                for camera in robot.cameras:
                    camera.snapshot(cfg.camera_max_age)
            row = {'t': now - started, 'mode': robot.args.mode, 'tcp': list(state.tcp_pose),
                   'joints': list(state.joints), 'sent': target is not None and robot.executing,
                   'would_send': target is not None and not robot.executing,
                   'grip_measured': None if robot.gripper is None else robot.gripper.position_raw / 1000,
                   'grip_state': None if robot.gripper is None else robot.gripper.grip_state,
                   'queue': 0 if index is None else len(chunk.actions) - index - 1,
                   'action': None if action is None else action.tolist(), 'target': target,
                   'committed_action': None if target is None else robot.trajectory.action.tolist(),
                   'action_complete': None if target is None else robot.trajectory.complete,
                   'chunk_index': index, 'infer_count': 0 if chunk is None else chunk.sequence,
                   'infer_ms': None if chunk is None else chunk.infer_ms,
                   'observation_age': None if chunk is None else now - chunk.observed_at}
            log.write(json.dumps(row) + '\n')
            ticks += 1
            last_tick = now
            next_tick += 1 / cfg.fps
            finished = time.monotonic()
            if next_tick <= finished:
                missed += 1
                next_tick = finished + 1 / cfg.fps
        return {'ticks': ticks, 'sent': sent, 'overruns': missed,
                'elapsed': time.monotonic() - started}
    finally:
        robot.hold()


def resolve_config(args, hardware):
    values = dict(hardware.get('inference', {}))
    unknown = values.keys() - RolloutConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown inference settings: {sorted(unknown)}")
    values.update({name: getattr(args, name) for name in RolloutConfig.__dataclass_fields__
                   if getattr(args, name) is not None})
    return RolloutConfig(**values)


def load_rollout_config(args):
    """Read the deployment config and validate the task's HOME convention."""
    sys.path.insert(0, args.ur5_repo)
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    path = Path(args.config) if args.config else Path(args.ur5_repo) / 'config/ur5.local.toml'
    hardware = tomllib.loads(path.read_text())
    cfg = resolve_config(args, hardware)
    from ur5_teleop.home import HOME_JOINTS_DEG
    home = np.asarray(hardware['home']['joints_deg'], dtype=float)
    if home.shape != (6,) or not np.array_equal(home, HOME_JOINTS_DEG):
        raise ValueError(f"This task requires HOME joints {HOME_JOINTS_DEG}")
    return hardware, cfg


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['observe', 'dry-run', 'record', 'home'], default='dry-run')
    ap.add_argument('--ur5-repo', default=os.environ.get('UR5_REPO', str(Path(__file__).resolve().parent)))
    ap.add_argument('--config', help='UR5 TOML config, default <ur5-repo>/config/ur5.local.toml')
    ap.add_argument('--robot-ip')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    for name in RolloutConfig.__dataclass_fields__:
        ap.add_argument('--' + name.replace('_', '-'), type=float,
                        help='Override [inference] setting in the UR5 TOML config')
    ap.add_argument('--global-dev')
    ap.add_argument('--wrist-dev')
    ap.add_argument('--gripper-dev')
    ap.add_argument('--prompt', default=PROMPT)
    ap.add_argument('--log-dir', default=str(Path(__file__).resolve().parent / '.runtime/logs'))
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--i-understand', action='store_true')
    ap.add_argument('--home-at-start', action='store_true')
    ap.add_argument('--no-gripper-read', action='store_true', help='Supported for observe mode')
    args = ap.parse_args()
    if (args.mode in ('record', 'home') or args.home_at_start) and not (args.execute and args.i_understand):
        ap.error('Motion requires --execute --i-understand')
    if args.home_at_start and args.mode != 'record':
        ap.error('--home-at-start requires --mode record')
    if args.no_gripper_read and args.mode != 'observe':
        ap.error('--no-gripper-read is supported only for observe')
    return args


def main():
    args = parse_args()
    hardware, cfg = load_rollout_config(args)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    LOG.info('Rollout config fps=%g action_fps=%g duration=%gs max_step=%gmm max_rotation=%gdeg',
             cfg.fps, cfg.action_fps, cfg.duration, cfg.max_step_mm, cfg.max_rot_step_deg)
    LOG.info('Policy state uses positive-Ry axis-angle; pose and gripper share waypoint progress')
    if cfg.duration <= 0:
        LOG.info('duration<=0: run until Ctrl-C / SIGINT')
    stop = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda *_: stop.set())
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        if args.mode == 'home':
            move_home(args, hardware, cfg, stop)
            return 0
        with ExitStack() as stack:
            robot = UR5Robot(args, cfg, hardware, stack)
            prepare_execution(robot, args, hardware, cfg, stop)
            if robot.executing:
                stack.callback(robot.hold)
            policy = None
            if args.mode != 'observe':
                policy = PolicyWorker(robot, args.host, args.port, cfg.timeout, cfg.infer_period)
                stack.callback(policy.close)
                policy.start()
            directory = Path(args.log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / f'rollout_{time.time_ns()}.jsonl'
            LOG.info('Writing rollout log to %s', log_path)
            with log_path.open('x') as log:
                summary = control_loop(robot, policy, cfg, stop, log)
            LOG.info('Rollout completed %s', summary)
            return 0
    except InterruptedError:
        LOG.info('Rollout interrupted')
        return 130
    except Exception:
        LOG.exception('Rollout failed')
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
