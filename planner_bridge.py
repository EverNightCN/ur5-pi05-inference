"""Planner-neutral access to one UR5 owner and the existing pi0.5 control loop."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
import threading
import time

import numpy as np

from policy_rollout import UR5Robot, control_loop, load_rollout_config, prepare_execution
from rollout_io import PolicyWorker, TrajectoryExecutor


class Pi05Session:
    """Run one action segment at a time, with asynchronous cancellation requests.

    act() blocks until its segment has held the robot and closed its inference
    worker. pause()/stop() request cancellation and return immediately. wait()
    acknowledges completion. Semantic task success belongs to the caller.
    """

    def __init__(self, robot, cfg, output_dir):
        self.robot = robot
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._phase = 'idle'
        self._reason = None
        self._stopped = False
        self._closed = False
        self._error = None
        self._summary = None
        self._execution_thread = None

    def observe(self):
        """Return RGB images, model input state and independently timestamped feedback."""
        with self._lock:
            if self._closed:
                raise RuntimeError('Session is closed')
        observation, observed_at = self.robot.get_observation()
        measured = self.robot.get_state()
        return dict(
            observation=observation,
            observed_at=observed_at,
            feedback=dict(
                timestamp=measured.timestamp,
                tcp_pose_base=list(measured.tcp_pose),
                joints_rad=list(measured.joints),
                gripper_position_raw=self.robot.gripper.position_raw,
                gripper_state=self.robot.gripper.grip_state,
                robot_mode=measured.robot_mode,
                safety_status=measured.safety_status,
                runtime_state=measured.runtime_state,
            ),
        )

    def _status_locked(self):
        return dict(execution_status=self._phase, execution_done=self._idle.is_set(),
                    error=self._error, task_success=None,
                    summary=None if self._summary is None else dict(self._summary))

    def status(self):
        with self._lock:
            return self._status_locked()

    def act(self, instruction, seconds=4.0):
        """Execute a fresh language-conditioned segment; errors propagate to the caller."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('instruction must be a nonempty string')
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('seconds must be positive and finite')
        with self._lock:
            if self._closed or self._stopped:
                raise RuntimeError('Session is stopped or closed')
            if not self._idle.is_set():
                raise RuntimeError('An action segment is already active')
            self._idle.clear()
            self._cancel.clear()
            self._phase = 'running'
            self._reason = None
            self._error = None
            self._summary = None
            self._execution_thread = threading.get_ident()
        previous_prompt = self.robot.args.prompt
        self.robot.args.prompt = instruction
        self.robot.trajectory = TrajectoryExecutor()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f'planner-actions-{time.time_ns()}.jsonl'
            worker = PolicyWorker(self.robot, self.robot.args.host, self.robot.args.port,
                                  self.cfg.timeout, self.cfg.infer_period)
            try:
                worker.start()
                with path.open('x') as log:
                    summary = control_loop(self.robot, worker, replace(self.cfg, duration=seconds),
                                           self._cancel, log)
                with self._lock:
                    self._summary = dict(**summary, log_path=str(path), dry_run=not self.robot.executing)
            finally:
                worker.close()
        except BaseException as error:
            with self._lock:
                self._phase = 'failed'
                self._error = f'{type(error).__name__}: {error}'
            raise
        finally:
            self.robot.args.prompt = previous_prompt
            with self._lock:
                if self._error is None:
                    self._phase = self._reason or 'completed'
                else:
                    self._phase = 'failed'
                self._execution_thread = None
                self._idle.set()
                result = self._status_locked()
        return result

    def pause(self):
        """Request a hold. wait() acknowledges it; a later act() starts a fresh segment."""
        with self._lock:
            if not self._closed and not self._stopped and self._error is None:
                self._reason = 'paused'
                self._cancel.set()
                self._phase = 'paused' if self._idle.is_set() else 'pausing'
        return self.status()

    def stop(self):
        """Request a terminal hold. Further act() calls require a new session."""
        with self._lock:
            self._stopped = True
            self._reason = 'stopped'
            self._cancel.set()
            if not self._closed and self._error is None:
                self._phase = 'stopped' if self._idle.is_set() else 'stopping'
        return self.status()

    def wait(self, timeout=None):
        """Wait for control-loop hold and worker shutdown, then return terminal status."""
        with self._lock:
            if not self._idle.is_set() and self._execution_thread == threading.get_ident():
                raise RuntimeError('An action segment cannot wait for itself')
        if not self._idle.wait(timeout):
            raise TimeoutError('Action segment has not finished holding and shutting down')
        return self.status()


@contextmanager
def open_session(args):
    """Use policy_rollout arguments and keep hardware connected across planner calls."""
    if args.mode not in ('observe', 'dry-run', 'record'):
        raise ValueError('Planner sessions support observe, dry-run and record modes')
    hardware, cfg = load_rollout_config(args)
    with ExitStack() as stack:
        robot = UR5Robot(args, cfg, hardware, stack)
        stack.callback(robot.hold)
        prepare_execution(robot, args, hardware, cfg, threading.Event())
        session = Pi05Session(robot, cfg, args.log_dir)
        try:
            yield session
        finally:
            session.stop()
            session.wait()
            with session._lock:
                session._closed = True
                session._phase = 'closed'
