"""Fresh camera frames and bounded asynchronous openpi WebSocket inference."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


def canonicalize_policy_state(state):
    """Use the dominant training axis-angle branch while preserving physical pose."""
    from ur5_teleop import frames
    result = np.asarray(state, dtype=np.float32).copy()
    if result.shape != (13,) or not np.isfinite(result).all():
        raise ValueError("Policy state must contain 13 finite values")
    result[3:6] = frames.policy_rotvec(result[3:6])
    return result


class TrajectoryExecutor:
    """One committed waypoint and one progress value for translation, rotation and gripper."""
    def __init__(self):
        self.target = None
        self.action = None
        self.complete = True

    def step(self, candidate, tcp, measured_gripper, max_position, max_rotation, gripper_done=True,
             *, control_gripper=True):
        from ur5_teleop import frames
        if self.target is None:
            self.target = np.asarray([*tcp, measured_gripper], dtype=np.float64)
        if self.complete:
            self.action = np.asarray(candidate, dtype=np.float64).copy()
        start = frames.rotvec_to_matrix(self.target[3:6])
        end = frames.rotvec_to_matrix(self.action[3:6])
        tangent = np.asarray(frames.matrix_to_rotvec(frames.mat_mul(end, frames.mat_transpose(start))))
        displacement = self.action[:3] - self.target[:3]
        fraction = min(1.0, max_position / max(np.linalg.norm(displacement), 1e-12),
                       max_rotation / max(np.linalg.norm(tangent), 1e-12))
        self.target[:3] += displacement * fraction
        self.target[3:6] = frames.matrix_to_rotvec(frames.mat_mul(frames.rotvec_to_matrix(tangent * fraction), start))
        pose_complete = bool(fraction >= 1.0)
        grip = float(self.action[6] >= .5)
        changed = control_gripper and pose_complete and self.target[6] != grip
        if control_gripper and pose_complete:
            self.target[6] = grip
        self.complete = bool(pose_complete and (not control_gripper or (gripper_done and not changed)))
        return self.target.copy()


class LatestCamera:
    def __init__(self, device, width, height, fps):
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Camera unavailable: {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.timestamp = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._read, name=f"camera-{device}", daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Camera read failed: {self.device}")
                stamp = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frame, self.timestamp = rgb, stamp
        except Exception as exc:
            self.error = exc

    def snapshot(self, max_age):
        self.check_fresh(max_age)
        with self.lock:
            frame, stamp = self.frame.copy(), self.timestamp
        return frame, stamp

    def check_fresh(self, max_age):
        if self.error is not None:
            raise RuntimeError(f"Camera failed: {self.device}") from self.error
        with self.lock:
            if self.frame is None:
                raise RuntimeError(f"Camera has no frame: {self.device}")
            stamp = self.timestamp
        if time.monotonic() - stamp > max_age:
            raise RuntimeError(f"Camera frame expired: {self.device}")

    def close(self):
        self.stop.set()
        # V4L2 normally returns within one frame. Never release a capture concurrently with read().
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError(f"Camera read did not stop: {self.device}")
        self.cap.release()


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    observed_at: float
    infer_ms: float
    sequence: int

    def action_at(self, now, fps):
        index = max(0, math.floor((now - self.observed_at) * fps))
        if index >= len(self.actions):
            raise RuntimeError("Policy action chunk expired")
        return self.actions[index], index


def validate_actions(actions):
    result = np.asarray(actions, dtype=np.float64)
    if result.shape != (50, 7) or not np.isfinite(result).all():
        raise ValueError("Policy must return finite actions with shape (50, 7)")
    return result.copy()


class PolicyWorker:
    """One inference owner; control reads the latest timestamped chunk under a short lock."""
    def __init__(self, robot, host, port, timeout, infer_period):
        self.robot = robot
        self.uri = f"ws://{host}:{port}"
        self.timeout = timeout
        self.infer_period = infer_period
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.connection = None
        self.chunk = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="pi05-policy", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect
            with connect(self.uri, open_timeout=self.timeout, close_timeout=1,
                         compression=None, max_size=None) as connection:
                with self.lock:
                    self.connection = connection
                if self.stop.is_set():
                    return
                metadata = msgpack_numpy.unpackb(connection.recv(timeout=self.timeout))
                if metadata.get("action_horizon") != 50 or metadata.get("num_steps") != 10:
                    raise ValueError(f"Unexpected policy metadata: {metadata}")
                packer = msgpack_numpy.Packer()
                sequence = 0
                while not self.stop.is_set():
                    started = time.monotonic()
                    obs, observed_at = self.robot.get_observation()
                    connection.send(packer.pack(obs))
                    response = connection.recv(timeout=self.timeout)
                    if isinstance(response, str):
                        raise RuntimeError(f"Policy server failed: {response}")
                    actions = validate_actions(msgpack_numpy.unpackb(response)["actions"])
                    sequence += 1
                    chunk = ActionChunk(actions, observed_at,
                                        (time.monotonic() - started) * 1000, sequence)
                    with self.lock:
                        self.chunk = chunk
                    self.stop.wait(max(0, self.infer_period - (time.monotonic() - started)))
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc

    def latest(self):
        if self.error is not None:
            raise RuntimeError("Policy inference failed") from self.error
        with self.lock:
            return self.chunk

    def close(self):
        self.stop.set()
        with self.lock:
            connection = self.connection
        try:
            if connection is not None:
                connection.close()
        finally:
            if self.thread.ident is not None:
                self.thread.join(timeout=self.timeout + 2)
        if self.thread.is_alive():
            raise RuntimeError("Policy worker did not stop")
