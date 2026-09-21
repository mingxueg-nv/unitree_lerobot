"""
Policy inference on H2 + Sharpa using a remote GR00T policy server.

The GR00T server runs in GR00T's uv environment (separate terminal):

    cd /home/nvidia/binliu/Isaac-GR00T
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path /path/to/checkpoint \\
        --embodiment-tag new_embodiment \\
        --port 5555

This script runs in the tv conda environment (no gr00t package required).
It communicates via ZMQ using the same msgpack wire format as GR00T's
PolicyServer / PolicyClient.

Observation keys sent to the server (single-eye head mode, default):
    video.high              (H, W, 3) uint8   ← chosen head eye (left or right)
    video.left_wrist_view   (H, W, 3) uint8
    video.right_wrist_view  (H, W, 3) uint8
    state.left_arm          (7,)  float32
    state.right_arm         (7,)  float32
    state.left_hand         (22,) float32
    state.right_hand        (22,) float32
    language.annotation.human.task_description  str

Stereo head mode (`--head-eye both`):
    video.left_eye_view     (H, W, 3) uint8
    video.right_eye_view    (H, W, 3) uint8
    (plus wrists and state as above)

Action chunk returned by the server:
    left_arm   (B, T, 7)
    right_arm  (B, T, 7)
    left_hand  (B, T, 22)
    right_hand (B, T, 22)

Image source: ROS 2 (CycloneDDS) on the four sensing_h2_ros_gstreamer topics,
or the legacy ZMQ teleimager server via `--image-source zmq`.

Hand control: through the C++ sharpa_dds_bridge on Thor (default), or the
direct Sharpa SDK on the local machine via `--hands-source sdk`.

Usage:
    # Standing replay, ROS cameras, DDS bridge for hands:
    python unitree_lerobot/eval_robot/eval_h2_groot.py \\
        --policy-host localhost --policy-port 5555 \\
        --task "pick up the apple" \\
        --motion

    # Right-eye only as video.high:
    python unitree_lerobot/eval_robot/eval_h2_groot.py \\
        --policy-host localhost --task "..." --head-eye right

    # Stereo head (video.left_eye_view + video.right_eye_view):
    python unitree_lerobot/eval_robot/eval_h2_groot.py \\
        --policy-host localhost --task "..." --head-eye both

    # Arm only (no hands), legacy ZMQ image server:
    python unitree_lerobot/eval_robot/eval_h2_groot.py \\
        --policy-host localhost --no-hands \\
        --image-source zmq --img-server-ip 192.168.124.162
"""

import argparse
from collections import deque
import io
import json
import math
import os
import sys
import threading
import time

import msgpack
import msgpack_numpy as msgpack_np
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_XR_ROOT    = os.environ.get(
    "XR_TELEOPERATE_ROOT",
    os.path.expanduser("~/workspace/xr_teleoperate"),
)
_SHARPA_SDK = os.environ.get("SHARPA_SDK_PATH", "/usr/lib/sharpa-wave-sdk/python")

for p in [_REPO_ROOT, _XR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── action layout ─────────────────────────────────────────────────────────────
ARM_LEFT_SLICE  = slice(0, 7)
ARM_RIGHT_SLICE = slice(7, 14)
LEFT_EE_SLICE   = slice(14, 36)
RIGHT_EE_SLICE  = slice(36, 58)
SHARPA_DOF      = 22
ARM_DOF         = 14

# Optional, data-derived right-hand grip assist. The robust limits come directly
# from this task's 275 Parquet episodes; the tight reference is the 75th
# percentile of the top-quartile episodes' tightest 10% frames.
RIGHT_GRIP_FLEX_INDICES = np.array(
    [0, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18, 20, 21], dtype=np.int64
)
RIGHT_GRIP_ASSIST_INDICES = np.array(
    [0, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18, 20, 21], dtype=np.int64
)
RIGHT_GRIP_DATA_Q01 = np.array(
    [-0.010086, -0.224762, -0.448843, -0.292790, 0.049835, -0.149491,
     -0.285544, 0.040267, 0.134379, -0.120019, -0.278353, 0.054019,
     0.089177, -0.151634, -0.196518, 0.036057, 0.118800, 0.117146,
     -0.065865, -0.190628, 0.055334, 0.106129],
    dtype=np.float64,
)
RIGHT_GRIP_DATA_Q99 = np.array(
    [1.436906, 0.254193, 0.746905, 0.250873, 0.396878, 0.820181,
     0.080465, 0.284368, 0.576840, 0.838638, 0.057478, 0.371396,
     0.685116, 1.091048, 0.019280, 0.569340, 1.310820, 0.178420,
     1.084540, 0.001880, 0.654376, 1.310549],
    dtype=np.float64,
)
RIGHT_GRIP_TIGHT_REFERENCE = np.array(
    [0.970855, -0.059733, 0.657004, 0.175105, 0.264262, 0.759542,
     -0.234731, 0.161338, 0.474259, 0.777466, -0.267898, 0.215217,
     0.550285, 1.007525, -0.098707, 0.430902, 1.308348, 0.170494,
     0.945218, -0.081950, 0.496444, 1.309591],
    dtype=np.float64,
)
RIGHT_GRIP_ASSIST_STRENGTH = 0.50
RIGHT_GRIP_ASSIST_MAX_DELTA_RAD = 0.10
RIGHT_GRIP_ASSIST_DELTA_SLEW_RAD = 0.01
RIGHT_GRIP_GATE_START = 0.40
RIGHT_GRIP_GATE_FULL = 0.55


class HandCommandSafetyError(RuntimeError):
    """Raised before DDS send when an assisted hand target is invalid."""


def _apply_right_hand_grip_assist(
    target,
    previous_delta,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Conservatively move selected flexion joints toward demonstrated tight grip."""
    q = np.asarray(target, dtype=np.float64)
    if q.shape != (SHARPA_DOF,) or not np.all(np.isfinite(q)):
        raise HandCommandSafetyError(
            f"invalid right-hand target: shape={q.shape}, finite={np.all(np.isfinite(q))}"
        )

    data_range = RIGHT_GRIP_DATA_Q99 - RIGHT_GRIP_DATA_Q01
    safety_low = RIGHT_GRIP_DATA_Q01 - 0.25 * data_range
    safety_high = RIGHT_GRIP_DATA_Q99 + 0.25 * data_range
    outside = np.flatnonzero((q < safety_low) | (q > safety_high))
    if outside.size:
        raise HandCommandSafetyError(
            "right-hand model target outside demonstrated safety envelope at "
            f"indices={outside.tolist()}"
        )

    flex = RIGHT_GRIP_FLEX_INDICES
    score = float(np.mean(
        (q[flex] - RIGHT_GRIP_DATA_Q01[flex])
        / (RIGHT_GRIP_DATA_Q99[flex] - RIGHT_GRIP_DATA_Q01[flex])
    ))
    gate = float(np.clip(
        (score - RIGHT_GRIP_GATE_START)
        / (RIGHT_GRIP_GATE_FULL - RIGHT_GRIP_GATE_START),
        0.0,
        1.0,
    ))

    desired_delta = np.zeros(SHARPA_DOF, dtype=np.float64)
    idx = RIGHT_GRIP_ASSIST_INDICES
    desired_delta[idx] = gate * np.clip(
        RIGHT_GRIP_ASSIST_STRENGTH * (RIGHT_GRIP_TIGHT_REFERENCE[idx] - q[idx]),
        0.0,
        RIGHT_GRIP_ASSIST_MAX_DELTA_RAD,
    )
    prev = np.asarray(previous_delta, dtype=np.float64)
    delta = prev + np.clip(
        desired_delta - prev,
        -RIGHT_GRIP_ASSIST_DELTA_SLEW_RAD,
        RIGHT_GRIP_ASSIST_DELTA_SLEW_RAD,
    )
    # Never reduce the model target, and never let the assist itself push a
    # joint above the raw dataset's q99. If the model is already above q99,
    # preserve it unchanged rather than clipping it downward.
    applied_delta = np.zeros(SHARPA_DOF, dtype=np.float64)
    applied_delta[idx] = np.minimum(
        delta[idx],
        np.maximum(RIGHT_GRIP_DATA_Q99[idx] - q[idx], 0.0),
    )
    assisted = q + applied_delta
    return assisted, applied_delta, score

# ── video key naming per checkpoint ───────────────────────────────────────────
# Camera roles inside this script are always high / left_wrist_view /
# right_wrist_view. Checkpoints trained with other names need a rename on the
# wire; N1.7 servers look the keys up verbatim and raise KeyError otherwise.
VIDEO_KEY_STYLES: dict[str, dict[str, str]] = {
    "high_view":  {},   # video.high, video.left_wrist_view, video.right_wrist_view
    "head_right": {     # gr00t-n1.7-h2-sharpa-trocar-ckpt44200
        "high":              "head_right",
        "left_wrist_view":   "wrist_left",
        "right_wrist_view":  "wrist_right",
    },
}

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger = logging_mp.getLogger(__name__)


# ── GR00T wire protocol (compatible with gr00t.policy.server_client) ──────────

def _np_encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    raise TypeError(f"Unknown type: {type(obj)}")


def _np_decode(obj):
    if not isinstance(obj, dict):
        return obj
    if b"__ndarray_class__" in obj or "__ndarray_class__" in obj:
        key = b"as_npy" if b"as_npy" in obj else "as_npy"
        return np.load(io.BytesIO(obj[key]), allow_pickle=False)
    # Isaac-GR00T's current PolicyServer encodes arrays with msgpack-numpy.
    # Keep accepting the legacy np.save envelope above for compatibility.
    return msgpack_np.decode(obj)


class GR00TClient:
    """
    Minimal ZMQ client compatible with gr00t.policy.server_client.PolicyServer.
    Only requires zmq, msgpack, numpy — no gr00t package needed.
    """

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000,
                 video_key_map: dict[str, str] | None = None):
        import zmq
        self._video_key_map = dict(video_key_map or {})
        self._ctx    = zmq.Context()
        self._address = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._address)
        self._zmq = zmq

    def _send(self, endpoint: str, data: dict | None = None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._socket.send(msgpack.packb(request, default=_np_encode))
        raw = self._socket.recv()
        resp = msgpack.unpackb(raw, object_hook=_np_decode)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def _reinit_socket(self):
        import zmq
        self._socket.close()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._address)

    def ping(self) -> bool:
        try:
            resp = self._send("ping")
            return isinstance(resp, dict) and resp.get("status") == "ok"
        except Exception:
            self._reinit_socket()
            return False

    def reset(self):
        self._send("reset", {"options": None})

    def _apply_video_key_map(self, obs: dict) -> dict:
        """Rename the camera-role keys to whatever the checkpoint was trained with."""
        video = obs.get("video")
        if not self._video_key_map or not isinstance(video, dict):
            return obs
        renamed = dict(obs)
        renamed["video"] = {self._video_key_map.get(k, k): v for k, v in video.items()}
        return renamed

    def get_action(self, obs: dict) -> tuple[dict, dict]:
        """
        obs: nested dict, values are np.ndarray with shape (B, T, ...) or lists of strings.
        Returns (action_chunk, info) where action_chunk values are (B, T, D) arrays.
        """
        obs = self._apply_video_key_map(obs)
        response = self._send("get_action", {"observation": obs, "options": None})
        if isinstance(response, (list, tuple)) and len(response) == 2:
            return response[0], response[1]
        return response, {}

    def __del__(self):
        try:
            self._socket.close()
            self._ctx.term()
        except Exception:
            pass


class GR00TN2Client(GR00TClient):
    """N2 adapter for the legacy H2 control loop.

    N2's server uses flat dotted keys and returns flat ``action.*`` arrays.
    It also conditions on temporal observations.  This adapter queries the
    checkpoint's modality config, maps the H2 observations to those keys, and
    maintains the required non-positive history.  It intentionally sends
    ``options=None``: N2 does not support RTC.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._modality_config = None
        self._history = None
        self._history_len = 1
        self._deltas = {}

    @staticmethod
    def _config_field(config: dict, modality: str, field: str):
        value = config.get(modality, {})
        if isinstance(value, dict):
            return value.get(field, [])
        return getattr(value, field, [])

    def configure(self):
        config = self._send("get_modality_config")
        if not isinstance(config, dict):
            raise RuntimeError(f"N2 returned invalid modality config: {type(config)}")

        self._modality_config = config
        video_keys = list(self._config_field(config, "video", "modality_keys"))
        state_keys = list(self._config_field(config, "state", "modality_keys"))
        action_keys = list(self._config_field(config, "action", "modality_keys"))
        language_keys = list(self._config_field(config, "language", "modality_keys"))

        # Fail before policy inference if this is not the three-camera,
        # 58-DoF H2+Sharpa checkpoint contract.
        video_roles = []
        for key in video_keys:
            name = key.lower()
            if "wrist" in name and "left" in name:
                video_roles.append("left_wrist")
            elif "wrist" in name and "right" in name:
                video_roles.append("right_wrist")
            elif "head" in name or "high" in name:
                video_roles.append("right_head")
            else:
                video_roles.append("unknown")
        if sorted(video_roles) != ["left_wrist", "right_head", "right_wrist"]:
            raise RuntimeError(
                "N2 checkpoint camera contract does not match this H2 deployment: "
                f"keys={video_keys}, mapped_roles={video_roles}"
            )

        split_keys = {"left_arm", "right_arm", "left_hand", "right_hand"}
        full_state_keys = {"state", "observation.state", "robot_state"}
        if not (
            set(state_keys) == split_keys
            or (len(state_keys) == 1 and state_keys[0] in full_state_keys)
        ):
            raise RuntimeError(
                "N2 checkpoint state contract is not the supported H2 58-DoF layout: "
                f"{state_keys}"
            )
        if not (
            set(action_keys) == split_keys
            or (len(action_keys) == 1 and action_keys[0] == "action")
        ):
            raise RuntimeError(
                "N2 checkpoint action contract is not the supported H2 58-DoF layout: "
                f"{action_keys}"
            )
        if not language_keys:
            raise RuntimeError("N2 checkpoint has no language modality key")

        for modality in ("video", "state"):
            all_deltas = [
                int(v) for v in self._config_field(config, modality, "delta_indices")
            ]
            # Positive video deltas are generated by N2, not observations supplied
            # by the real robot.
            deltas = tuple(v for v in all_deltas if v <= 0) or (0,)
            self._deltas[modality] = deltas
        self._history_len = 1 + max(
            (-min(values) for values in self._deltas.values()), default=0
        )
        self._history = deque(maxlen=self._history_len)

        action_deltas = self._config_field(config, "action", "delta_indices")
        logger.info(
            "N2 modality config: "
            f"video={video_keys}, mapped_roles={video_roles}, "
            f"video_history={list(self._deltas['video'])}, "
            f"state={state_keys}, "
            f"state_history={list(self._deltas['state'])}, "
            f"action={action_keys}, "
            f"action_horizon={len(action_deltas)}"
        )

    @staticmethod
    def _latest(value):
        value = np.asarray(value)
        if value.ndim < 2 or value.shape[0] != 1 or value.shape[1] < 1:
            raise ValueError(f"expected observation with (B=1,T,...) dims, got {value.shape}")
        return value[0, -1]

    @staticmethod
    def _video_value(video: dict, required_key: str):
        if required_key in video:
            return video[required_key]
        name = required_key.lower()
        if "wrist" in name and "left" in name:
            source = "left_wrist_view"
        elif "wrist" in name and "right" in name:
            source = "right_wrist_view"
        elif "head" in name or "high" in name:
            source = "high"
        else:
            raise KeyError(
                f"cannot map N2 video key '{required_key}' to H2 cameras; "
                f"available={list(video)}"
            )
        if source not in video:
            raise KeyError(
                f"N2 video key '{required_key}' needs H2 camera '{source}', "
                f"available={list(video)}"
            )
        return video[source]

    @staticmethod
    def _state_value(state: dict, required_key: str):
        if required_key in state:
            return state[required_key]
        if required_key in ("state", "observation.state", "robot_state"):
            return np.concatenate(
                [
                    state["left_arm"],
                    state["right_arm"],
                    state["left_hand"],
                    state["right_hand"],
                ],
                axis=-1,
            )
        raise KeyError(
            f"cannot map N2 state key '{required_key}'; available={list(state)}"
        )

    def _current_flat_observation(self, observation: dict) -> dict:
        if self._modality_config is None:
            raise RuntimeError("GR00TN2Client.configure() must be called after connecting")

        video = observation.get("video", {})
        state = observation.get("state", {})
        flat = {}
        for key in self._config_field(self._modality_config, "video", "modality_keys"):
            flat[f"video.{key}"] = self._latest(self._video_value(video, key))
        for key in self._config_field(self._modality_config, "state", "modality_keys"):
            flat[f"state.{key}"] = self._latest(self._state_value(state, key)).astype(
                np.float32, copy=False
            )

        language = observation.get("language", {})
        task = next(iter(language.values()), [[""]])
        while isinstance(task, (list, tuple)) and task:
            task = task[0]
        for key in self._config_field(self._modality_config, "language", "modality_keys"):
            flat[key] = task
        return flat

    def observe(self, observation: dict):
        """Record one 30 Hz observation for the next N2 inference call."""
        current = self._current_flat_observation(observation)
        if not self._history:
            self._history.extend(dict(current) for _ in range(self._history_len))
        else:
            self._history.append(current)

    def _policy_observation(self) -> dict:
        if not self._history:
            raise RuntimeError("N2 observation history is empty")
        latest = dict(self._history[-1])
        for modality in ("video", "state"):
            for key in self._config_field(
                self._modality_config, modality, "modality_keys"
            ):
                flat_key = f"{modality}.{key}"
                latest[flat_key] = np.stack(
                    [self._history[delta - 1][flat_key] for delta in self._deltas[modality]],
                    axis=0,
                )
        return latest

    @staticmethod
    def _with_batch(value):
        value = np.asarray(value)
        return value[np.newaxis, ...] if value.ndim == 2 else value

    def _legacy_action(self, action: dict) -> dict:
        converted = {
            key.removeprefix("action."): np.asarray(value)
            for key, value in action.items()
        }
        expected_dims = {
            "left_arm": 7,
            "right_arm": 7,
            "left_hand": 22,
            "right_hand": 22,
        }
        if all(key in converted for key in expected_dims):
            legacy = {
                key: self._with_batch(converted[key])
                for key in expected_dims
            }
        else:
            full = next(
                (value for value in converted.values() if value.shape[-1] == 58),
                None,
            )
            if full is None:
                raise KeyError(
                    "N2 action cannot be mapped to H2's 58 joints; "
                    f"received={[(key, value.shape) for key, value in converted.items()]}"
                )
            full = self._with_batch(full)
            legacy = {
                "left_arm": full[..., ARM_LEFT_SLICE],
                "right_arm": full[..., ARM_RIGHT_SLICE],
                "left_hand": full[..., LEFT_EE_SLICE],
                "right_hand": full[..., RIGHT_EE_SLICE],
            }

        for key, expected_dim in expected_dims.items():
            value = legacy[key]
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != expected_dim:
                raise ValueError(
                    f"N2 action '{key}' must be (1,T,{expected_dim}), got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"N2 action '{key}' contains NaN or Inf")
        return legacy

    def reset(self):
        result = super().reset()
        if self._history is not None:
            self._history.clear()
        return result

    def get_action(self, obs: dict) -> tuple[dict, dict]:
        if not self._history:
            self.observe(obs)
        response = self._send(
            "get_action",
            {"observation": self._policy_observation(), "options": None},
        )
        if isinstance(response, (list, tuple)) and len(response) == 2:
            action, info = response
        else:
            action, info = response, {}
        return self._legacy_action(action), info


# ── Async policy worker (background inference) ───────────────────────────────
#
# Mirrors the isaac_ros leapp consumer protocol: the worker captures an
# observation, queries the GR00T server, and caches (chunk, t_obs, version)
# in a thread-safe slot. The main loop walks `action_horizon` ticks of the
# current chunk, then at the boundary snapshots the latest cached chunk and
# resumes execution from a *time-aligned* start_step so we skip the steps
# of the new chunk that correspond to "the past."
#
# Only one thread owns the ZMQ socket. Reset between episodes is performed
# inside the worker via `request_reset()` so the socket isn't touched from
# the main thread.

class AsyncPolicyWorker:
    """Background producer for GR00T action chunks.

    Capture cadence is whatever the server can sustain (inference is the
    bottleneck). The main thread consumes the cache without blocking on
    inference, achieving receding-horizon control with minimal idle time.
    """

    def __init__(self, client: GR00TClient, obs_fn, log_every: int = 30):
        self._client = client
        self._obs_fn = obs_fn        # () -> obs dict
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()            # start paused
        self._reset_request = threading.Event()
        self._cache_lock = threading.Lock()
        # (chunk_dict, t_obs_monotonic, version)
        self._latest = None
        self._version = 0
        self._thread = None
        self._infer_count = 0
        self._log_every = log_every

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="async-policy-worker", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._pause.clear()  # wake from pause
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def pause(self):
        """Stop producing new chunks (used during go_home / reset)."""
        self._pause.set()

    def resume(self):
        """Resume producing. Drops any stale chunk from before the pause."""
        with self._cache_lock:
            self._latest = None
        self._pause.clear()

    def request_reset(self):
        """Have the worker call client.reset() on its next loop iteration.

        The ZMQ REQ socket is single-threaded; only the worker may touch it
        once it's running. We signal here and wait for the worker to acknowledge.
        """
        self._reset_request.set()
        # Block until the worker has cleared the flag (handled it).
        deadline = time.monotonic() + 5.0
        while self._reset_request.is_set():
            if time.monotonic() > deadline or self._stop.is_set():
                logger.warning("async worker: reset request timed out")
                return
            time.sleep(0.005)

    def get_latest(self):
        """Snapshot the latest cached chunk, or None if nothing cached yet.

        Returns (chunk_dict, t_obs_monotonic, version) or None.
        """
        with self._cache_lock:
            return self._latest

    def _loop(self):
        while not self._stop.is_set():
            if self._pause.is_set():
                # Even when paused, honour reset requests so they're not
                # blocked behind the next resume().
                if self._reset_request.is_set():
                    try:
                        self._client.reset()
                    except Exception as e:
                        logger.warning(f"async worker reset (paused) failed: {e}")
                    self._reset_request.clear()
                time.sleep(0.01)
                continue
            if self._reset_request.is_set():
                try:
                    self._client.reset()
                except Exception as e:
                    logger.warning(f"async worker reset failed: {e}")
                self._reset_request.clear()
                continue
            t_obs = time.monotonic()
            try:
                obs = self._obs_fn()
                chunk, _ = self._client.get_action(obs)
            except Exception as e:
                logger.warning(f"async inference iteration failed: {e}")
                time.sleep(0.05)
                continue
            with self._cache_lock:
                self._version += 1
                self._latest = (chunk, t_obs, self._version)
            self._infer_count += 1
            if self._log_every > 0 and self._infer_count % self._log_every == 0:
                logger.info(
                    f"async worker: {self._infer_count} chunks cached "
                    f"(latest version={self._version})"
                )


# ── Sharpa hand helpers (same as eval_h2.py) ──────────────────────────────────

def _sharpa_path():
    if _SHARPA_SDK not in sys.path:
        sys.path.insert(0, _SHARPA_SDK)


def connect_hand(side: str, speed_coeff: float = 0.5, current_coeff: float = 0.6):
    _sharpa_path()
    from sharpa import SharpaWaveManager, ControlMode, ControlSource, HandSide

    hand_side = HandSide.LEFT if side == "left" else HandSide.RIGHT
    manager   = SharpaWaveManager.get_instance()
    time.sleep(1.0)

    logger.info(f"[{side}] connecting to {hand_side.name} hand...")
    try:
        hand = manager.connect(hand_side)
    except Exception as e:
        logger.warning(f"[{side}] connect failed: {e}  — skipping")
        return None

    if hand is None:
        logger.warning(f"[{side}] manager.connect returned None — skipping")
        return None

    err = hand.set_control_mode(ControlMode.POSITION)
    if err.code != 0:
        logger.error(f"[{side}] set_control_mode failed: {err.message}")
        return None

    err = hand.set_speed_coeff(speed_coeff)
    if err.code != 0:
        logger.warning(f"[{side}] set_speed_coeff failed: {err.message}")

    err = hand.set_current_coeff(current_coeff)
    if err.code != 0:
        logger.warning(f"[{side}] set_current_coeff failed: {err.message}")

    err = hand.set_control_source(ControlSource.SDK)
    if err.code != 0:
        logger.error(f"[{side}] set_control_source failed: {err.message}")
        return None

    hand.start()
    hand.set_joint_position([0.0] * SHARPA_DOF, True)  # blocking: wait for hand to reach neutral
    logger.info(f"[{side}] Sharpa hand ready.")
    return hand


def disconnect_hand(hand, side: str):
    if hand is None:
        return
    _sharpa_path()
    from sharpa import SharpaWaveManager
    try:
        hand.set_joint_position([0.0] * SHARPA_DOF, True)
        time.sleep(0.5)
        hand.stop()
    except Exception:
        pass
    try:
        SharpaWaveManager.get_instance().disconnect_all()
    except Exception:
        pass
    logger.info(f"[{side}] disconnected.")


def read_hand_state(hand) -> np.ndarray:
    """Read 22 joint angles (radians) from either the SDK SharpaWave or the DDS client."""
    if hand is None:
        return np.zeros(SHARPA_DOF, dtype=np.float32)
    try:
        err, angles_deg = hand.get_joint_position_degree()
        # SDK returns an Error obj with .code; DDS client returns a plain int.
        code = getattr(err, "code", err)
        if code != 0:
            return np.zeros(SHARPA_DOF, dtype=np.float32)
        return np.array([math.radians(a) for a in angles_deg[:SHARPA_DOF]], dtype=np.float32)
    except Exception:
        return np.zeros(SHARPA_DOF, dtype=np.float32)


def _hand_diagnostic_snapshot(hand) -> tuple[np.ndarray, int, float]:
    """Return (joint radians, DDS sequence, DDS age seconds) without commanding."""
    if hand is None:
        return np.full(SHARPA_DOF, np.nan), -1, math.inf
    if hasattr(hand, "get_state_diagnostics"):
        snapshot = hand.get_state_diagnostics()
        positions = np.asarray(snapshot["positions_rad"], dtype=np.float64)
        if positions.size != SHARPA_DOF:
            padded = np.full(SHARPA_DOF, np.nan)
            padded[:min(positions.size, SHARPA_DOF)] = positions[:SHARPA_DOF]
            positions = padded
        return (
            positions,
            int(snapshot["sequence_id"]),
            float(snapshot["age_s"]),
        )
    return read_hand_state(hand).astype(np.float64), -1, math.nan


class EvalDiagnosticsRecorder:
    """Save synchronized 58-DoF targets/feedback and hand DDS freshness."""

    def __init__(self, output_dir: str, args):
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.metadata_json = json.dumps(
            {
                "task": args.task,
                "frequency_hz": args.frequency,
                "action_horizon": args.action_horizon,
                "policy_format": args.policy_format,
                "hands_source": args.hands_source,
                "right_hand_grip_assist": args.right_hand_grip_assist,
                "right_hand_grip_assist_strength": RIGHT_GRIP_ASSIST_STRENGTH,
                "right_hand_grip_assist_max_delta_rad": RIGHT_GRIP_ASSIST_MAX_DELTA_RAD,
                "layout": "left_arm[0:7],right_arm[7:14],left_hand[14:36],right_hand[36:58]",
                "units": "radians",
            },
            sort_keys=True,
        )
        self.episode = None
        self.rows = None

    def start_episode(self, episode: int):
        if self.rows:
            self.save_episode("aborted_before_next_episode")
        self.episode = int(episode)
        self.rows = {
            "wall_time_ns": [],
            "monotonic_s": [],
            "episode_step": [],
            "chunk_index": [],
            "chunk_step": [],
            "model_output_58_rad": [],
            "target_58_rad": [],
            "state_before_58_rad": [],
            "state_after_58_rad": [],
            "left_sequence_before": [],
            "right_sequence_before": [],
            "left_sequence_after": [],
            "right_sequence_after": [],
            "left_age_before_s": [],
            "right_age_before_s": [],
            "left_age_after_s": [],
            "right_age_after_s": [],
            "right_grip_assist_score": [],
            "right_grip_assist_delta_rad": [],
        }

    def record(
        self,
        *,
        episode_step: int,
        chunk_index: int,
        chunk_step: int,
        model_output_58,
        target_58,
        state_before_58,
        state_after_58,
        left_freshness_before,
        right_freshness_before,
        left_freshness_after,
        right_freshness_after,
        right_grip_assist_score,
        right_grip_assist_delta,
    ):
        if self.rows is None:
            return
        self.rows["wall_time_ns"].append(time.time_ns())
        self.rows["monotonic_s"].append(time.monotonic())
        self.rows["episode_step"].append(int(episode_step))
        self.rows["chunk_index"].append(int(chunk_index))
        self.rows["chunk_step"].append(int(chunk_step))
        self.rows["model_output_58_rad"].append(
            np.asarray(model_output_58, dtype=np.float64)
        )
        self.rows["target_58_rad"].append(np.asarray(target_58, dtype=np.float64))
        self.rows["state_before_58_rad"].append(np.asarray(state_before_58, dtype=np.float64))
        self.rows["state_after_58_rad"].append(np.asarray(state_after_58, dtype=np.float64))
        self.rows["right_grip_assist_score"].append(float(right_grip_assist_score))
        self.rows["right_grip_assist_delta_rad"].append(
            np.asarray(right_grip_assist_delta, dtype=np.float64)
        )
        for side, before, after in (
            ("left", left_freshness_before, left_freshness_after),
            ("right", right_freshness_before, right_freshness_after),
        ):
            self.rows[f"{side}_sequence_before"].append(before[1])
            self.rows[f"{side}_age_before_s"].append(before[2])
            self.rows[f"{side}_sequence_after"].append(after[1])
            self.rows[f"{side}_age_after_s"].append(after[2])

    def save_episode(self, status: str):
        if self.rows is None or self.episode is None:
            return None
        if not self.rows["episode_step"]:
            self.rows = None
            self.episode = None
            return None

        episode = self.episode
        path = os.path.join(
            self.output_dir,
            f"hand_diag_{self.run_id}_ep{episode:03d}_{status}.npz",
        )
        payload = {key: np.asarray(value) for key, value in self.rows.items()}
        payload["status"] = np.asarray(status)
        payload["metadata_json"] = np.asarray(self.metadata_json)
        np.savez_compressed(path, **payload)
        logger.info(
            f"58-DoF eval diagnostics saved: {path} "
            f"({len(payload['episode_step'])} control ticks, status={status})"
        )
        self.rows = None
        self.episode = None
        return path


# ── Sharpa hand helpers — DDS bridge path ────────────────────────────────────

def connect_hand_dds(side: str):
    """Construct a SharpaHandDDSClient that talks to the C++ bridge on Thor."""
    from teleop.robot_control.robot_hand_sharpa_dds import SharpaHandDDSClient
    logger.info(f"[{side}] creating DDS client (bridge on Thor must be running)...")
    client = SharpaHandDDSClient(side)
    client.set_joint_position([0.0] * SHARPA_DOF, False)
    logger.info(f"[{side}] DDS client ready.")
    return client


def disconnect_hand_dds(client, side: str):
    if client is None:
        return
    try:
        client.go_neutral()
        time.sleep(0.5)
    except Exception as e:
        logger.warning(f"[{side}] DDS go_neutral error: {e}")
    logger.info(f"[{side}] DDS client released.")


# ── home / countdown helpers (mirrors replay_episode.py) ─────────────────────

def _countdown(seconds: int, label: str):
    for i in range(seconds, 0, -1):
        print(f"  {label}: {i}s ", end="\r", flush=True)
        time.sleep(1.0)
    print(f"  {label}: done.          ")


def _go_home(arm_ctrl, left_hand, right_hand, hold_s: int):
    """Move arm to home, hands to neutral, then hold for hold_s seconds."""
    logger.info(f"Going to home position (hold {hold_s}s)...")
    if arm_ctrl is not None:
        arm_ctrl.ctrl_dual_arm_go_home()
    for side, hand in [("left", left_hand), ("right", right_hand)]:
        if hand is not None:
            hand.set_joint_position([0.0] * SHARPA_DOF, True)
    _countdown(hold_s, "holding home")


def _camera_failure_hold(arm_ctrl):
    """Freeze measured arm pose until the operator enters damping mode.

    Automatically moving home or opening the hands after vision loss can be
    unsafe near the task scene. Keep publishing a fixed measured arm target,
    leave the hand targets unchanged, and require explicit operator takeover.
    """
    try:
        hold_q = arm_ctrl.get_current_dual_arm_q().astype(np.float64)
        arm_ctrl.ctrl_dual_arm(hold_q, np.zeros(ARM_DOF, dtype=np.float64))
        logger.critical(
            "CAMERA FAILURE HOLD: policy commands stopped; arms hold the measured "
            "pose and hand targets remain unchanged. Enter damping mode with the "
            "remote, then press Ctrl+C once to exit."
        )
    except Exception as exc:
        logger.critical(
            f"CAMERA FAILURE HOLD could not latch measured arm pose: {exc}. "
            "Use the remote emergency/damping procedure immediately."
        )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Operator ended camera-failure hold.")


def _load_init_state_yaml(yaml_path):
    """Load an average-initial-state YAML produced by
    teleop/utils/compute_average_initial_state.py. Returns the nested dict
    under "states" (left_arm.qpos, right_arm.qpos, left_ee.qpos, right_ee.qpos)."""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict) or "states" not in payload:
        raise ValueError(
            f"{yaml_path} does not contain a top-level 'states' dict. "
            "Was this file generated by compute_average_initial_state.py?"
        )
    logger.info(
        f"Loaded init state from {yaml_path} "
        f"(first_n_frames={payload.get('first_n_frames')}, "
        f"num_episodes={payload.get('num_episodes')}, "
        f"num_frames={payload.get('num_frames')})."
    )
    return payload["states"]


def _go_to_init_state(arm_ctrl, left_hand, right_hand, init_state: dict, hold_s: int):
    """Move arm + hands to the recorded average initial posture and hold.

    init_state is the dict loaded from the YAML's "states" key, e.g.
        {"left_arm": {"qpos": [...7]}, "right_arm": {"qpos": [...7]},
         "left_ee":  {"qpos": [...22]}, "right_ee": {"qpos": [...22]}}
    Body waist is read-only on H2 and is ignored even if present.
    """
    if not init_state:
        return

    if arm_ctrl is not None:
        left_arm_q  = np.asarray(
            init_state.get("left_arm", {}).get("qpos", [0.0] * 7), dtype=np.float64
        )
        right_arm_q = np.asarray(
            init_state.get("right_arm", {}).get("qpos", [0.0] * 7), dtype=np.float64
        )
        if left_arm_q.size != 7 or right_arm_q.size != 7:
            raise ValueError(
                f"Init state arm dims wrong (got left={left_arm_q.size}, right={right_arm_q.size}); "
                f"expected 7 each."
            )
        arm_q_target = np.concatenate([left_arm_q, right_arm_q])
        logger.info(
            f"Going to init posture (hold {hold_s}s). "
            f"Arm rate-limit will smooth the move from current pose."
        )
        arm_ctrl.ctrl_dual_arm(arm_q_target, np.zeros(ARM_DOF, dtype=np.float64))

    if left_hand is not None:
        left_ee_q = init_state.get("left_ee", {}).get("qpos")
        if left_ee_q and len(left_ee_q) == SHARPA_DOF:
            left_hand.set_joint_position(list(map(float, left_ee_q)), True)
    if right_hand is not None:
        right_ee_q = init_state.get("right_ee", {}).get("qpos")
        if right_ee_q and len(right_ee_q) == SHARPA_DOF:
            right_hand.set_joint_position(list(map(float, right_ee_q)), True)

    _countdown(hold_s, "holding init posture")


# ── camera client factory ────────────────────────────────────────────────────

def _make_image_client(args):
    """Build a camera client and return (client, cam_config)."""
    if args.image_source == "dds":
        from teleop.utils.dds_image_client import DDSImageClient
        client = DDSImageClient(
            domain_id=args.camera_dds_domain,
            network_interface=args.camera_network_interface,
            warmup_timeout=args.cam_warmup_s,
        )
        cam_config = client.get_cam_config()
        logger.info(f"Camera client: native CycloneDDS domain {args.camera_dds_domain}")
    elif args.image_source == "ros":
        from teleop.utils.ros_image_client import ROSImageClient
        client = ROSImageClient(warmup_timeout=args.cam_warmup_s)
        cam_config = client.get_cam_config()
        logger.info("Camera client: ROS 2 (CycloneDDS)")
    else:
        from teleimager.image_client import ImageClient
        client = ImageClient(host=args.img_server_ip, request_bgr=True)
        cam_config = client.get_cam_config()
        logger.info(f"Camera client: ZMQ teleimager at {args.img_server_ip}")
    return client, cam_config


def _warm_cameras(img_client, timeout_s: float = 5.0):
    """Poll all four getters until each returns a non-None bgr or timeout."""
    getters = [
        ("head",        img_client.get_head_frame),
        ("left_wrist",  img_client.get_left_wrist_frame),
        ("right_wrist", img_client.get_right_wrist_frame),
    ]
    deadline = time.perf_counter() + timeout_s
    warmed = set()
    while len(warmed) < len(getters) and time.perf_counter() < deadline:
        for name, getter in getters:
            if name in warmed:
                continue
            f = getter()
            if f is not None and getattr(f, "bgr", None) is not None:
                warmed.add(name)
        if len(warmed) < len(getters):
            time.sleep(0.05)
    logger.info(f"Cameras warmed: {sorted(warmed)} ({len(warmed)}/{len(getters)})")


class CameraSafetyError(RuntimeError):
    """Raised when a required policy camera stops delivering fresh frames."""


class CameraWatchdog:
    """Monitor fresh frame sequence IDs independently of policy inference.

    DDSImageClient intentionally caches the latest decoded image. Checking only
    ``frame.bgr is not None`` therefore cannot distinguish a live camera from a
    publisher that died while an old image remained cached. This watchdog polls
    the three physical streams used by the policy and requires each sequence ID
    to keep advancing.
    """

    def __init__(
        self,
        img_client,
        head_eye: str,
        timeout_s: float,
        poll_s: float = 0.02,
    ):
        if timeout_s <= 0:
            raise ValueError("camera watchdog timeout must be positive")
        if poll_s <= 0:
            raise ValueError("camera watchdog poll interval must be positive")

        if head_eye == "right" and hasattr(img_client, "get_right_head_frame"):
            head_name = "head_right"
            head_getter = img_client.get_right_head_frame
        elif head_eye == "left" and hasattr(img_client, "get_left_head_frame"):
            head_name = "head_left"
            head_getter = img_client.get_left_head_frame
        else:
            head_name = "head"
            head_getter = img_client.get_head_frame

        self._streams = [
            (head_name, head_getter),
            ("wrist_left", img_client.get_left_wrist_frame),
            ("wrist_right", img_client.get_right_wrist_frame),
        ]
        self._timeout_s = float(timeout_s)
        self._poll_s = float(poll_s)
        self._stop = threading.Event()
        self._failed = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure_message = ""
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="camera-safety-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._timeout_s + self._poll_s))
            self._thread = None

    def _fail(self, message: str):
        with self._failure_lock:
            if not self._failed.is_set():
                self._failure_message = message
                logger.critical(f"CAMERA SAFETY WATCHDOG: {message}")
                self._failed.set()

    def raise_if_failed(self):
        if self._failed.is_set():
            with self._failure_lock:
                message = self._failure_message
            raise CameraSafetyError(message)

    def _loop(self):
        now = time.monotonic()
        last_sequence = {}
        last_advance = {name: now for name, _ in self._streams}

        while not self._stop.is_set() and not self._failed.is_set():
            now = time.monotonic()
            for name, getter in self._streams:
                try:
                    frame = getter()
                except Exception as exc:
                    self._fail(f"{name}: frame getter failed: {exc}")
                    break

                image = None if frame is None else getattr(frame, "bgr", None)
                if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                    self._fail(f"{name}: empty or invalid frame")
                    break

                sequence = getattr(frame, "sequence_id", None)
                if sequence is None:
                    self._fail(f"{name}: frame has no sequence_id; freshness cannot be verified")
                    break
                sequence = int(sequence)
                previous = last_sequence.get(name)
                if previous is None or sequence != previous:
                    last_sequence[name] = sequence
                    last_advance[name] = now
                else:
                    frozen_s = now - last_advance[name]
                    if frozen_s > self._timeout_s:
                        self._fail(
                            f"{name}: sequence {sequence} has not advanced for "
                            f"{frozen_s:.3f}s (limit {self._timeout_s:.3f}s)"
                        )
                        break

            self._stop.wait(self._poll_s)


# ── hand setup / teardown ────────────────────────────────────────────────────

def _setup_hands(args):
    """Connect both hands via the chosen source. Returns (left, right)."""
    if not args.hands:
        return None, None

    if args.hands_source == "dds":
        logger.info("Connecting hands via DDS bridge...")
        left  = connect_hand_dds("left")
        right = connect_hand_dds("right")
    else:
        logger.info("Connecting hands via direct Sharpa SDK...")
        left  = connect_hand("left",  args.hand_speed_coeff, args.hand_current_coeff)
        right = connect_hand("right", args.hand_speed_coeff, args.hand_current_coeff)
    logger.info("Waiting for hands to reach neutral...")
    time.sleep(2.0)
    return left, right


def _teardown_hands(args, left_hand, right_hand):
    if not args.hands:
        return
    if args.hands_source == "dds":
        disconnect_hand_dds(left_hand,  "left")
        disconnect_hand_dds(right_hand, "right")
    else:
        disconnect_hand(left_hand,  "left")
        disconnect_hand(right_hand, "right")


# ── observation helpers ───────────────────────────────────────────────────────

def _add_bt_dims(obs: dict) -> dict:
    """
    Add (B=1, T=1) leading dims to every numpy array in the nested obs dict.
    Strings become [[str]] (two levels of list wrapping).
    """
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v[np.newaxis, np.newaxis, ...]   # (H,W,3) → (1,1,H,W,3)
        elif isinstance(v, dict):
            out[k] = _add_bt_dims(v)
        else:
            out[k] = [[v]]   # str → [[str]]
    return out


def _split_head_image(stitched: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROSImageClient.get_head_frame() returns a (H, 2W, 3) horizontally-stitched
    stereo image. Split it back into (left_eye, right_eye)."""
    if stitched is None:
        return None, None
    w = stitched.shape[1] // 2
    return stitched[:, :w, :], stitched[:, w:, :]


def get_observations(img_client, arm_ctrl, left_hand, right_hand,
                     cam_config, task: str, head_eye: str = "left",
                     image_source: str = "ros",
                     swap_wrist_cams: bool = False) -> dict:
    """Build the nested GR00T observation dict (before adding batch/time dims)."""
    video: dict[str, np.ndarray] = {}

    if image_source in ("dds", "ros"):
        head_frame  = img_client.get_head_frame()
        l_wrist     = img_client.get_left_wrist_frame()
        r_wrist     = img_client.get_right_wrist_frame()
        # Optional: physical wiring sometimes swaps the wrist cameras. This flag
        # corrects it at the observation layer without re-wiring or re-recording.
        if swap_wrist_cams:
            l_wrist, r_wrist = r_wrist, l_wrist

        if head_frame is not None and head_frame.bgr is not None:
            left_eye, right_eye = _split_head_image(head_frame.bgr)
            if head_eye == "left":
                video["high"] = left_eye[:, :, ::-1].copy()
            elif head_eye == "right":
                video["high"] = right_eye[:, :, ::-1].copy()
            else:  # both
                video["left_eye_view"]  = left_eye[:, :, ::-1].copy()
                video["right_eye_view"] = right_eye[:, :, ::-1].copy()
        if l_wrist is not None and l_wrist.bgr is not None:
            video["left_wrist_view"]  = l_wrist.bgr[:, :, ::-1].copy()
        if r_wrist is not None and r_wrist.bgr is not None:
            video["right_wrist_view"] = r_wrist.bgr[:, :, ::-1].copy()
    else:  # legacy ZMQ teleimager
        cam_name_to_key = {
            "head_camera":        "high",
            "left_wrist_camera":  "left_wrist_view",
            "right_wrist_camera": "right_wrist_view",
        }
        # Swap which physical wrist feeds which key when requested.
        if swap_wrist_cams:
            getter_map = {
                "high":             img_client.get_head_frame,
                "left_wrist_view":  img_client.get_right_wrist_frame,
                "right_wrist_view": img_client.get_left_wrist_frame,
            }
        else:
            getter_map = {
                "high":             img_client.get_head_frame,
                "left_wrist_view":  img_client.get_left_wrist_frame,
                "right_wrist_view": img_client.get_right_wrist_frame,
            }
        for cam_name, cfg in cam_config.items():
            if not cfg.get("enable_zmq"):
                continue
            key    = cam_name_to_key.get(cam_name)
            getter = getter_map.get(key) if key else None
            if getter is None:
                continue
            frame = getter()
            if frame is not None and frame.bgr is not None:
                video[key] = frame.bgr[:, :, ::-1].copy()

    arm_q = arm_ctrl.get_current_dual_arm_q().astype(np.float32)
    state = {
        "left_arm":  arm_q[ARM_LEFT_SLICE],
        "right_arm": arm_q[ARM_RIGHT_SLICE],
        "left_hand":  read_hand_state(left_hand),
        "right_hand": read_hand_state(right_hand),
    }

    obs = {
        "video":    video,
        "state":    state,
        "language": {"annotation.human.task_description": task},
    }
    return _add_bt_dims(obs)


# ── check mode ────────────────────────────────────────────────────────────────

def check_observations(args):
    """
    Build one full observation dict (exactly as sent to the GR00T server) and
    print every field's key path, shape, dtype, and value range.
    Camera snapshots are saved to /tmp/check_<key>.png for visual inspection.
    """
    import cv2

    # DDS factory: needed for arm and for DDS-bridge hand client.
    need_dds = args.arm or (args.hands and args.hands_source == "dds") \
        or args.image_source in ("dds", "ros")
    if args.arm or (args.hands and args.hands_source == "dds"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(args.dds_domain)
        logger.info(f"DDS ChannelFactory initialised on domain {args.dds_domain}.")

    left_hand, right_hand = _setup_hands(args)

    arm_ctrl = None
    if args.arm:
        from teleop.robot_control.robot_arm import H2_ArmController
        logger.info(
            f"Initialising H2_ArmController (motion_mode={args.motion}, "
            f"head_pitch_home={args.head_pitch_home})..."
        )
        arm_ctrl = H2_ArmController(
            motion_mode=args.motion,
            head_pitch_home=args.head_pitch_home,
            kp_low=args.kp_low, kp_wrist=args.kp_wrist,
            kd_low=args.kd_low, kd_wrist=args.kd_wrist,
        )
        logger.info("H2_ArmController ready.")

    img_client, cam_config = _make_image_client(args)
    logger.info(f"Camera config keys: {list(cam_config.keys())}")
    _warm_cameras(img_client, timeout_s=args.cam_warmup_s)

    obs = get_observations(img_client, arm_ctrl, left_hand, right_hand,
                           cam_config, args.task,
                           head_eye=args.head_eye,
                           image_source=args.image_source,
                           swap_wrist_cams=args.swap_wrist_cams)

    def _print_nested(d: dict, prefix: str = ""):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _print_nested(v, path)
            elif isinstance(v, np.ndarray):
                logger.info(
                    f"  {path:45s}  shape={str(v.shape):20s}  dtype={v.dtype}"
                    f"  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}"
                )
            else:
                logger.info(f"  {path:45s}  value={v!r}")

    logger.info("\n=== Observation fields (after batch/time dims added) ===")
    _print_nested(obs)

    if not obs.get("video"):
        logger.error("video dict is EMPTY — no camera frames received.")

    logger.info("\n=== Camera snapshots ===")
    for key, arr in obs.get("video", {}).items():
        img_rgb = arr[0, 0]   # (1, 1, H, W, 3) → (H, W, 3) RGB
        img_bgr = img_rgb[:, :, ::-1].copy()
        save_path = f"/tmp/check_{key}.png"
        cv2.imwrite(save_path, img_bgr)
        logger.info(f"  [{key}] saved → {save_path}")

    logger.info("\nCheck complete.")
    _teardown_hands(args, left_hand, right_hand)
    try:
        img_client.close()
    except Exception:
        pass


# ── main eval loop ─────────────────────────────────────────────────────────────

def eval_h2_groot(args):
    if args.check:
        check_observations(args)
        return

    if args.policy_format == "n2" and args.async_policy:
        raise ValueError(
            "N2 real-robot mode currently requires synchronous policy calls so its "
            "30 Hz temporal observation history remains aligned. Do not pass --async-policy."
        )

    # DDS factory must be ready before any DDS publisher/subscriber is created
    # (H2_ArmController and SharpaHandDDSClient both need this).
    if args.arm or (args.hands and args.hands_source == "dds"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(args.dds_domain)
        logger.info(f"DDS ChannelFactory initialised on domain {args.dds_domain}.")

    left_hand, right_hand = _setup_hands(args)

    arm_ctrl = None
    if args.arm:
        from teleop.robot_control.robot_arm import H2_ArmController
        logger.info(
            f"Initialising H2_ArmController (motion_mode={args.motion}, "
            f"head_pitch_home={args.head_pitch_home})..."
        )
        arm_ctrl = H2_ArmController(
            motion_mode=args.motion,
            head_pitch_home=args.head_pitch_home,
            kp_low=args.kp_low, kp_wrist=args.kp_wrist,
            kd_low=args.kd_low, kd_wrist=args.kd_wrist,
        )
        arm_ctrl.speed_gradual_max()
        logger.info("H2_ArmController ready.")

    # Optional: load the average initial state YAML so the robot starts each
    # episode from the same posture distribution the policy saw during training.
    init_state = None
    if args.init_state_yaml:
        init_state = _load_init_state_yaml(args.init_state_yaml)

    _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s)
    if init_state is not None:
        _go_to_init_state(arm_ctrl, left_hand, right_hand, init_state,
                          hold_s=args.init_state_hold_s)

    img_client, cam_config = _make_image_client(args)
    logger.info(f"Camera config: {list(cam_config.keys())}")
    _warm_cameras(img_client, timeout_s=args.cam_warmup_s)
    if args.head_eye in ("left", "right"):
        left_wrist_source = "wrist_right" if args.swap_wrist_cams else "wrist_left"
        right_wrist_source = "wrist_left" if args.swap_wrist_cams else "wrist_right"
        logger.info(
            "Camera routing: "
            f"high <- head_{args.head_eye}, "
            f"left_wrist_view <- {left_wrist_source}, "
            f"right_wrist_view <- {right_wrist_source}"
        )

    # ── GR00T policy client ───────────────────────────────────────────────────
    logger.info(f"Connecting to GR00T server at {args.policy_host}:{args.policy_port} ...")
    client_class = GR00TN2Client if args.policy_format == "n2" else GR00TClient
    client = client_class(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.timeout_ms,
        video_key_map=VIDEO_KEY_STYLES[args.video_key_style],
    )
    for attempt in range(30):
        if client.ping():
            logger.info("GR00T server: connected.")
            break
        logger.info(f"  waiting for server... ({attempt+1}/30)")
        time.sleep(2.0)
    else:
        raise RuntimeError("Could not reach GR00T server after 60s. Is it running?")

    if isinstance(client, GR00TN2Client):
        client.configure()

    # ── prompt user ───────────────────────────────────────────────────────────
    user_input = input("\nPress [Enter] to start inference (or 'q' to quit): ")
    if user_input.lower() == "q":
        return

    logger.info(f"Inference config: {args.frequency} Hz, "
                f"action_horizon={args.action_horizon}, steps_per_episode={args.steps}, "
                f"task='{args.task}', async_policy={args.async_policy}")
    logger.info("Running: home(20s) → inference → home(20s) → inference → ...  Ctrl+C to stop.")

    dt       = 1.0 / args.frequency
    step_dt  = dt  # the chunk's per-step time spacing matches the consumer rate
    episode  = 0
    camera_watchdog = None
    if args.camera_watchdog_timeout_s > 0:
        if args.image_source != "dds":
            raise ValueError(
                "--camera-watchdog-timeout-s currently requires --image-source dds"
            )
        camera_watchdog = CameraWatchdog(
            img_client,
            head_eye=args.head_eye,
            timeout_s=args.camera_watchdog_timeout_s,
            poll_s=args.camera_watchdog_poll_s,
        )
        camera_watchdog.start()
        logger.info(
            "Camera safety watchdog enabled: "
            f"head_{args.head_eye} + wrist_left + wrist_right, "
            f"timeout={args.camera_watchdog_timeout_s:.3f}s, "
            f"poll={args.camera_watchdog_poll_s:.3f}s"
        )

    # ── optional async worker setup ───────────────────────────────────────────
    worker = None
    if args.async_policy:
        def _obs_capture_fn():
            return get_observations(img_client, arm_ctrl, left_hand, right_hand,
                                    cam_config, args.task,
                                    head_eye=args.head_eye,
                                    image_source=args.image_source,
                                    swap_wrist_cams=args.swap_wrist_cams)
        worker = AsyncPolicyWorker(client, _obs_capture_fn)
        worker.start()
        logger.info("Async policy worker started (paused).")

    diagnostics = None
    if args.eval_diagnostics_dir:
        if not args.arm or not args.hands:
            raise ValueError("--eval-diagnostics-dir requires both arms and hands enabled")
        diagnostics = EvalDiagnosticsRecorder(args.eval_diagnostics_dir, args)
        logger.info(
            "58-DoF eval diagnostics enabled: "
            f"{diagnostics.output_dir}"
        )
    if args.right_hand_grip_assist:
        logger.info(
            "Right-hand grip assist enabled: data-derived per-joint reference, "
            f"strength={RIGHT_GRIP_ASSIST_STRENGTH:.2f}, "
            f"max_delta={RIGHT_GRIP_ASSIST_MAX_DELTA_RAD:.3f}rad, "
            f"delta_slew={RIGHT_GRIP_ASSIST_DELTA_SLEW_RAD:.3f}rad/tick"
        )

    latched_safety_abort = None
    try:
        while True:
            if camera_watchdog is not None:
                camera_watchdog.raise_if_failed()
            episode += 1
            if worker is not None:
                # Pause worker, reset policy via worker (sole socket owner),
                # then resume so chunks start flowing again for this episode.
                worker.pause()
                worker.request_reset()
                worker.resume()
            else:
                client.reset()
            logger.info(f"\n=== Episode {episode} ===")
            if diagnostics is not None:
                diagnostics.start_episode(episode)

            # EMA state on the per-step action targets. Smooths high-frequency
            # noise from the policy output (visible as arm/hand jitter).
            # alpha == 1.0 -> raw action, no smoothing (backwards compatible).
            # smaller alpha -> heavier smoothing + slightly more lag.
            smooth_alpha = float(args.smooth_alpha)
            boundary_alpha = float(args.boundary_smooth_alpha)
            boundary_total = max(0, int(args.boundary_smooth_steps))
            blend_total = max(0, int(args.boundary_blend_steps))
            # alpha > 0 → exp(-alpha * t/N) decays. alpha <= 0 would give
            # w_old >= 1 forever — clamp to a tiny positive.
            blend_alpha = max(1e-6, float(args.boundary_blend_alpha))
            prev_left_arm = None
            prev_right_arm = None
            prev_left_hand = None
            prev_right_hand = None
            right_grip_assist_delta = np.zeros(SHARPA_DOF, dtype=np.float64)
            last_loaded_version = -1
            chunk_index = 0
            # Counter — how many remaining boundary-smoothing ticks for the
            # current chunk. Reset to boundary_total on every chunk switch.
            boundary_remaining = 0
            # Cross-chunk crossfade state: the just-finished chunk and the
            # step index where it would have continued. Used to blend the
            # old chunk's unexecuted future steps with the new chunk's first
            # few steps (same wall-clock-time pairs).
            prev_chunk = None
            prev_chunk_next_t = 0

            steps_executed = 0
            while steps_executed < args.steps:
                if camera_watchdog is not None:
                    camera_watchdog.raise_if_failed()
                # ── fetch next chunk ─────────────────────────────────────────
                if worker is not None:
                    # Wait for the worker to produce a chunk we haven't seen.
                    # While waiting we hold the previous target via the arm
                    # controller's internal latch (no new ctrl_dual_arm calls).
                    wait_started = time.monotonic()
                    while True:
                        latest = worker.get_latest()
                        if latest is not None and latest[2] > last_loaded_version:
                            break
                        if time.monotonic() - wait_started > 30.0:
                            raise RuntimeError("Async worker produced no chunk in 30s.")
                        time.sleep(args.async_cache_poll_s)
                    action_chunk, t_obs, version = latest
                    last_loaded_version = version

                    chunk_len = (action_chunk["left_arm"].shape[1]
                                 if "left_arm" in action_chunk else args.action_horizon)

                    # Time-aligned start_step: skip the steps of the new chunk
                    # whose intended execution time has already elapsed during
                    # observation + inference latency.
                    age = max(0.0, time.monotonic() - t_obs)
                    start_step = int(age / step_dt)
                    if start_step < 0:
                        start_step = 0
                    if start_step >= chunk_len:
                        start_step = chunk_len - 1
                else:
                    # Sync mode — current behavior, observation + inference
                    # blocks the loop here.
                    obs = get_observations(img_client, arm_ctrl, left_hand, right_hand,
                                           cam_config, args.task,
                                           head_eye=args.head_eye,
                                           image_source=args.image_source,
                                           swap_wrist_cams=args.swap_wrist_cams)
                    action_chunk, _ = client.get_action(obs)
                    if camera_watchdog is not None:
                        # Inference is blocking. A camera may have died while the
                        # request was running, so re-check before using its action.
                        camera_watchdog.raise_if_failed()
                    chunk_len = (action_chunk["left_arm"].shape[1]
                                 if "left_arm" in action_chunk else args.action_horizon)
                    start_step = 0

                horizon = min(args.action_horizon,
                              chunk_len - start_step,
                              args.steps - steps_executed)
                if horizon <= 0:
                    # The latest cached chunk is so old that start_step is
                    # already at the end. Drop it and try again next iter.
                    logger.warning(
                        f"chunk skipped: start_step={start_step} chunk_len={chunk_len}"
                    )
                    continue

                chunk_index += 1
                # New chunk just loaded → restart boundary-smoothing window.
                # (Only meaningful if prev_* is set; the very first chunk has
                # no anchor so EMA passes through raw values regardless.)
                boundary_remaining = boundary_total

                # Cross-chunk blend window: number of leading steps where we
                # mix prev_chunk[prev_next_t + i] with action_chunk[start+i].
                # Bounded by:
                #   * blend_total (user-requested),
                #   * prev chunk's unused future (chunk_A_len - prev_next_t),
                #   * new chunk's available horizon.
                if (blend_total > 0 and prev_chunk is not None
                        and "left_arm" in prev_chunk):
                    prev_chunk_len = prev_chunk["left_arm"].shape[1]
                    old_remaining = max(0, prev_chunk_len - prev_chunk_next_t)
                    blend_window = min(blend_total, old_remaining, horizon)
                else:
                    blend_window = 0

                if blend_window > 0:
                    # w_old at step 0 = 1.0 always; w_old at step N = exp(-alpha).
                    # We report the value at the LAST included step (N-1).
                    w_old_final = math.exp(-blend_alpha * (blend_window - 1) / blend_window)
                    logger.info(
                        f"boundary blend: {blend_window} steps "
                        f"(old @ {prev_chunk_next_t}..{prev_chunk_next_t + blend_window - 1} "
                        f"× new @ {start_step}..{start_step + blend_window - 1}, "
                        f"alpha={blend_alpha:.2f}, "
                        f"w_old: 1.000→{w_old_final:.3f})"
                    )

                # ── walk the chunk for `horizon` ticks (action_horizon or less) ──
                for offset in range(horizon):
                    if camera_watchdog is not None:
                        # This check precedes every arm/hand policy command.
                        camera_watchdog.raise_if_failed()
                    t = start_step + offset
                    t0 = time.perf_counter()

                    # Effective alpha for this tick — heavier blend right after
                    # a chunk switch to hide the discontinuity between two
                    # independently-sampled chunks; ramps back to user's
                    # smooth_alpha once boundary_remaining hits 0.
                    if boundary_remaining > 0:
                        effective_alpha = boundary_alpha
                        boundary_remaining -= 1
                    else:
                        effective_alpha = smooth_alpha

                    # Build raw targets — either pure new chunk, or a linear
                    # blend with the old chunk at the same wall-clock-time index.
                    if offset < blend_window:
                        old_t = prev_chunk_next_t + offset
                        # w_new ramps from 1/N to N/N over the blend window:
                        # old wins early, new wins late, clean handoff at the
                        # end of the window (w_new == 1.0 for offset == N-1).
                        # Exponential blend: w_old = exp(-alpha * t/N) where
                        # t = offset, N = blend_window. Step 0 starts with
                        # w_old=1.0 (pure old chunk) and decays toward
                        # exp(-alpha) at step N.
                        w_old = math.exp(-blend_alpha * offset / blend_window)
                        w_new = 1.0 - w_old
                        raw_left_arm  = (w_new * action_chunk["left_arm"][0][t].astype(np.float64)
                                         + w_old * prev_chunk["left_arm"][0][old_t].astype(np.float64))
                        raw_right_arm = (w_new * action_chunk["right_arm"][0][t].astype(np.float64)
                                         + w_old * prev_chunk["right_arm"][0][old_t].astype(np.float64))
                    else:
                        raw_left_arm  = action_chunk["left_arm"][0][t].astype(np.float64)
                        raw_right_arm = action_chunk["right_arm"][0][t].astype(np.float64)

                    if effective_alpha < 1.0:
                        if prev_left_arm is None:
                            left_arm = raw_left_arm
                            right_arm = raw_right_arm
                        else:
                            left_arm  = effective_alpha * raw_left_arm  + (1.0 - effective_alpha) * prev_left_arm
                            right_arm = effective_alpha * raw_right_arm + (1.0 - effective_alpha) * prev_right_arm
                    else:
                        left_arm = raw_left_arm
                        right_arm = raw_right_arm
                    prev_left_arm = left_arm
                    prev_right_arm = right_arm

                    if diagnostics is not None:
                        arm_state_before = arm_ctrl.get_current_dual_arm_q().astype(np.float64)
                        left_freshness_before = _hand_diagnostic_snapshot(left_hand)
                        right_freshness_before = _hand_diagnostic_snapshot(right_hand)

                    arm_q_target = np.concatenate([left_arm, right_arm])
                    arm_ctrl.ctrl_dual_arm(arm_q_target, np.zeros(ARM_DOF, dtype=np.float64))

                    left_hand_target = np.full(SHARPA_DOF, np.nan)
                    right_hand_target = np.full(SHARPA_DOF, np.nan)
                    right_grip_assist_score = math.nan
                    applied_right_grip_assist_delta = np.zeros(
                        SHARPA_DOF, dtype=np.float64
                    )
                    if args.hands:
                        if left_hand is not None and "left_hand" in action_chunk:
                            if offset < blend_window and "left_hand" in (prev_chunk or {}):
                                old_t = prev_chunk_next_t + offset
                                w_new = (offset + 1) / float(blend_window)
                                w_old = 1.0 - w_new
                                raw_lh = (w_new * np.asarray(action_chunk["left_hand"][0][t], dtype=np.float64)
                                          + w_old * np.asarray(prev_chunk["left_hand"][0][old_t], dtype=np.float64))
                            else:
                                raw_lh = np.asarray(action_chunk["left_hand"][0][t], dtype=np.float64)
                            if effective_alpha < 1.0 and prev_left_hand is not None:
                                lh = effective_alpha * raw_lh + (1.0 - effective_alpha) * prev_left_hand
                            else:
                                lh = raw_lh
                            prev_left_hand = lh
                            left_hand_target = np.asarray(lh, dtype=np.float64)
                            err = left_hand.set_joint_position(lh.tolist(), False)
                            # DDS client returns None; SDK returns Error obj.
                            if err is not None and getattr(err, "code", 0) != 0:
                                logger.warning(f"left hand error t={t}: {err.message}")
                        if right_hand is not None and "right_hand" in action_chunk:
                            if offset < blend_window and "right_hand" in (prev_chunk or {}):
                                old_t = prev_chunk_next_t + offset
                                w_new = (offset + 1) / float(blend_window)
                                w_old = 1.0 - w_new
                                raw_rh = (w_new * np.asarray(action_chunk["right_hand"][0][t], dtype=np.float64)
                                          + w_old * np.asarray(prev_chunk["right_hand"][0][old_t], dtype=np.float64))
                            else:
                                raw_rh = np.asarray(action_chunk["right_hand"][0][t], dtype=np.float64)
                            if effective_alpha < 1.0 and prev_right_hand is not None:
                                rh = effective_alpha * raw_rh + (1.0 - effective_alpha) * prev_right_hand
                            else:
                                rh = raw_rh
                            prev_right_hand = rh
                            right_hand_target = np.asarray(rh, dtype=np.float64)
                            if args.right_hand_grip_assist:
                                (
                                    right_hand_target,
                                    right_grip_assist_delta,
                                    right_grip_assist_score,
                                ) = _apply_right_hand_grip_assist(
                                    right_hand_target,
                                    right_grip_assist_delta,
                                )
                                applied_right_grip_assist_delta = (
                                    right_grip_assist_delta.copy()
                                )
                            err = right_hand.set_joint_position(
                                right_hand_target.tolist(), False
                            )
                            if err is not None and getattr(err, "code", 0) != 0:
                                logger.warning(f"right hand error t={t}: {err.message}")

                    elapsed = time.perf_counter() - t0
                    time.sleep(max(0.0, dt - elapsed))
                    if diagnostics is not None:
                        arm_state_after = arm_ctrl.get_current_dual_arm_q().astype(np.float64)
                        left_freshness_after = _hand_diagnostic_snapshot(left_hand)
                        right_freshness_after = _hand_diagnostic_snapshot(right_hand)
                        diagnostics.record(
                            episode_step=steps_executed,
                            chunk_index=chunk_index,
                            chunk_step=t,
                            model_output_58=np.concatenate(
                                [
                                    np.asarray(action_chunk["left_arm"][0][t], dtype=np.float64),
                                    np.asarray(action_chunk["right_arm"][0][t], dtype=np.float64),
                                    np.asarray(action_chunk["left_hand"][0][t], dtype=np.float64),
                                    np.asarray(action_chunk["right_hand"][0][t], dtype=np.float64),
                                ]
                            ),
                            target_58=np.concatenate(
                                [arm_q_target, left_hand_target, right_hand_target]
                            ),
                            state_before_58=np.concatenate(
                                [
                                    arm_state_before,
                                    left_freshness_before[0],
                                    right_freshness_before[0],
                                ]
                            ),
                            state_after_58=np.concatenate(
                                [
                                    arm_state_after,
                                    left_freshness_after[0],
                                    right_freshness_after[0],
                                ]
                            ),
                            left_freshness_before=left_freshness_before,
                            right_freshness_before=right_freshness_before,
                            left_freshness_after=left_freshness_after,
                            right_freshness_after=right_freshness_after,
                            right_grip_assist_score=right_grip_assist_score,
                            right_grip_assist_delta=applied_right_grip_assist_delta,
                        )
                    if isinstance(client, GR00TN2Client):
                        # N2 needs observations from the actual 30 Hz control
                        # timeline, not only one frame at each inference boundary.
                        client.observe(
                            get_observations(
                                img_client,
                                arm_ctrl,
                                left_hand,
                                right_hand,
                                cam_config,
                                args.task,
                                head_eye=args.head_eye,
                                image_source=args.image_source,
                                swap_wrist_cams=args.swap_wrist_cams,
                            )
                        )
                    steps_executed += 1

                logger.info(
                    f"chunk done ({horizon} steps from start_step={start_step}), "
                    f"total={steps_executed}/{args.steps}"
                )

                # Stash this chunk for cross-chunk blending at the next switch.
                # prev_chunk_next_t is the step the chunk would have continued
                # at if we hadn't switched (= start + horizon). Capped at
                # chunk_len so old_remaining = 0 when we ran the whole chunk.
                prev_chunk = action_chunk
                prev_chunk_next_t = start_step + horizon

            # Episode complete — pause inference so we don't waste cycles
            # while the robot is going home.
            if worker is not None:
                worker.pause()

            if diagnostics is not None:
                diagnostics.save_episode("complete")
            logger.info(f"Episode {episode} complete ({args.steps} steps). Going home...")
            _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s)
            if init_state is not None:
                _go_to_init_state(arm_ctrl, left_hand, right_hand, init_state,
                                  hold_s=args.init_state_hold_s)

    except CameraSafetyError as exc:
        latched_safety_abort = "camera"
        logger.critical(f"SAFETY STOP: camera stream lost; policy execution aborted: {exc}")
        if worker is not None:
            worker.pause()
        if diagnostics is not None:
            diagnostics.save_episode("camera_safety_abort")
        _camera_failure_hold(arm_ctrl)
    except HandCommandSafetyError as exc:
        latched_safety_abort = "right_hand_command"
        logger.critical(f"SAFETY STOP: invalid right-hand command; execution aborted: {exc}")
        if worker is not None:
            worker.pause()
        if diagnostics is not None:
            diagnostics.save_episode("hand_command_safety_abort")
        _camera_failure_hold(arm_ctrl)
    except KeyboardInterrupt:
        if diagnostics is not None:
            diagnostics.save_episode("interrupted")
        logger.info("Stopped by user.")
    finally:
        if diagnostics is not None:
            diagnostics.save_episode("aborted")
        if camera_watchdog is not None:
            try:
                camera_watchdog.stop()
            except Exception as e:
                logger.warning(f"camera watchdog stop failed: {e}")
        if worker is not None:
            try:
                worker.stop()
            except Exception as e:
                logger.warning(f"async worker stop failed: {e}")
        if latched_safety_abort is not None:
            logger.critical(
                f"{latched_safety_abort} safety abort complete: skipping automatic "
                "go-home and hand neutral commands. Robot must already be in "
                "operator-controlled damping mode before this process exits."
            )
        else:
            _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s)
            _teardown_hands(args, left_hand, right_hand)
        try:
            img_client.close()
        except Exception:
            pass
        logger.info("Done.")


def parse_args():
    ap = argparse.ArgumentParser(
        description="H2 + Sharpa real-robot eval using a remote GR00T policy server.")
    ap.add_argument("--check", action="store_true",
                    help="Check mode: verify camera inputs only, skip arm/hands/GR00T")
    ap.add_argument("--policy-host",   default="localhost",      help="GR00T server host")
    ap.add_argument("--policy-port",   type=int, default=5555,   help="GR00T server port")
    ap.add_argument(
        "--policy-format",
        choices=["n1.7", "n2"],
        default="n1.7",
        help="Policy-server schema. N2 uses flat keys and temporal history; it does not use RTC.",
    )
    ap.add_argument("--timeout-ms",    type=int, default=15000,  help="ZMQ recv timeout ms")
    ap.add_argument("--task",          default="",               help="Language instruction for the policy")
    ap.add_argument("--img-server-ip", default="192.168.124.162", help="Image server IP")
    ap.add_argument("--frequency",     type=float, default=30.0, help="Control frequency Hz")
    ap.add_argument("--action-horizon", type=int, default=16,  # matches training delta_indices=[0..15]
                    help="Number of steps to execute per action chunk before re-querying")
    ap.add_argument("--steps", type=int, default=30,
                    help="Total number of control steps to run before stopping (default: 30)")

    # Hand and arm enable/disable
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hands",    dest="hands", action="store_true",  default=True)
    g.add_argument("--no-hands", dest="hands", action="store_false")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arm",      dest="arm",   action="store_true",  default=True)
    g.add_argument("--no-arm",   dest="arm",   action="store_false")

    # Image source: native DDS (default), ROS 2, or legacy ZMQ teleimager.
    ap.add_argument("--image-source", choices=["dds", "ros", "zmq"], default="dds",
                    help="Camera source. 'dds' = native JPEG topics from Thor; "
                         "'ros' = sensor_msgs/Image topics; 'zmq' = legacy teleimager. "
                         "Default: dds.")
    ap.add_argument("--camera-dds-domain", type=int, default=10,
                    help="Native camera DDS domain; separate from robot domain 0. Default: 10.")
    ap.add_argument("--camera-network-interface", default=None,
                    help="Optional NIC name used for native camera DDS discovery.")
    ap.add_argument("--head-eye", choices=["left", "right", "both"], default="left",
                    help="Which head eye(s) to feed to the policy. 'left' or 'right' map to "
                         "video.high; 'both' produces video.left_eye_view + video.right_eye_view. "
                         "Default: left.")
    ap.add_argument("--video-key-style", choices=sorted(VIDEO_KEY_STYLES), default="high_view",
                    help="Video key names the N1.7 checkpoint was trained with. 'high_view' "
                         "sends video.high/left_wrist_view/right_wrist_view; 'head_right' sends "
                         "video.head_right/wrist_left/wrist_right. N2 ignores this — it reads "
                         "the names from the checkpoint. Default: high_view.")
    ap.add_argument("--cam-warmup-s", type=float, default=5.0,
                    help="How long to wait for the first frame from each camera (default: 5s).")
    ap.add_argument("--camera-watchdog-timeout-s", type=float, default=0.0,
                    help="Safety-stop policy execution if any required DDS camera sequence "
                         "does not advance for this many seconds. 0 disables. "
                         "The N2 launcher enables 0.25s by default.")
    ap.add_argument("--camera-watchdog-poll-s", type=float, default=0.02,
                    help="Required-camera watchdog polling interval. Default: 0.02s.")
    ap.add_argument("--swap-wrist-cams", action="store_true",
                    help="Swap left/right wrist camera streams. Use this when the physical "
                         "wiring puts the wrist cameras on the wrong sides — fixes the "
                         "observation at the eval-script layer without re-recording.")

    # Sharpa hands source: through the C++ DDS bridge on Thor or direct SDK locally.
    ap.add_argument("--hands-source", choices=["dds", "sdk"], default="dds",
                    help="How to drive Sharpa hands. 'dds' = publish HandCmd_ to "
                         "rt/sharpa/{left,right}/cmd via sharpa_dds_bridge on Thor. "
                         "'sdk' = direct Sharpa SDK locally (bridge must be stopped). "
                         "Default: dds.")
    ap.add_argument("--dds-domain", type=int, default=0,
                    help="DDS domain ID for the arm controller and DDS-bridge hand client.")
    ap.add_argument("--eval-diagnostics-dir", type=str, default=None,
                    help="Write one compressed NPZ per episode with synchronized 58-DoF "
                         "targets/states plus hand DDS freshness. Disabled by default.")
    ap.add_argument(
        "--right-hand-grip-assist",
        action="store_true",
        help="Slightly tighten selected right-hand flexion joints using a conservative "
             "reference derived from the top-quartile training demonstrations. Includes "
             "finite/range checks, a 0.10-rad assist cap, and 0.01-rad/tick slew limit. "
             "Disabled by default; omitting this flag preserves the previous command path.",
    )

    # H2 motion mode for standing replay (publish to rt/arm_sdk so WBC yields arms).
    ap.add_argument("--motion", action="store_true",
                    help="Enable H2 motion mode (publish on rt/arm_sdk with handover weight=1). "
                         "Required when the robot is standing on its legs.")
    ap.add_argument("--head-pitch-home", type=float, default=0.6, metavar="RAD",
                    help="H2 head pitch held during inference (radians). "
                         "Range -0.523 (up) to 0.837 (down). Default: 0.6.")

    # Hold timing
    ap.add_argument("--hold-s", type=int, default=20,
                    help="Seconds to hold at home before/between/after inference. Default: 20.")

    # Output smoothing — EMA on the per-step policy action.
    # 1.0 disables smoothing (raw); 0.3 - 0.6 is a typical range for noisy policies.
    ap.add_argument("--smooth-alpha", type=float, default=1.0,
                    help="EMA blend factor for action smoothing: target = alpha*raw + "
                         "(1-alpha)*prev. 1.0 = no smoothing (default). 0.3-0.6 quiets "
                         "high-frequency jitter from noisier models at the cost of mild lag.")

    # Optional: start each episode from a recorded average-initial-state YAML
    # produced by teleop/utils/compute_average_initial_state.py.
    ap.add_argument("--init-state-yaml", type=str, default=None,
                    help="Path to a YAML file with the average initial state "
                         "(states.left_arm.qpos, ...left_ee.qpos, etc.). When provided, the "
                         "robot moves to that posture after going home, before inference each "
                         "episode. Body waist is ignored (read-only on H2).")
    ap.add_argument("--init-state-hold-s", type=int, default=3,
                    help="Seconds to hold the init posture before starting/resuming inference. "
                         "Default: 3.")

    # H2 PD gains
    ap.add_argument("--kp-low",   type=float, default=None)
    ap.add_argument("--kp-wrist", type=float, default=None)
    ap.add_argument("--kd-low",   type=float, default=None)
    ap.add_argument("--kd-wrist", type=float, default=None)

    # Sharpa SDK tuning (only used when --hands-source sdk)
    ap.add_argument("--hand-speed-coeff",   type=float, default=0.5)
    ap.add_argument("--hand-current-coeff", type=float, default=0.6)

    # Async policy mode — run GR00T inference in a background thread so the
    # robot keeps executing while the next chunk is being computed. At each
    # chunk boundary, snap to the latest cached chunk and resume from a
    # time-aligned start_step (skip steps that already elapsed during
    # inference). Off by default to preserve the sync baseline.
    ap.add_argument("--async-policy", action="store_true",
                    help="Run GR00T inference concurrently with action execution.")
    ap.add_argument("--async-cache-poll-s", type=float, default=0.005,
                    help="How often the main loop polls for a fresh cached chunk "
                         "when blocked between chunks. Default: 5 ms.")

    # Boundary smoothing — heavier EMA blend for the first few ticks after
    # each chunk switch, to mask the discontinuity between two independently-
    # sampled chunks (diffusion noise + slightly different observations).
    # Default 1.0 / 0 steps = no boundary smoothing (matches old behavior).
    ap.add_argument("--boundary-smooth-alpha", type=float, default=1.0,
                    help="EMA alpha to use *only* during the first N ticks after "
                         "a chunk switch (where N = --boundary-smooth-steps). "
                         "Lower = stronger crossfade. Typical: 0.3.")
    ap.add_argument("--boundary-smooth-steps", type=int, default=0,
                    help="Number of consumer ticks at the start of each new chunk "
                         "during which --boundary-smooth-alpha replaces "
                         "--smooth-alpha. Typical: 3-6. 0 disables.")

    # Boundary blend — chunk-pair crossfade using the OLD chunk's still-unused
    # future steps. At a chunk switch, the two chunks' overlap steps describe
    # predictions for the SAME wall-clock time, so we linearly blend them
    # from "mostly old" → "mostly new" over N steps. Requires the model to
    # produce chunks longer than the action_horizon (otherwise the old
    # chunk has no unused future steps to blend with).
    ap.add_argument("--boundary-blend-steps", type=int, default=0,
                    help="Number of cross-chunk blend steps at each boundary. "
                         "Pairs old chunk's unexecuted future steps with new "
                         "chunk's leading steps and exponentially fades from "
                         "old to new. Typical: 5-8. Requires "
                         "chunk_len > action_horizon. 0 disables.")
    ap.add_argument("--boundary-blend-alpha", type=float, default=2.0,
                    help="Exponential decay rate for the cross-chunk blend. "
                         "Per-step old-chunk weight: w_old(t) = exp(-alpha * t / N) "
                         "where t is the step offset and N = --boundary-blend-steps. "
                         "Step 0: w_old=1.0 (pure old chunk). Step N: w_old=exp(-alpha). "
                         "Larger alpha → faster fade-out. Typical: 1.5-4.0. Default: 2.0.")
    return ap.parse_args()


if __name__ == "__main__":
    eval_h2_groot(parse_args())
