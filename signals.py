"""Raw touch capture, trial storage, and the "virtual electrode array" that turns touches into network input.

A *trial* is one calibration sentence: what you were asked to say (`prompt`) and every trackpad frame recorded while
you said it. Trials are stored one per line in data/<session>.jsonl:
    {"prompt": ["EY", "|", "OW"], "duration": 4.2, "pad": [157.8, 97.8],
     "frames": [[t, [[touch_id, state, x_mm, y_mm, major, minor, pressure, angle, size], ...]], ...]}
angle (radians, the contact oval's orientation) and size (the trackpad's total-contact measure) are only in
recordings from 2026-10-01 on; older touches have the first seven fields.
"""

import glob
import json
import math
import os
import threading
import time

import numpy as np

from record import ContactCallback, MT, list_devices

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

CONTACT_STATES = {3, 4, 5}  # make_touch, touching, break_touch

# The virtual electrode array
GRID_COLS, GRID_ROWS = 16, 10  # 160 cells, each ~10 mm square on a built-in trackpad
N_CHANNELS = GRID_COLS * GRID_ROWS  # cells per map

# What each cell reports. "basic" is one map; "shape" adds maps for contact size and orientation, so the network can
# tell a thumb (big, tilted oval) from a fingertip (small, round) even when they land in the same place.
FEATURE_SETS = {
    "basic": ["activity"],
    "size": ["activity", "size"],
    "orientation": ["activity", "orient_cos", "orient_sin"],
    "shape": ["activity", "size", "orient_cos", "orient_sin"],
}
BIN_S = 0.02                   # 20 ms time bins, as in the BrainGate decoders
SIGMA_MM = 5.0                 # how far a finger's "blob" spreads to neighbouring cells
MAX_QUIET_S = 1.5              # thinking pauses longer than this are shortened to this (training and live alike)
MAX_QUIET_BINS = int(MAX_QUIET_S / BIN_S)
TYPICAL_AREA = 9.0 * 8.0       # fingertip contact (mm x mm); a bigger contact makes a brighter blob
TYPICAL_ELONGATION = 0.45      # a thumb's 1 - minor/major (~14 x 7.7 mm): scales orientation maps to ~1 like activity,
                               # so training's 0.05 augmentation noise doesn't drown the angle signal


# ---------------------------------------------------------------------------
# Live capture
# ---------------------------------------------------------------------------

class RawTouches:
    """Collects every trackpad frame in the background.

        with RawTouches() as rt:
            frames = rt.take()   # [(time, touches), ...] since the last take(); time = time.monotonic()
    """

    def __enter__(self):
        devices = list_devices()
        if not devices:
            raise SystemExit("No multitouch trackpad found.")
        self.device = devices[0]
        self.pad = [self.device["width_mm"], self.device["height_mm"]]
        self.lock = threading.Lock()
        self.frames = []
        self.callback = ContactCallback(self._on_frame)  # keep a reference or it gets GC'd
        MT.MTRegisterContactFrameCallback(self.device["ref"], self.callback)
        MT.MTDeviceStart(self.device["ref"], 0)
        return self

    def __exit__(self, *exc):
        MT.MTUnregisterContactFrameCallback(self.device["ref"], self.callback)
        MT.MTDeviceStop(self.device["ref"])
        time.sleep(0.05)

    def _on_frame(self, device, touches, n, timestamp, frame):
        now = time.monotonic()
        w, h = self.pad
        rows = []
        for i in range(n):
            tc = touches[i]
            rows.append([tc.path_index, tc.state,
                         round(tc.normalized.x * w, 2), round(tc.normalized.y * h, 2),
                         round(tc.major_axis, 2), round(tc.minor_axis, 2), round(tc.z_density, 3),
                         round(tc.angle, 3), round(tc.z_total, 3)])
        with self.lock:
            self.frames.append((now, rows))
        return 0

    def take(self):
        with self.lock:
            frames, self.frames = self.frames, []
        return frames


# ---------------------------------------------------------------------------
# Trial files
# ---------------------------------------------------------------------------

def save_trial(path, prompt, frames, duration, pad, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"prompt": prompt, "duration": round(duration, 3), "pad": pad,
           "frames": [[round(t, 4), touches] for t, touches in frames], **extra}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def remove_last_trial(path):
    """Delete the most recently saved trial from a session file."""
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    with open(path, "w") as f:
        f.writelines(lines[:-1])


def load_trials(*sources):
    """Trials from .jsonl files and/or directories of them (default: data/)."""
    paths = []
    for src in sources or [DATA_DIR]:
        paths += sorted(glob.glob(os.path.join(src, "*.jsonl"))) if os.path.isdir(src) else [src]
    trials = []
    for p in paths:
        with open(p) as f:
            for i, line in enumerate(line for line in f if line.strip()):
                trial = json.loads(line)
                trial["session"], trial["index"] = os.path.basename(p), i  # where it came from
                trials.append(trial)
    return trials


# ---------------------------------------------------------------------------
# Touches -> virtual electrode array
# ---------------------------------------------------------------------------

def n_channels(features="basic"):
    return len(FEATURE_SETS[features]) * N_CHANNELS


def _cell_centers(pad):
    w, h = pad
    cx = (np.arange(GRID_COLS) + 0.5) * w / GRID_COLS
    cy = (np.arange(GRID_ROWS) + 0.5) * h / GRID_ROWS
    gx, gy = np.meshgrid(cx, cy)          # rows = y, cols = x
    return gx.ravel().astype(np.float32), gy.ravel().astype(np.float32)


_centers_cache = {}


def frame_activity(touches, pad, features="basic"):
    """One frame -> one 160-cell map per feature, concatenated. Each finger lights nearby cells as a soft blob.

    activity    pressure x contact area           (the original single map)
    size        contact area relative to a fingertip
    orient_cos  cos(2 x angle) x elongation       angle repeats every 180 degrees, so it's doubled; round contacts
    orient_sin  sin(2 x angle) x elongation       have no real direction, so they're weighted towards 0
                                                  (elongation is scaled so a typical thumb is ~1)
    """
    key = (tuple(pad), GRID_COLS, GRID_ROWS)
    if key not in _centers_cache:
        _centers_cache[key] = _cell_centers(pad)
    gx, gy = _centers_cache[key]
    names = FEATURE_SETS[features]
    maps = np.zeros((len(names), N_CHANNELS), dtype=np.float32)
    for tc in touches:
        _id, state, x, y, major, minor, pressure = tc[:7]
        if state not in CONTACT_STATES:
            continue
        blob = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * SIGMA_MM ** 2))
        area = major * minor / TYPICAL_AREA
        angle = tc[7] if len(tc) > 7 else None  # older recordings have no angle
        elongation = (1 - minor / major) / TYPICAL_ELONGATION if major > 0 else 0.0
        for i, name in enumerate(names):
            if name == "activity":
                maps[i] = np.maximum(maps[i], pressure * area * blob)
            elif name == "size":
                maps[i] = np.maximum(maps[i], area * blob)
            elif angle is not None:  # orientation: fingers rarely overlap within a blob, so contributions add
                maps[i] += elongation * (math.cos(2 * angle) if name == "orient_cos" else math.sin(2 * angle)) * blob
    return maps.ravel()


def bin_vector(frames, pad, features="basic"):
    """All frames that fell in one 20 ms bin -> each cell's strongest reading (largest magnitude) in the bin."""
    if not frames:
        return np.zeros(n_channels(features), dtype=np.float32)
    stack = np.array([frame_activity(touches, pad, features) for _, touches in frames])
    strongest = np.abs(stack).argmax(axis=0)
    return stack[strongest, np.arange(stack.shape[1])].astype(np.float32)


def cap_silence(bins):
    """Keep at most MAX_QUIET_BINS of any no-touch stretch. live.py applies the same rule bin by bin."""
    keep, quiet = [], 0
    for i, row in enumerate(bins):
        quiet = 0 if row.any() else quiet + 1
        if quiet <= MAX_QUIET_BINS:
            keep.append(i)
    return bins[keep]


def featurize(trial, features="basic"):
    """A whole trial -> array [time bins, channels]. Uses bin_vector and cap_silence, exactly like live decoding."""
    n_bins = max(1, math.ceil(trial["duration"] / BIN_S))
    by_bin = [[] for _ in range(n_bins)]
    for t, touches in trial["frames"]:
        b = int(t / BIN_S)
        if 0 <= b < n_bins:
            by_bin[b].append((t, touches))
    return cap_silence(np.stack([bin_vector(fr, trial["pad"], features) for fr in by_bin]))
