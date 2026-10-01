"""Turn raw touch frames into discrete gestures, describe them, and match them against a dictionary.

A *gesture* is everything from the moment the first finger lands until every finger has lifted.
It's a list of *strokes*, one per finger contact, each holding that finger's samples over time:
    {"t": [...], "x": [...], "y": [...], "major": [...], "minor": [...], "pressure": [...]}
(t in seconds from the start of the gesture, x/y in mm from the bottom-left corner.)

The dictionary (dictionary.json) maps each symbol to a list of example gestures you performed:
    {"A": [gesture, gesture, ...], "B": [...]}
Examples are stored raw, so changing the features below re-interprets all of them automatically.
"""

import csv
import itertools
import json
import math
import os
import queue
import statistics
import time

from record import STATE_NAMES, ContactCallback, MT, list_devices

HERE = os.path.dirname(os.path.abspath(__file__))
DICTIONARY_PATH = os.path.join(HERE, "dictionary.json")

DOWN_STATES = {3, 4, 5}  # make_touch, touching, break_touch -- hovering doesn't count
DOWN_STATE_NAMES = {STATE_NAMES[s] for s in DOWN_STATES}
GAP_S = 0.1  # no frames for this long = everything lifted
MIN_SAMPLES = 3  # ~24 ms; shorter contacts are ignored

# How big a difference in each feature counts as "one unit" of difference between two gestures.
# Smaller number = that feature matters more. Tune these as you learn what separates your symbols.
SCALES = {
    "x": 12.0,         # mm: where the finger landed
    "y": 12.0,
    "dx": 8.0,         # mm: how far it travelled from landing to lifting
    "dy": 8.0,
    "duration": 0.15,  # s: how long it stayed down
    "major": 1.5,      # mm: contact ellipse long axis  (a thumb is bigger than a fingertip)
    "minor": 1.5,      # mm: contact ellipse short axis
    "pressure": 0.5,   # peak pressure
}

# A gesture further than this from every example is reported as "?" instead of a symbol.
MATCH_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Segmentation: frames -> gestures
# ---------------------------------------------------------------------------

def _new_stroke():
    return {k: [] for k in ("t", "x", "y", "major", "minor", "pressure")}


class Segmenter:
    """Feed it one frame at a time; it hands back a finished gesture when the last finger lifts."""

    def __init__(self):
        self.active = {}    # touch key -> stroke still in contact
        self.finished = []  # strokes in the current gesture that already lifted
        self.last_t = None

    def feed(self, t, touches):
        """touches: list of dicts with key, down, x, y, major, minor, pressure."""
        done = None
        if self.last_t is not None and t - self.last_t > GAP_S:
            done = self.flush()
        self.last_t = t

        down = {tc["key"]: tc for tc in touches if tc["down"]}
        for key in list(self.active):
            if key not in down:
                self.finished.append(self.active.pop(key))
        for key, tc in down.items():
            s = self.active.setdefault(key, _new_stroke())
            s["t"].append(t)
            for k in ("x", "y", "major", "minor", "pressure"):
                s[k].append(tc[k])

        if not self.active and self.finished:
            done = self._emit()
        return done

    def flush(self):
        """Close the current gesture (if any) regardless of fingers still marked down."""
        self.finished.extend(self.active.values())
        self.active = {}
        return self._emit() if self.finished else None

    def _emit(self):
        # Contacts shorter than a few frames are palm/edge brushes, not intentional touches
        strokes = sorted((s for s in self.finished if len(s["t"]) >= MIN_SAMPLES), key=lambda s: s["t"][0])
        self.finished = []
        if not strokes:
            return None
        t0 = strokes[0]["t"][0]
        for s in strokes:
            s["t"] = [t - t0 for t in s["t"]]
        return strokes


class LiveTouches:
    """Context manager streaming gestures from the trackpad:

        with LiveTouches() as src:
            for gesture in src.gestures():
                ...
    """

    def __enter__(self):
        self.devices = list_devices()
        if not self.devices:
            raise SystemExit("No multitouch trackpad found.")
        self.by_ref = {d["ref"]: d for d in self.devices}
        self.q = queue.Queue()
        self.callback = ContactCallback(self._on_frame)  # keep a reference or it gets GC'd
        for d in self.devices:
            MT.MTRegisterContactFrameCallback(d["ref"], self.callback)
            MT.MTDeviceStart(d["ref"], 0)
        return self

    def __exit__(self, *exc):
        for d in self.devices:
            MT.MTUnregisterContactFrameCallback(d["ref"], self.callback)
            MT.MTDeviceStop(d["ref"])
        time.sleep(0.05)

    def _on_frame(self, device, touches, n, timestamp, frame):
        dev = self.by_ref.get(device)
        if dev is None:
            return 0
        frame_touches = []
        for i in range(n):
            tc = touches[i]
            frame_touches.append({
                "key": (dev["index"], tc.path_index),
                "down": tc.state in DOWN_STATES,
                "x": tc.normalized.x * dev["width_mm"],
                "y": tc.normalized.y * dev["height_mm"],
                "major": tc.major_axis,
                "minor": tc.minor_axis,
                "pressure": tc.z_density,
            })
        self.q.put((timestamp, frame_touches))
        return 0

    def gestures(self):
        seg = Segmenter()
        while True:
            try:
                t, touches = self.q.get(timeout=0.15)
            except queue.Empty:
                g = seg.flush()
            else:
                g = seg.feed(t, touches)
            if g:
                yield g


def gestures_from_csv(path):
    """Replay a record.py CSV through the same segmenter the live decoder uses."""
    seg = Segmenter()
    with open(path) as f:
        rows = csv.DictReader(f)
        for (_, _), frame_rows in itertools.groupby(rows, key=lambda r: (r["device"], r["frame"])):
            frame_rows = list(frame_rows)
            touches = [{
                "key": (r["device"], r["touch_id"]),
                "down": r["state"] in DOWN_STATE_NAMES,
                "x": float(r["x_mm"]),
                "y": float(r["y_mm"]),
                "major": float(r["major_axis"]),
                "minor": float(r["minor_axis"]),
                "pressure": float(r["pressure"]),
            } for r in frame_rows]
            g = seg.feed(float(frame_rows[0]["t"]), touches)
            if g:
                yield g
    g = seg.flush()
    if g:
        yield g


# ---------------------------------------------------------------------------
# Features and matching
# ---------------------------------------------------------------------------

def features(gesture):
    """One feature dict per finger, fingers ordered left to right by where they landed."""
    out = []
    for s in sorted(gesture, key=lambda s: s["x"][0]):
        out.append({
            "x": s["x"][0],
            "y": s["y"][0],
            "dx": s["x"][-1] - s["x"][0],
            "dy": s["y"][-1] - s["y"][0],
            "duration": s["t"][-1] - s["t"][0],
            "major": statistics.median(s["major"]),
            "minor": statistics.median(s["minor"]),
            "pressure": max(s["pressure"]),
        })
    return out


def distance(fa, fb):
    """0 = identical. Different finger counts can never match."""
    if len(fa) != len(fb):
        return math.inf
    total = sum(((a[k] - b[k]) / SCALES[k]) ** 2 for a, b in zip(fa, fb) for k in SCALES)
    return math.sqrt(total / len(fa))


def rank(dictionary, gesture):
    """[(distance, symbol), ...] closest first, using each symbol's nearest example."""
    f = features(gesture)
    scored = []
    for symbol, examples in dictionary.items():
        d = min((distance(f, features(ex)) for ex in examples), default=math.inf)
        if d < math.inf:
            scored.append((d, symbol))
    return sorted(scored)


def describe(gesture):
    fingers = features(gesture)
    parts = [
        f"landed ({f['x']:.0f},{f['y']:.0f}) moved ({f['dx']:+.0f},{f['dy']:+.0f}) mm, "
        f"{f['duration']:.2f}s, contact {f['major']:.1f}x{f['minor']:.1f} mm, pressure {f['pressure']:.2f}"
        for f in fingers
    ]
    n = len(fingers)
    return f"{n} finger{'s' if n > 1 else ''}: " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Dictionary file
# ---------------------------------------------------------------------------

def load_dictionary(path=DICTIONARY_PATH):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_dictionary(dictionary, path=DICTIONARY_PATH):
    def compact(s):
        return {"t": [round(v, 4) for v in s["t"]],
                **{k: [round(v, 2) for v in s[k]] for k in ("x", "y", "major", "minor", "pressure")}}

    data = {sym: [[compact(s) for s in g] for g in examples] for sym, examples in dictionary.items()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
