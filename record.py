#!/usr/bin/env python3
"""Record raw multitouch data from a Mac trackpad to CSV.

Uses Apple's private MultitouchSupport framework, which reports every finger
on the trackpad ~90-120 times per second -- not just where the cursor is.

Usage:
    python3 record.py                  # record until Ctrl+C
    python3 record.py --label hello    # tag the recording (goes in filename + metadata)
    python3 record.py --duration 10    # stop automatically after 10 seconds

Output (in ./recordings/):
    <timestamp>_<label>.csv   one row per finger per frame
    <timestamp>_<label>.json  metadata (trackpad size, start time, label, column docs)
"""

import argparse
import csv
import ctypes
import ctypes.util
import json
import os
import sys
import threading
import time
from ctypes import POINTER, Structure, byref, c_bool, c_double, c_float, c_int, c_long, c_void_p
from datetime import datetime

# ---------------------------------------------------------------------------
# Private framework bindings
# ---------------------------------------------------------------------------

MT = ctypes.CDLL("/System/Library/PrivateFrameworks/MultitouchSupport.framework/MultitouchSupport")
CF = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))


class MTVector(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("vx", c_float), ("vy", c_float)]


class MTTouch(Structure):
    # Reverse-engineered layout (96 bytes); field names follow OpenMultitouchSupport.
    _fields_ = [
        ("frame", c_int),
        ("timestamp", c_double),
        ("path_index", c_int),     # stable ID for one continuous contact (touch -> swipe -> lift)
        ("state", c_int),
        ("finger_id", c_int),
        ("hand_id", c_int),
        ("normalized", MTVector),  # position 0..1 (origin bottom-left), velocity in the same units/s
        ("z_total", c_float),      # contact size
        ("_field9", c_int),
        ("angle", c_float),        # orientation of the contact ellipse (radians)
        ("major_axis", c_float),
        ("minor_axis", c_float),
        ("absolute", MTVector),
        ("_field14", c_int),
        ("_field15", c_int),
        ("z_density", c_float),    # roughly: how hard/flat the finger is pressed
    ]


assert ctypes.sizeof(MTTouch) == 96, ctypes.sizeof(MTTouch)

ContactCallback = ctypes.CFUNCTYPE(c_int, c_void_p, POINTER(MTTouch), c_int, c_double, c_int)

MT.MTDeviceCreateList.restype = c_void_p
MT.MTRegisterContactFrameCallback.argtypes = [c_void_p, ContactCallback]
MT.MTUnregisterContactFrameCallback.argtypes = [c_void_p, ContactCallback]
MT.MTDeviceStart.argtypes = [c_void_p, c_int]
MT.MTDeviceStop.argtypes = [c_void_p]
MT.MTDeviceIsBuiltIn.argtypes = [c_void_p]
MT.MTDeviceIsBuiltIn.restype = c_bool
MT.MTDeviceGetSensorSurfaceDimensions.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
CF.CFArrayGetCount.argtypes = [c_void_p]
CF.CFArrayGetCount.restype = c_long
CF.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_long]
CF.CFArrayGetValueAtIndex.restype = c_void_p

STATE_NAMES = {
    0: "not_tracking",
    1: "start_in_range",
    2: "hover_in_range",
    3: "make_touch",     # finger just landed
    4: "touching",
    5: "break_touch",    # finger lifting off
    6: "linger_in_range",
    7: "out_of_range",   # finger gone
}

COLUMNS = [
    "t",            # seconds since recording started
    "frame",        # sensor frame counter (all fingers in one frame share it)
    "device",       # index of the trackpad (0 = first found)
    "touch_id",     # constant for one contact from landing to lifting
    "finger_id",
    "hand_id",
    "state",
    "x", "y",       # normalized 0..1, origin bottom-left
    "vx", "vy",     # normalized units per second
    "x_mm", "y_mm", # x/y scaled by the physical trackpad size
    "size",
    "pressure",
    "angle",
    "major_axis",
    "minor_axis",
]


def list_devices():
    lst = MT.MTDeviceCreateList()
    devices = []
    for i in range(CF.CFArrayGetCount(lst)):
        dev = CF.CFArrayGetValueAtIndex(lst, i)
        w, h = c_int(), c_int()
        MT.MTDeviceGetSensorSurfaceDimensions(dev, byref(w), byref(h))
        devices.append({
            "ref": dev,
            "index": i,
            "built_in": bool(MT.MTDeviceIsBuiltIn(dev)),
            "width_mm": w.value / 100.0,
            "height_mm": h.value / 100.0,
        })
    return devices


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self, devices, writer):
        self.devices = devices
        self.by_ref = {d["ref"]: d for d in devices}
        self.writer = writer
        self.lock = threading.Lock()
        self.t0 = None
        self.rows = 0
        self.live = {}  # touch_id -> (x_mm, y_mm), for the status line
        self.callback = ContactCallback(self._on_frame)  # keep a reference or it gets GC'd

    def _on_frame(self, device, touches, n, timestamp, frame):
        dev = self.by_ref.get(device)
        if dev is None:
            return 0
        with self.lock:
            if self.t0 is None:
                self.t0 = timestamp
            t = timestamp - self.t0
            live = {}
            for i in range(n):
                tc = touches[i]
                x, y = tc.normalized.x, tc.normalized.y
                x_mm, y_mm = x * dev["width_mm"], y * dev["height_mm"]
                self.writer.writerow([
                    f"{t:.6f}", frame, dev["index"], tc.path_index, tc.finger_id, tc.hand_id,
                    STATE_NAMES.get(tc.state, tc.state),
                    f"{x:.5f}", f"{y:.5f}", f"{tc.normalized.vx:.4f}", f"{tc.normalized.vy:.4f}",
                    f"{x_mm:.2f}", f"{y_mm:.2f}",
                    f"{tc.z_total:.4f}", f"{tc.z_density:.4f}", f"{tc.angle:.4f}",
                    f"{tc.major_axis:.3f}", f"{tc.minor_axis:.3f}",
                ])
                self.rows += 1
                if tc.state in (3, 4):
                    live[tc.path_index] = (x_mm, y_mm)
            self.live = live
        return 0

    def start(self):
        for d in self.devices:
            MT.MTRegisterContactFrameCallback(d["ref"], self.callback)
            MT.MTDeviceStart(d["ref"], 0)

    def stop(self):
        for d in self.devices:
            MT.MTUnregisterContactFrameCallback(d["ref"], self.callback)
            MT.MTDeviceStop(d["ref"])

    def status(self, started_at):
        with self.lock:
            fingers = "  ".join(f"#{k}({x:5.1f},{y:5.1f})" for k, (x, y) in sorted(self.live.items()))
            return f"\r{time.time() - started_at:6.1f}s  rows={self.rows:<7} fingers={len(self.live)}  {fingers}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="", help="what you're about to 'say', e.g. hello")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings"))
    args = p.parse_args()

    devices = list_devices()
    if not devices:
        sys.exit("No multitouch trackpad found.")

    os.makedirs(args.out, exist_ok=True)
    started = datetime.now()
    stem = started.strftime("%Y%m%d-%H%M%S") + (f"_{args.label}" if args.label else "")
    csv_path = os.path.join(args.out, stem + ".csv")
    meta_path = os.path.join(args.out, stem + ".json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        rec = Recorder(devices, writer)

        print(f"Recording to {csv_path}")
        for d in devices:
            kind = "built-in" if d["built_in"] else "external"
            print(f"  device {d['index']}: {kind} trackpad, {d['width_mm']:.1f} x {d['height_mm']:.1f} mm")
        print("Touch the trackpad. Ctrl+C to stop.\n")

        started_at = time.time()
        rec.start()
        try:
            while args.duration is None or time.time() - started_at < args.duration:
                sys.stdout.write(rec.status(started_at).ljust(100)[:120])
                sys.stdout.flush()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            rec.stop()
            time.sleep(0.05)  # let any in-flight callback finish before closing the file

    meta = {
        "label": args.label,
        "started_at": started.isoformat(),
        "duration_s": round(time.time() - started_at, 3),
        "rows": rec.rows,
        "devices": [{k: v for k, v in d.items() if k != "ref"} for d in devices],
        "columns": COLUMNS,
        "states": STATE_NAMES,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n\nSaved {rec.rows} rows -> {csv_path}")


if __name__ == "__main__":
    main()
