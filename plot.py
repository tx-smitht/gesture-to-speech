#!/usr/bin/env python3
"""Plot a recording: finger paths on the trackpad, and x/y over time.

Saves a PNG next to the CSV and opens it in Preview.

Usage:
    python3 plot.py                         # most recent recording
    python3 plot.py recordings/<file>.csv
"""

import csv
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob(os.path.join(HERE, "recordings", "*.csv")))
        if not files:
            sys.exit("No recordings yet -- run record.py first.")
        path = files[-1]

    meta_path = os.path.splitext(path)[0] + ".json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    dev = (meta.get("devices") or [{"width_mm": 1, "height_mm": 1}])[0]

    # Only real contacts (not hovering), grouped by touch_id
    strokes = defaultdict(lambda: {"t": [], "x": [], "y": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["state"] not in ("make_touch", "touching"):
                continue
            s = strokes[(row["device"], row["touch_id"])]
            s["t"].append(float(row["t"]))
            s["x"].append(float(row["x_mm"]))
            s["y"].append(float(row["y_mm"]))

    fig, (ax_pad, ax_time) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    title = os.path.basename(path) + (f"   label: {meta['label']}" if meta.get("label") else "")
    fig.suptitle(title)
    for ax in (ax_pad, ax_time):
        ax.set_prop_cycle(color=plt.get_cmap("tab20").colors)

    ax_pad.add_patch(plt.Rectangle((0, 0), dev["width_mm"], dev["height_mm"], fill=False, lw=1.5))
    for i, ((_, tid), s) in enumerate(sorted(strokes.items(), key=lambda kv: kv[1]["t"][0])):
        line, = ax_pad.plot(s["x"], s["y"], "-", lw=2, alpha=0.8)
        c = line.get_color()
        ax_pad.plot(s["x"][0], s["y"][0], "o", color=c)       # landing point
        ax_pad.annotate(str(i + 1), (s["x"][0], s["y"][0]), xytext=(4, 4), textcoords="offset points", color=c)
        ax_time.plot(s["t"], s["x"], "-", color=c)
        ax_time.plot(s["t"], s["y"], "--", color=c)

    ax_pad.set_xlim(-2, dev["width_mm"] + 2)
    ax_pad.set_ylim(-2, dev["height_mm"] + 2)
    ax_pad.set_aspect("equal")
    ax_pad.set_xlabel("x (mm)")
    ax_pad.set_ylabel("y (mm)")
    ax_pad.set_title(f"{len(strokes)} touches (dot = landing, number = order)")

    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("position (mm)")
    ax_time.set_title("x (solid) and y (dashed) over time")

    plt.tight_layout()
    png = os.path.splitext(path)[0] + ".png"
    fig.savefig(png, dpi=110)
    print(f"Saved {png}")
    subprocess.run(["open", png])


if __name__ == "__main__":
    main()
