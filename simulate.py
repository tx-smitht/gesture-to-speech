#!/usr/bin/env python3
"""A simulated participant: invents a secret trackpad language and "performs" calibration prompts with it.

    python simulate.py -n 300                  # uses your language.json sounds
    python simulate.py -n 600 --all-phonemes   # all 39 sounds -- a much harder language

Writes trials in exactly the format collect.py does (to data_sim/ by default), so the rest of the pipeline can't
tell the difference. Use it to check the pipeline learns before spending your own time recording.
"""

import argparse
import json
import os
import random

from phonemes import ARPABET, WORD_BREAK, load_inventory, make_prompt
from signals import HERE, save_trial

FRAME_S = 0.008       # the real trackpad reports every 8 ms
PAD = [157.8, 97.8]

# The fake person's sloppiness
POS_JITTER_MM = 3.0   # how far each tap lands from where they "meant" it
PAUSE_S = (0.12, 0.30)       # gap between sounds within a word
WORD_PAUSE_S = (0.55, 0.90)  # gap between words


def invent_language(sounds, rng):
    """Give every sound a distinct move: a spot on the pad x a motion x which finger."""
    spots = [(x, y) for x in (20, 48, 78, 108, 138) for y in (18, 50, 80)]
    motions = [(0, 0), (0, 15), (0, -15), (15, 0), (-15, 0)]  # tap, or a 15 mm swipe
    combos = [(s, m) for s in spots for m in motions]
    rng.shuffle(combos)
    lang = {}
    for sound, ((x, y), (dx, dy)) in zip(sounds, combos):
        lang[sound] = {"x": x, "y": y, "dx": dx, "dy": dy, "thumb": rng.random() < 0.25}
    return lang


def perform(prompt, lang, rng):
    """Turn a prompt into trackpad frames the way a (slightly sloppy) person would."""
    t = rng.uniform(0.4, 1.2)  # reaction time before starting
    strokes = []
    for i, tok in enumerate(prompt):
        if tok == WORD_BREAK:
            t += rng.uniform(*WORD_PAUSE_S)
            continue
        if i and prompt[i - 1] != WORD_BREAK:
            t += rng.uniform(*PAUSE_S)
        g = lang[tok]
        moving = g["dx"] or g["dy"]
        dur = rng.uniform(0.15, 0.30) if moving else rng.uniform(0.06, 0.14)
        strokes.append({
            "t0": t, "t1": t + dur,
            "x": g["x"] + rng.gauss(0, POS_JITTER_MM), "y": g["y"] + rng.gauss(0, POS_JITTER_MM),
            "dx": g["dx"] * rng.uniform(0.7, 1.3), "dy": g["dy"] * rng.uniform(0.7, 1.3),
            "major": rng.uniform(13, 15.5) if g["thumb"] else rng.uniform(8, 10),
            "minor": rng.uniform(7.2, 8.4),
            "pressure": rng.uniform(1.05, 1.45),
        })
        t += dur
    duration = t + rng.uniform(0.3, 0.8)  # until they press Enter

    frames = []
    for k in range(int(duration / FRAME_S)):
        ft = k * FRAME_S
        touches = []
        for sid, s in enumerate(strokes):
            if s["t0"] <= ft <= s["t1"]:
                a = (ft - s["t0"]) / (s["t1"] - s["t0"])  # 0 -> 1 through the stroke
                state = 3 if a < 0.1 else 5 if a > 0.9 else 4
                touches.append([sid, state, round(s["x"] + a * s["dx"], 2), round(s["y"] + a * s["dy"], 2),
                                round(s["major"], 2), round(s["minor"], 2), round(s["pressure"], 3)])
        if touches:
            frames.append((ft, touches))
    return frames, duration


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=300, help="number of trials")
    p.add_argument("--all-phonemes", action="store_true", help="use all 39 sounds instead of language.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "data_sim"))
    args = p.parse_args()

    rng = random.Random(args.seed)
    sounds = list(ARPABET) if args.all_phonemes else load_inventory()
    lang = invent_language(sounds, rng)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"sim_{len(sounds)}sounds_seed{args.seed}.jsonl")
    if os.path.exists(path):
        os.remove(path)
    for _ in range(args.n):
        prompt = make_prompt(sounds, rng)
        frames, duration = perform(prompt, lang, rng)
        save_trial(path, prompt, frames, duration, PAD, simulated=True)
    with open(path.replace(".jsonl", "_language.json"), "w") as f:
        json.dump(lang, f, indent=1)
    print(f"{args.n} simulated trials ({len(sounds)} sounds) -> {path}")


if __name__ == "__main__":
    main()
