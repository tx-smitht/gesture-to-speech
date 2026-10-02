#!/usr/bin/env python3
"""A simulated participant: invents a secret trackpad language and "performs" calibration prompts with it.

    python simulate.py -n 300                  # uses your language.json sounds
    python simulate.py -n 600 --all-phonemes   # all 39 sounds -- a much harder language
    python simulate.py --sentences english/corpus.txt --all-phonemes   # perform real English sentences

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


def perform(prompt, lang, rng, jitter_mm=POS_JITTER_MM, word_pause_s=WORD_PAUSE_S):
    """Turn a prompt into trackpad frames the way a (slightly sloppy) person would.
    Also returns when each sound's move started and ended ([t0, t1] per sound, word breaks skipped), so
    experiments can measure how long after a word was finished the decoder spoke it.
    word_pause_s: the gap between words. Set it to PAUSE_S for someone who doesn't pause between words at all."""
    t = rng.uniform(0.4, 1.2)  # reaction time before starting
    strokes = []
    for i, tok in enumerate(prompt):
        if tok == WORD_BREAK:
            t += rng.uniform(*word_pause_s)
            continue
        if i and prompt[i - 1] != WORD_BREAK:
            t += rng.uniform(*PAUSE_S)
        g = lang[tok]
        moving = g["dx"] or g["dy"]
        dur = rng.uniform(0.15, 0.30) if moving else rng.uniform(0.06, 0.14)
        strokes.append({
            "t0": t, "t1": t + dur,
            "x": g["x"] + rng.gauss(0, jitter_mm), "y": g["y"] + rng.gauss(0, jitter_mm),
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
    return frames, duration, [[round(s["t0"], 3), round(s["t1"], 3)] for s in strokes]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=300, help="number of trials")
    p.add_argument("--all-phonemes", action="store_true", help="use all 39 sounds instead of language.json")
    p.add_argument("--sentences", metavar="FILE",
                   help="perform these English sentences (one per line, words from english/pronunciations.txt) "
                        "instead of random prompts; -n is then how many times each sentence is performed")
    p.add_argument("--jitter", type=float, default=POS_JITTER_MM, help="how sloppy the taps are (mm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", help="output file name (default: sim_<sounds>sounds_seed<seed>)")
    p.add_argument("--out", default=os.path.join(HERE, "data_sim"))
    args = p.parse_args()

    rng = random.Random(args.seed)
    sounds = list(ARPABET) if args.all_phonemes else load_inventory()
    lang = invent_language(sounds, rng)  # drawn first, so the same seed is the same participant in every mode

    if args.sentences:
        from lexicon import load_pronunciations, read_sentences
        pron = load_pronunciations()
        jobs = [(words, [tok for w in words for tok in (*pron[w], WORD_BREAK)])
                for words in read_sentences(args.sentences) for _ in range(args.n)]
        rng.shuffle(jobs)
    else:  # a generator, so prompts and performances draw from rng in the same order as before
        jobs = ((None, make_prompt(sounds, rng)) for _ in range(args.n))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, (args.name or f"sim_{len(sounds)}sounds_seed{args.seed}") + ".jsonl")
    if os.path.exists(path):
        os.remove(path)
    n = 0
    for words, prompt in jobs:
        n += 1
        frames, duration, timing = perform(prompt, lang, rng, args.jitter)
        extra = {"words": words} if words else {}
        save_trial(path, prompt, frames, duration, PAD, simulated=True, timing=timing, **extra)
    with open(path.replace(".jsonl", "_language.json"), "w") as f:
        json.dump(lang, f, indent=1)
    print(f"{n} simulated trials ({len(sounds)} sounds) -> {path}")


if __name__ == "__main__":
    main()
