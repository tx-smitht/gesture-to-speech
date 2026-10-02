#!/usr/bin/env python3
"""Calibration session ("Copy Task"): say prompted sentences with your trackpad language.

    python collect.py          # 20 prompts
    python collect.py -n 50
    python collect.py --words   # prompts made of real English words your sounds can say ("see my team")

For each prompt, perform it on the trackpad, then press a key (no Enter needed):
    Enter            accept -- save this trial and move on
    r                redo this prompt (throw away what you just did)
    u / Backspace    undo the last SAVED trial: delete it and show its prompt again (repeat to go further back)
    q                quit
Every accepted trial is saved immediately to data/session_<time>.jsonl, so stopping early never loses anything.
"""

import argparse
import os
import sys
import termios
import time
import tty
from datetime import datetime

from phonemes import count_sounds, load_inventory, make_prompt, make_word_prompt, show
from signals import CONTACT_STATES, DATA_DIR, RawTouches, load_trials, remove_last_trial, save_trial

KEYS = {"\n": "accept", "\r": "accept", "r": "redo", "u": "undo", "\x7f": "undo", "\x08": "undo", "q": "quit"}


def read_key():
    """Wait for a single keypress (falls back to a whole line when input isn't a terminal)."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return (line.strip().lower() or "\n")[0] if line else "q"
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=20, help="number of prompts")
    p.add_argument("--words", action="store_true",
                   help="prompts of real English words (for testing word read-back) instead of random sounds")
    args = p.parse_args()

    inventory = load_inventory()
    path = os.path.join(DATA_DIR, f"session_{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    print(f"Sounds: {' '.join(inventory)}.  Saving to {path}")
    print("Perform each prompt, then:  Enter = save   r = redo   u/Backspace = undo last saved   q = quit\n")

    saved_prompts = []  # prompts of the trials saved so far, so undo can bring them back
    earlier = [t["prompt"] for t in load_trials(DATA_DIR)] if os.path.isdir(DATA_DIR) else []

    words = {}  # prompt (as a tuple) -> its English words, for --words prompts

    def next_prompt():
        if args.words:
            tokens, w = make_word_prompt(inventory, counts=count_sounds(earlier + saved_prompts))
            words[tuple(tokens)] = w
            return tokens
        # Favour sounds with the fewest examples, counting this session's saved trials too
        return make_prompt(inventory, counts=count_sounds(earlier + saved_prompts))

    counts = count_sounds(earlier)
    rare = sorted(inventory, key=lambda s: counts.get(s, 0))[:3]
    print("Fewest examples so far (prompts will favour these): "
          + ", ".join(f"{s} {counts.get(s, 0)}" for s in rare) + "\n")

    with RawTouches() as rt:
        prompt = next_prompt()
        try:
            while len(saved_prompts) < args.n:
                said = f"  {' '.join(words[tuple(prompt)]).upper()}:" if tuple(prompt) in words else ""
                print(f"[{len(saved_prompts) + 1}/{args.n}]{said}  {show(prompt)}")
                rt.take()  # discard anything before the prompt appeared
                t_start = time.monotonic()
                action = None
                while action is None:
                    action = KEYS.get(read_key())
                t_end = time.monotonic()
                frames = [(t - t_start, touches) for t, touches in rt.take()]

                if action == "quit":
                    break
                if action == "redo":
                    print("        redo")
                    continue
                if action == "undo":
                    if not saved_prompts:
                        print("        nothing saved yet to undo")
                        continue
                    remove_last_trial(path)
                    prompt = saved_prompts.pop()
                    print(f"        undid trial {len(saved_prompts) + 1} -- do it again")
                    continue
                if not any(tc[1] in CONTACT_STATES for _, touches in frames for tc in touches):
                    print("        (no touches recorded -- try again)")
                    continue
                extra = {"words": words[tuple(prompt)]} if tuple(prompt) in words else {}
                save_trial(path, prompt, frames, t_end - t_start, rt.pad, **extra)
                saved_prompts.append(prompt)
                print("        saved")
                prompt = next_prompt()
        except KeyboardInterrupt:
            pass

    print(f"\nSaved {len(saved_prompts)} trials to {path}")


if __name__ == "__main__":
    main()
