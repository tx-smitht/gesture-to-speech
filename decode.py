#!/usr/bin/env python3
"""Decode trackpad gestures into symbols using dictionary.json.

    python3 decode.py                              # live; Ctrl+C to stop
    python3 decode.py --speak                      # also say each symbol out loud
    python3 decode.py -v                           # also print each gesture's measurements
    python3 decode.py --file recordings/X.csv      # decode a saved recording instead

A symbol named "space" is written as a space in the running text.
"""

import argparse
import subprocess

from gestures import (DICTIONARY_PATH, MATCH_THRESHOLD, LiveTouches, describe, features, gestures_from_csv,
                      load_dictionary, rank)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speak", action="store_true", help="speak each decoded symbol with macOS `say`")
    p.add_argument("-v", "--verbose", action="store_true", help="print each gesture's measurements")
    p.add_argument("--file", help="decode a record.py CSV instead of the live trackpad")
    p.add_argument("--dict", default=DICTIONARY_PATH, help="dictionary file (default: dictionary.json)")
    args = p.parse_args()

    d = load_dictionary(args.dict)
    if not d:
        raise SystemExit("The dictionary is empty. Teach it first, e.g.: python3 teach.py A")
    print(f"Dictionary: {', '.join(sorted(d))}\n")

    text = []

    def handle(g):
        ranking = rank(d, g)
        if not ranking:
            n = len(features(g))
            sym, note = "?", f"(nothing in the dictionary uses {n} finger{'s' if n > 1 else ''})"
        else:
            dist, best = ranking[0]
            sym = best if dist <= MATCH_THRESHOLD else "?"
            note = f"{best} {dist:.2f}" if sym == "?" else f"{dist:.2f}"
            if len(ranking) > 1:
                note += f"   next: {ranking[1][1]} {ranking[1][0]:.2f}"
            if sym == "?":
                note = f"(nearest {note})"
        text.append(" " if sym == "space" else sym)
        print(f"{sym:<8} {note:<34} text: {''.join(text)}")
        if args.verbose:
            print(f"         {describe(g)}")
        if args.speak and sym != "?":
            subprocess.Popen(["say", sym])

    if args.file:
        for g in gestures_from_csv(args.file):
            handle(g)
        return

    print("Go. Ctrl+C to stop.\n")
    with LiveTouches() as src:
        try:
            for g in src.gestures():
                handle(g)
        except KeyboardInterrupt:
            pass
    print(f"\nText: {''.join(text)}")


if __name__ == "__main__":
    main()
