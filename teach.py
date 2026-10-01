#!/usr/bin/env python3
"""Teach the dictionary a symbol by example.

    python3 teach.py A            # perform your gesture for "A" several times; Ctrl+C when done
    python3 teach.py --list       # what the dictionary knows
    python3 teach.py --list A     # every example of A, numbered
    python3 teach.py --remove A 1 2   # remove examples 1 and 2 of A
    python3 teach.py --undo A     # remove the most recent example of A (e.g. a slip)
    python3 teach.py --delete A   # forget A entirely

A gesture ends when every finger has lifted. 5-10 examples per symbol is a good start;
vary them a little the way you naturally would.
"""

import argparse

from gestures import DICTIONARY_PATH, MATCH_THRESHOLD, LiveTouches, describe, load_dictionary, rank, save_dictionary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbol", nargs="?", help="the symbol/sound you're about to perform, e.g. A or ah")
    p.add_argument("--list", nargs="?", const="", metavar="SYMBOL", help="all symbols, or every example of one")
    p.add_argument("--remove", nargs="+", metavar=("SYMBOL", "N"), help="remove examples by number (see --list SYMBOL)")
    p.add_argument("--undo", metavar="SYMBOL")
    p.add_argument("--delete", metavar="SYMBOL")
    p.add_argument("--dict", default=DICTIONARY_PATH, help="dictionary file (default: dictionary.json)")
    args = p.parse_args()

    d = load_dictionary(args.dict)

    if args.list is not None:
        if args.list:  # one symbol: every example, numbered
            for i, g in enumerate(d.get(args.list, []), 1):
                print(f"{i:>3}  {describe(g)}")
            return
        if not d:
            print("Dictionary is empty.")
        for sym, examples in sorted(d.items()):
            print(f"{sym:<10} {len(examples):>3} examples   latest: {describe(examples[-1])}")
        return

    if args.remove:
        sym, *nums = args.remove
        if sym not in d:
            raise SystemExit(f"{sym!r} isn't in the dictionary.")
        drop = {int(n) for n in nums}
        d[sym] = [g for i, g in enumerate(d[sym], 1) if i not in drop]
        if not d[sym]:
            del d[sym]
        save_dictionary(d, args.dict)
        print(f"Removed example(s) {', '.join(map(str, sorted(drop)))} of {sym!r}; {len(d.get(sym, []))} left.")
        return

    if args.undo or args.delete:
        sym = args.undo or args.delete
        if sym not in d:
            raise SystemExit(f"{sym!r} isn't in the dictionary.")
        if args.delete or len(d[sym]) == 1:
            del d[sym]
            print(f"Removed {sym!r}.")
        else:
            d[sym].pop()
            print(f"Removed the latest example of {sym!r}; {len(d[sym])} left.")
        save_dictionary(d, args.dict)
        return

    if not args.symbol:
        p.error("give a symbol to teach, or --list / --undo / --delete")

    sym = args.symbol
    print(f"Teaching {sym!r} ({len(d.get(sym, []))} examples so far).")
    print("Perform the gesture, lift every finger, repeat. Ctrl+C when done.\n")

    with LiveTouches() as src:
        try:
            for g in src.gestures():
                ranking = rank(d, g)  # what the dictionary thinks *before* learning this one
                d.setdefault(sym, []).append(g)
                save_dictionary(d, args.dict)
                print(f"#{len(d[sym]):<3} {describe(g)}")
                if ranking:
                    dist, best = ranking[0]
                    if dist > MATCH_THRESHOLD:
                        print(f"      new territory: nothing taught is close (nearest {best!r}, distance {dist:.2f})")
                    elif best == sym:
                        print(f"      already reads as {sym!r} (distance {dist:.2f})")
                    else:
                        print(f"      !! looks like {best!r} (distance {dist:.2f}) -- these two may get confused")
        except KeyboardInterrupt:
            pass

    print(f"\n{sym!r} now has {len(d.get(sym, []))} examples. (Slip? python3 teach.py --undo {sym})")


if __name__ == "__main__":
    main()
