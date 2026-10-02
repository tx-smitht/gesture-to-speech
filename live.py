#!/usr/bin/env python3
"""Real-time decoding in the terminal: trackpad -> virtual electrode array -> network -> sounds, as you go.

    python live.py                          # uses models/decoder.pt
    python live.py --speak                  # say each word out loud when it ends
    python live.py --keep-words             # only your word-break move ends a word, not a long pause
    python live.py --readback               # say real words as soon as they're certain (see readback.py)
    python live.py --model models/other.pt

Every 20 ms a new bin of the signal arrives; every 80 ms the network looks at the last 280 ms and updates its
memory (see streaming.py -- the same code the web app uses).
"""

import argparse
import os
import sys
import time

import torch

from model import BLANK, load_model
from phonemes import ARPABET
from signals import BIN_S, HERE, RawTouches, bin_vector
from phonemes import WORD_BREAK
from streaming import StreamingDecoder, make_readback

from speech import speak, warm_up  # speak is also used by server.py


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(HERE, "models", "decoder.pt"))
    p.add_argument("--speak", action="store_true", help="speak each word when it ends")
    p.add_argument("--keep-words", action="store_true",
                   help="don't end a word on a long pause (use if your language has its own word-break move)")
    p.add_argument("--readback", nargs="?", const=0.9, type=float, metavar="THRESHOLD",
                   help="speak each real word as soon as it's this likely (default 0.9), often before its word "
                        "break; anything else is spoken when it ends")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model} -- train one first: python train.py")
    model, vocab, config = load_model(args.model)
    features = config.get("features", "basic")  # feed the model the same input it was trained on
    torch.set_num_threads(1)  # one tiny step every 80 ms: a single thread has the lowest latency
    rb = make_readback(vocab, [s for s in vocab if s not in (BLANK, WORD_BREAK)]) if args.readback else None
    decoder = StreamingDecoder(model, vocab, keep_words=args.keep_words, readback=rb,
                               threshold=args.readback or 0.9)
    shown = [0]  # read-back: how many words' spellings have been printed
    if args.speak:
        warm_up()
    print(f"Decoding sounds: {' '.join(vocab[1:])}.  Ctrl+C to stop.\n")

    def show(events):
        for kind, value in events:
            if kind == "sound":
                sys.stdout.write(f"{value}({ARPABET[value]}) ")
            else:
                if rb is not None:  # decoder.texts has one spelling per read-back word, in order
                    sys.stdout.write(f"=> {decoder.texts[shown[0]]}")
                    shown[0] += 1
                sys.stdout.write(" /  ")
                if args.speak:
                    speak(value)
            sys.stdout.flush()

    with RawTouches() as rt:
        next_t = time.monotonic()
        try:
            while True:
                next_t += BIN_S
                time.sleep(max(0.0, next_t - time.monotonic()))
                show(decoder.push(bin_vector(rt.take(), rt.pad, features)))
        except KeyboardInterrupt:
            show(decoder.flush())
    print()


if __name__ == "__main__":
    main()
