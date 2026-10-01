#!/usr/bin/env python3
"""Real-time decoding in the terminal: trackpad -> virtual electrode array -> network -> sounds, as you go.

    python live.py                          # uses models/decoder.pt
    python live.py --speak                  # say each word out loud when it ends
    python live.py --keep-words             # only your word-break move ends a word, not a long pause
    python live.py --model models/other.pt

Every 20 ms a new bin of the signal arrives; every 80 ms the network looks at the last 280 ms and updates its
memory (see streaming.py -- the same code the web app uses).
"""

import argparse
import os
import subprocess
import sys
import time

import torch

from model import load_model
from phonemes import ARPABET
from signals import BIN_S, HERE, RawTouches, bin_vector
from streaming import StreamingDecoder

# ARPAbet -> the phoneme symbols macOS `say` understands in [[inpt PHON]] mode
MAC_PHONES = {
    "AA": "AA", "AE": "AE", "AH": "UX", "AO": "AO", "AW": "AW", "AY": "AY", "EH": "EH", "ER": "UXr", "EY": "EY",
    "IH": "IH", "IY": "IY", "OW": "OW", "OY": "OY", "UH": "UH", "UW": "UW",
    "B": "b", "CH": "C", "D": "d", "DH": "D", "F": "f", "G": "g", "HH": "h", "JH": "J", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "N", "P": "p", "R": "r", "S": "s", "SH": "S", "T": "t", "TH": "T", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "ZH": "Z",
}


def speak(sounds):
    if sounds:
        subprocess.Popen(["say", "[[inpt PHON]]" + "".join(MAC_PHONES[s] for s in sounds) + "[[inpt TEXT]]"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(HERE, "models", "decoder.pt"))
    p.add_argument("--speak", action="store_true", help="speak each word when it ends")
    p.add_argument("--keep-words", action="store_true",
                   help="don't end a word on a long pause (use if your language has its own word-break move)")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model} -- train one first: python train.py")
    model, vocab, config = load_model(args.model)
    features = config.get("features", "basic")  # feed the model the same input it was trained on
    torch.set_num_threads(1)  # one tiny step every 80 ms: a single thread has the lowest latency
    decoder = StreamingDecoder(model, vocab, keep_words=args.keep_words)
    print(f"Decoding sounds: {' '.join(vocab[1:])}.  Ctrl+C to stop.\n")

    def show(events):
        for kind, value in events:
            if kind == "sound":
                sys.stdout.write(f"{value}({ARPABET[value]}) ")
            else:
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
