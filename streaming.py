"""Streaming decoding shared by live.py and the web app: feed one 20 ms bin at a time, get sounds as they appear.

It mirrors the offline path exactly (signals.cap_silence + Decoder.patch), so a model behaves live the way it
scored in training.
"""

import time
from collections import deque

import numpy as np
import torch

from model import BLANK
from phonemes import WORD_BREAK
from signals import MAX_QUIET_BINS


class StreamingDecoder:
    def __init__(self, model, vocab, keep_words=False):
        self.model, self.vocab, self.keep_words = model, vocab, keep_words
        self.h, self.prev, self.last = None, None, None
        self.word = []                                  # sounds of the word in progress
        self.quiet = 0                                  # bins in a row with no touch
        self.window = deque(maxlen=model.patch_len)     # the last 280 ms of bins
        self.n_bins = 0                                 # bins fed to the network so far
        self.probs = None                               # latest symbol probabilities (for display)
        self.infer_ms = 0.0                             # how long the latest network step took
        self.stepped = False                            # did the latest push() run the network?

    def push(self, x):
        """One bin (160 values). Returns events: ("sound", sym) | ("word", [sounds]) -- a word just ended."""
        events = []
        self.stepped = False
        self.quiet = 0 if x.any() else self.quiet + 1
        if self.quiet > MAX_QUIET_BINS:
            # A long pause: skipped, exactly like signals.cap_silence does in training. It also ends the word in
            # progress unless the language has its own word-break move (keep_words).
            if self.word and not self.keep_words:
                events.append(("word", self.word))
                self.word, self.last = [], WORD_BREAK
            return events

        self.window.append(x)
        self.n_bins += 1
        # Same schedule as Decoder.patch(): first step once 14 bins exist, then every 4 bins
        if self.n_bins < self.model.patch_len or (self.n_bins - self.model.patch_len) % self.model.patch_stride:
            return events

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, self.h = self.model(torch.from_numpy(np.stack(self.window).reshape(1, 1, -1)), self.h)
        self.infer_ms = (time.perf_counter() - t0) * 1000
        self.stepped = True
        self.probs = logits[0, -1].softmax(-1).numpy()

        i = int(self.probs.argmax())
        sym = self.vocab[i]
        if i != self.prev and sym != BLANK and not (sym == WORD_BREAK and self.last in (WORD_BREAK, None)):
            self.last = sym
            if sym == WORD_BREAK:
                events.append(("word", self.word))
                self.word = []
            else:
                events.append(("sound", sym))
                self.word.append(sym)
        self.prev = i
        return events

    def flush(self):
        """End of decoding: the unfinished word, if any."""
        word, self.word = self.word, []
        return [("word", word)] if word else []
