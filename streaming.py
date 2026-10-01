"""Streaming decoding shared by live.py and the web app: feed one 20 ms bin at a time, get sounds as they appear.

It mirrors the offline path exactly (signals.cap_silence + Decoder.patch), so a model behaves live the way it
scored in training.

With `readback` (a readback.WordBeamSearch), words are no longer ended by the decoded word break: each word is
emitted the moment the word decoder is `threshold` sure of it -- often before its last sound -- and comes out as a
real word's sounds (or an unknown word's, see make_readback). Sound events are unchanged, for display.
"""

import time
from collections import deque

import numpy as np
import torch

from model import BLANK
from phonemes import WORD_BREAK
from signals import MAX_QUIET_BINS


def make_readback(vocab, inventory, unknown=-6.0, alpha=0.5, prefix_guard=True, beta=2.0):
    """A word decoder for live use: the common English words this inventory can say (lexicon.makeable_words),
    weighted by how common they are (alpha: 0 = all equally likely, 1 = by frequency); anything else comes out as
    an unknown word (its sounds), at a penalty. beta is a bonus per word, so that "nothing was said" (which pays no
    word cost at all) doesn't keep stealing probability from real but rare words. See ablations/readback_real.py
    for how these were chosen (best on real-word sentences made from Tom's own gestures)."""
    from lexicon import Lexicon, UnigramLM, makeable_words
    from readback import WordBeamSearch
    prons, ranks = makeable_words(inventory)
    lex = Lexicon(prons, ranks)
    return WordBeamSearch(lex, UnigramLM(lex, ranks, prons), vocab, alpha=alpha, beta=beta, unknown=unknown,
                          prefix_guard=prefix_guard)


class StreamingDecoder:
    def __init__(self, model, vocab, keep_words=False, readback=None, threshold=0.9):
        self.model, self.vocab, self.keep_words = model, vocab, keep_words
        self.readback, self.threshold = readback, threshold   # optional early word read-back
        self.texts = []                                 # read-back: how each emitted word is written ("see/sea")
        self.h, self.prev, self.last = None, None, None
        self.word = []                                  # sounds of the word in progress
        self.quiet = 0                                  # bins in a row with no touch
        self.window = deque(maxlen=model.patch_len)     # the last 280 ms of bins
        self.n_bins = 0                                 # bins fed to the network so far
        self.probs = None                               # latest symbol probabilities (for display)
        self.infer_ms = 0.0                             # how long the latest network step took
        self.stepped = False                            # did the latest push() run the network?

    def push(self, x):
        """One bin. Returns events: ("sound", sym) | ("word", [sounds]) -- a word just ended (or, with read-back,
        was just decided; its spelling is appended to self.texts)."""
        events = []
        self.stepped = False
        self.quiet = 0 if x.any() else self.quiet + 1
        if self.quiet > MAX_QUIET_BINS:
            # A long pause: skipped, exactly like signals.cap_silence does in training. It also ends the word in
            # progress unless the language has its own word-break move (keep_words).
            if self.readback is not None:
                if not self.keep_words and self.quiet == MAX_QUIET_BINS + 1:
                    events += self._readback_finish()
                return events
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
            if sym != WORD_BREAK:
                events.append(("sound", sym))
                self.word.append(sym)
            elif self.readback is None:
                events.append(("word", self.word))
                self.word = []
        self.prev = i
        if self.readback is not None:
            rb = self.readback
            rb.step(logits[0, -1].log_softmax(-1).numpy())
            while (w := rb.decide(self.threshold)) is not None:
                rb.commit(w)
                events.append(("word", rb.sounds(w)))
                self.texts.append(rb.name(w))
                self.word = []
        return events

    def _readback_finish(self):
        """Read-back: the sentence is over (long pause or stop) -- speak the best guess for whatever is left."""
        rb, events = self.readback, []
        for w in rb.best(finish=True)[len(rb.committed):]:
            events.append(("word", rb.sounds(w)))
            self.texts.append(rb.name(w))
        rb.reset()
        self.word = []
        return events

    def flush(self):
        """End of decoding: the unfinished word, if any."""
        if self.readback is not None:
            return self._readback_finish()
        word, self.word = self.word, []
        return [("word", word)] if word else []
