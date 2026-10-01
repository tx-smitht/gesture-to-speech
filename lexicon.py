"""Words for the word-level decoder: pronunciations, a prefix tree of them, and a small n-gram language model.

The decoder's output is *speech*, so a "word" here is a pronunciation, not a spelling: "to", "too" and "two" are
one entry (T UW). Homophones sound identical, so merging them costs nothing when the words are spoken aloud, and it
means the decoder never has to guess a spelling.

    english/pronunciations.txt   word -> ARPAbet sounds (hand-written, CMU Pronouncing Dictionary style)
    english/corpus.txt           everyday sentences: the language model learns from these

Pieces:
    Lexicon     every pronunciation as a path in a prefix tree ("trie"): sounds so far -> which words are still possible
    NgramLM     P(next word | previous two words), interpolated Kneser-Ney, learned by counting the corpus
"""

import math
import os
import zlib
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRONUNCIATIONS = os.path.join(HERE, "english", "pronunciations.txt")
CORPUS = os.path.join(HERE, "english", "corpus.txt")
START, END = "<s>", "</s>"  # sentence start / end, as in every n-gram model


def _lines(path):
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                yield line


def load_pronunciations(path=PRONUNCIATIONS):
    """{"water": ("W", "AO", "T", "ER"), ...}"""
    return {w: tuple(rest) for w, *rest in (line.split() for line in _lines(path))}


def read_sentences(path=CORPUS):
    """[["i", "need", "some", "water"], ...]"""
    return [line.split() for line in _lines(path)]


def split_of(sentence, val_frac=0.2, test_frac=0.2):
    """A fixed coin flip per sentence: "lm" (the language model may learn it), "val" (for tuning) or "test"."""
    r = zlib.crc32(" ".join(sentence).encode()) % 1000 / 1000
    return "test" if r < test_frac else "val" if r < test_frac + val_frac else "lm"


class Lexicon:
    """Pronunciations as a prefix tree. Node 0 is the root (no sounds yet); each node is a sequence of sounds, and a
    node "ends a word" if that sequence is a whole word's pronunciation.

        lex.child[node]["W"]   -> the node after one more sound (missing key: no word continues that way)
        lex.unit_at[node]      -> the word ending exactly here, or None
        lex.under[node]        -> 0/1 mask over words: which words are still possible from here (the "cohort")
    """

    def __init__(self, pronunciations):
        by_sound = defaultdict(list)
        for word, pron in sorted(pronunciations.items()):
            by_sound[pron].append(word)
        self.prons = sorted(by_sound)                                 # unit id -> pronunciation
        self.names = ["/".join(by_sound[p]) for p in self.prons]      # unit id -> "to/too"
        self.unit_of = {p: i for i, p in enumerate(self.prons)}       # pronunciation -> unit id
        self.child, self.unit_at, self.sound_in, self.depth = [{}], [None], [None], [0]
        self.path = []                                                # unit id -> nodes along its pronunciation
        for u, pron in enumerate(self.prons):
            node, path = 0, [0]
            for s in pron:
                if s not in self.child[node]:
                    self.child[node][s] = len(self.child)
                    self.child.append({})
                    self.unit_at.append(None)
                    self.sound_in.append(s)
                    self.depth.append(self.depth[node] + 1)
                node = self.child[node][s]
                path.append(node)
            self.unit_at[node] = u
            self.path.append(path)
        self.under = np.zeros((len(self.child), len(self.prons)), dtype=np.float64)
        for u, path in enumerate(self.path):
            self.under[path, u] = 1.0

    def __len__(self):
        return len(self.prons)

    def units(self, words):
        return [self.unit_of[p] for p in words]


class NgramLM:
    """Trigram language model with interpolated Kneser-Ney smoothing, over a closed vocabulary.

    Counting alone gives zero probability to any word pair the corpus never contains, which would forbid the
    decoder from ever outputting a new sentence. Kneser-Ney takes a little probability from every seen count
    (the discount D) and shares it out using a simpler model with less context (trigram -> bigram -> how many
    different words each word follows), so unseen combinations get small but non-zero probability.
    """

    def __init__(self, sentences, vocab, discount=0.75):
        self.vocab = list(vocab) + [END]
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.D = discount
        tri, bi = Counter(), Counter()
        for s in sentences:
            seq = [START, START, *s, END]
            for i in range(2, len(seq)):
                tri[seq[i - 2], seq[i - 1], seq[i]] += 1
        for (u, v, w) in tri:
            bi[v, w] += 1                                              # continuation counts: N1+(. v w)
        self.tri, self.bi = defaultdict(dict), defaultdict(dict)
        for (u, v, w), c in tri.items():
            self.tri[u, v][self.index[w]] = c
        for (v, w), c in bi.items():
            self.bi[v][self.index[w]] = c
        cont = Counter()
        for (v, w) in bi:
            cont[w] += 1                                               # how many different words precede w
        n = np.array([cont[w] for w in self.vocab], dtype=np.float64) + 0.5   # +0.5: every word stays possible
        self.unigram = n / n.sum()
        self._cache = {}

    def _level(self, counts, lower):
        if not counts:
            return lower
        total = sum(counts.values())
        p = lower * (self.D * len(counts) / total)
        for i, c in counts.items():
            p[i] += max(c - self.D, 0) / total
        return p

    def probs(self, context):
        """P(next | last two words of context) for every word in self.vocab (a numpy vector summing to 1)."""
        u, v = ([START, START] + list(context))[-2:]
        if (u, v) not in self._cache:
            self._cache[u, v] = self._level(self.tri.get((u, v)), self._level(self.bi.get(v), self.unigram))
        return self._cache[u, v]

    def perplexity(self, sentences):
        """How surprised the model is by these sentences: roughly, how many words it's choosing between each time."""
        logp, n = 0.0, 0
        for s in sentences:
            for i, w in enumerate([*s, END]):
                logp += math.log(self.probs(s[:i])[self.index[w]])
                n += 1
        return math.exp(-logp / n)
