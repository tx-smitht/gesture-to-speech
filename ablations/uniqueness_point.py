#!/usr/bin/env python3
"""ANALYSIS: with perfect sound decoding, how early does each word become certain?

QUESTION
    Speaking a word early only works if the word is already certain before it's finished. How often is that true
    in principle -- before any decoding errors -- and how much of it comes from the word list versus the context?
    This is the "uniqueness point" of the cohort model of human listening: the sound at which only one word still
    fits. Here it's softened to "the point where one word has >= 90% of the probability".

WHAT'S COMPARED (no training, no simulated signal: sounds are assumed perfectly decoded)
    lexicon only     every word equally likely; certainty comes only from which words fit the sounds so far
    lexicon + LM     words weighted by the trigram LM given the TRUE previous words (english/corpus.txt "lm" split)
    personal LM      the same, but the LM has also seen the test sentences (someone repeating their usual phrases)

    For every word in the test sentences, the first point at which P(true word) >= 0.9:
        before       before its first sound (predicted from context alone)
        mid-word     after some but not all of its sounds
        at end       right after its last sound, before the pause that follows it
        needs pause  only once the pause shows the word is over ("go" vs "going": "go" is never certain until then)

HOW TO READ THE RESULT
    "before" + "mid-word" is the share of words a decoder could say before the person finishes them (and the person
    could skip the rest). "needs pause" words can never be spoken at the end of their last sound without guessing.
    Real decoding errors only push words later, so this is the best case.

HOW TO RUN
    uv run ablations/uniqueness_point.py      # a few seconds

RESULTS
    2026-10-01 -- 95 test sentences, 368 words:

                        before  mid-word   at end  needs pause  sounds saved
        lexicon only        0%       28%      48%          24%           14%
        lexicon + LM        0%       50%      42%           8%           26%
        personal LM         1%       71%      25%           2%           44%

    The word list alone settles about a quarter of words before their last sound; context doubles that. Only ~8% of
    words truly need the pause (prefixes of longer words), which is why the decoders' 90th-percentile delays rise.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lexicon import Lexicon, NgramLM, load_pronunciations, read_sentences, split_of  # noqa: E402

THRESHOLD = 0.9


def when_certain(lex, prior, truth):
    """First point where P(truth) >= THRESHOLD: -1 before any sound, i after i sounds (1..n), n+1 after the pause."""
    path = lex.path[truth]
    n = len(lex.prons[truth])
    for i, node in enumerate(path):                  # i sounds heard; the word might still continue
        share = prior * lex.under[node]
        if share[truth] / share.sum() >= THRESHOLD:
            return i - 1 if i == 0 else i
    return n + 1                                     # after the pause only the exact word fits


def main():
    pron = load_pronunciations()
    lex = Lexicon(pron)
    sents = read_sentences()
    units = lambda s: lex.units([pron[w] for w in s])
    lms = {"lexicon only": None,
           "lexicon + LM": NgramLM([units(s) for s in sents if split_of(s) == "lm"], range(len(lex))),
           "personal LM": NgramLM([units(s) for s in sents], range(len(lex)))}
    test = [units(s) for s in sents if split_of(s) == "test"]
    n_words = sum(len(s) for s in test)
    print(f"{len(test)} test sentences, {n_words} words; certain = P(true word) >= {THRESHOLD}\n")
    print(f"{'':<16}{'before':>9}{'mid-word':>10}{'at end':>9}{'needs pause':>13}{'sounds saved':>14}")
    for name, lm in lms.items():
        counts = np.zeros(4)
        saved = total = 0
        for s in test:
            for k, u in enumerate(s):
                prior = np.ones(len(lex)) if lm is None else lm.probs(s[:k])[:len(lex)]
                n = len(lex.prons[u])
                t = when_certain(lex, prior, u)
                counts[0 if t < 0 else 1 if t < n else 2 if t == n else 3] += 1
                saved += n - max(t, 0) if t < n else 0
                total += n
        pct = 100 * counts / counts.sum()
        print(f"{name:<16}{pct[0]:8.0f}%{pct[1]:9.0f}%{pct[2]:8.0f}%{pct[3]:12.0f}%{100 * saved / total:13.0f}%")


if __name__ == "__main__":
    main()
