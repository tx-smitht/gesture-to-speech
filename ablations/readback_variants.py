#!/usr/bin/env python3
"""EXPERIMENT: two ways to make word-by-word read-back faster still -- drop the pause, and know the person's phrases.

Follows ablations/realtime_readback.py (read its note first: same decoder, policies and metrics).

QUESTION
    realtime_readback showed that a confidence threshold can speak each word ~0.1 s after its last sound instead of
    waiting for the sentence to end. Two more ideas, tested separately and together:

    A. CONTINUOUS TYPING. Today every word ends with a pause (or a space move) so the decoder knows where words
       stop. With a lexicon and LM, the decoder can find word boundaries itself: "S AH M TH IH NG" is "something",
       not "some thing", because that's what the words and context make likely. Dropping the pause makes every
       sentence shorter (more words per minute), and nothing has to wait for a pause to be detected.
    B. A PERSONAL LANGUAGE MODEL. People who communicate through a device repeat themselves a lot ("can you call
       the nurse", "i need my medicine"). An LM that has learned this person's usual sentences predicts their next
       word far better than a general one, so words become certain sooner -- often before they're finished.

HYPOTHESIS
    A. Continuous typing raises words per minute by ~25-35% (no ~0.7 s pause per word) at some cost in accuracy
       for the word-boundary mistakes it introduces ("some thing"), which the LM mostly prevents.
    B. A personal LM cuts error AND delay, and makes "early" (spoken-before-finished) words common. It's an upper
       bound for phrases the person has said before -- new sentences would behave like the corpus LM.

WHAT CHANGES
    pauses / continuous   the simulated person pauses 0.55-0.9 s between words / pauses between words exactly as
                          between sounds (0.12-0.3 s). WordBeamSearch(continuous=True) for the latter.
    corpus LM / personal  trigram LM learned from the "lm" sentences only (test sentences are new) / from all
                          corpus sentences, including the test ones (the person has said each of them before).

WHAT'S HELD FIXED
    Participant j11 (14% phoneme error, closest to Tom's real recordings); its phoneme decoder from
    realtime_readback (trained on random prompts WITH pauses -- it was never trained on continuous typing); the
    same val/test sentences, 3 performances each; alpha/beta and the learned policy re-tuned on val per condition.

METRICS
    As realtime_readback (WER of spoken words, latency after each word's last sound, % early), plus
    wpm         words per minute while making the sentence: words / (end of last sound - start of first sound).

CAVEATS
    - The phoneme decoder only ever saw pauses between words, so continuous typing is out of distribution for it
      (a real user would recalibrate with continuous prompts; this is a pessimistic test of A).
    - "personal" LM has seen the exact test sentences: it measures the best case for repeated phrases.

HOW TO RUN
    .venv/bin/python ablations/realtime_readback.py prepare     # once (trains the phoneme decoders)
    .venv/bin/python ablations/readback_variants.py             # ~10 min

RESULTS
    2026-10-01 -- participant j11 (14.3% phoneme error), 1104 test words. Spoken WER @ median delay after the word's
    last sound (negative = spoken before the word was finished):

        condition                 wpm   threshold 0.9      threshold 0.98     learned 0.95       sentence end
        pauses / corpus LM        42.2   6.5% @ +0.12 s     6.2% @ +0.21 s     5.8% @ +0.13 s     5.8% @ +3.90 s
        pauses / personal LM      42.2   5.1% @ -0.36 s     2.0% @ -0.28 s     2.2% @ -0.07 s     1.8% @ +3.81 s
        continuous / corpus LM    57.7  26.0% @ -0.02 s    12.7% @ +0.06 s    10.6% @ +0.54 s    10.7% @ +2.57 s
        continuous / personal LM  57.7   6.0% @ -0.33 s     2.0% @ -0.21 s     2.2% @ -0.20 s     1.8% @ +2.52 s

    - A confirmed: continuous typing is 37% more words per minute. With the general LM the error roughly doubles
      (word-boundary mistakes), so it needs a strong LM -- or a decoder recalibrated on continuous typing.
    - B confirmed: with a personal LM, threshold 0.98 speaks 66% of words before they're finished (median 0.28 s
      early) and 35% of all sounds were never needed, at sentence-end accuracy (2.0% vs 1.8%).
    - Together: 58 wpm, 2.0% spoken WER, the median word spoken 0.21 s before it's finished.
"""

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_readback import (OUT, PERFORMANCES, SEED, corpus_lm, evaluate, load_sentence_trials,  # noqa: E402
                               print_table, sentence_data)
from lexicon import Lexicon, load_pronunciations, read_sentences, split_of  # noqa: E402
from model import load_model  # noqa: E402
from phonemes import WORD_BREAK  # noqa: E402

PARTICIPANT, JITTER = "j11", 11.0
SETTINGS = [("word break", None), ("threshold", 0.8), ("threshold", 0.9), ("threshold", 0.95),
            ("threshold", 0.98), ("learned", 0.8), ("learned", 0.9), ("learned", 0.95), ("sentence end", None)]


def simulate_continuous(pron):
    """Val/test performances with no pause between words (same participant, same sentences)."""
    import simulate
    from signals import save_trial
    lang = json.load(open(os.path.join(OUT, f"sim_{PARTICIPANT}", f"train_{PARTICIPANT}_language.json")))
    rng = random.Random(200 + SEED)
    for part in ("val", "test"):
        path = os.path.join(OUT, f"{part}_{PARTICIPANT}cont.jsonl")
        if os.path.exists(path):
            continue
        for words in [s for s in read_sentences() if split_of(s) == part] * PERFORMANCES:
            prompt = [tok for w in words for tok in (*pron[w], WORD_BREAK)]
            frames, duration, timing = simulate.perform(prompt, lang, rng, JITTER, word_pause_s=simulate.PAUSE_S)
            save_trial(path, prompt, frames, duration, simulate.PAD, simulated=True, timing=timing, words=words)


def wpm(test):
    words = sum(len(d["truth"]) for _, d in test)
    seconds = sum(d["sound_ends"][-1] - d["first_start"] for _, d in test)
    return 60 * words / seconds


def main():
    import torch
    torch.set_num_threads(4)
    pron = load_pronunciations()
    lex = Lexicon(pron)
    simulate_continuous(pron)
    model, symbols, _ = load_model(os.path.join(OUT, f"decoder_{PARTICIPANT}.pt"))
    lms = {"corpus LM": corpus_lm(lex, pron), "personal LM": corpus_lm(lex, pron, ("lm", "val", "test"))}
    report = {}
    for typing, suffix in (("pauses", ""), ("continuous", "cont")):
        data = {}
        for part in ("val", "test"):
            data[part] = sentence_data(model, lex, pron, part, PARTICIPANT + suffix)
            for (_, d), t in zip(data[part], load_sentence_trials(part, PARTICIPANT + suffix)):
                d["first_start"] = t["timing"][0][0]
        for lm_name, lm in lms.items():
            print(f"\n=== {typing}, {lm_name}: {wpm(data['test']):.1f} words per minute while typing ===")
            ctx = {"lex": lex, "lm": lm, "symbols": symbols, "continuous": typing == "continuous"}
            rows, learned, alpha, beta = evaluate(data, ctx, SETTINGS, baselines=typing == "pauses", verbose=False)
            print_table(rows, data["test"])
            report[f"{typing} / {lm_name}"] = {"wpm": wpm(data["test"]), "alpha": alpha, "beta": beta, "rows": rows}
    json.dump(report, open(os.path.join(OUT, "variants.json"), "w"), indent=1)
    print("\nSUMMARY (threshold 0.9 and sentence end)")
    print(f"  {'condition':<28}{'wpm':>6}{'WER @0.9':>10}{'median':>9}{'early':>7}{'WER @end':>10}{'median':>9}")
    for cond, r in report.items():
        th = next(s for p, q, s in r["rows"] if p == "threshold" and q == "0.9")
        end = next(s for p, q, s in r["rows"] if p == "sentence end")
        print(f"  {cond:<28}{r['wpm']:6.1f}{th['wer']:9.1f}%{th['median']:+8.2f}s{th['early']:6.0f}%"
              f"{end['wer']:9.1f}%{end['median']:+8.2f}s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
