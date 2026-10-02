#!/usr/bin/env python3
"""EXPERIMENT: word read-back on Tom's real recordings and decoder (not simulated).

QUESTION
    realtime_readback.py showed, on simulated participants with all 39 sounds, that speaking each word once the
    word decoder is >= 90-95% sure is about as accurate as waiting for the sentence, and much sooner. Does it work
    with Tom's real decoder and recordings -- 10 sounds, and calibration prompts made of RANDOM sounds?

    Two differences from the simulation matter:
      - Few English words can be said with 10 sounds (70 common ones: see, my, time, team, eat...), so the word
        list is small, and the "language model" is just how common each word is (lexicon.UnigramLM).
      - Most calibration "words" are random sound strings, not English. The decoder therefore needs an
        UNKNOWN-WORD fallback (readback.WordBeamSearch(unknown=...)): any sound string can still come out, at a
        penalty, so real words are preferred only when the signal fits them about as well.

WHAT CHANGES
    today              streaming.StreamingDecoder as used now: a word is spoken when its word break is decoded.
    read-back          the same decoder with read-back (streaming.make_readback), varying:
        U              unknown-word penalty (log-probability: -3 = unknown words cost little, -6 = strongly prefer
                       real words);
        T              commit threshold (with a deadline: once the word has ended, the best guess is spoken anyway,
                       so read-back is never later than today);
        a              prior weight alpha: 0 = every known word equally likely, 1 = by English word frequency;
        guard          prefix_guard: words that start longer words ("a"/"aim") wait until they've ended;
        b              beta, a bonus per word, so "nothing was said" doesn't out-compete rare real words.

WHAT'S HELD FIXED
    - Every trial is scored by a decoder that never trained on it: 5-fold cross-validation (train.py --fold i/5),
      so all 101 recordings (2026-09-28 to 2026-10-01) are test data once. Each fold's model picks its best epoch
      on its own held-out fold, which flatters all conditions equally.
    - The exact live code path: each trial is featurized as in training and pushed bin by bin through
      StreamingDecoder (keep_words=True, as the web app defaults: only the space move ends a word).

METRICS
    WER         word error rate over prompt words (a word is right only if all its sounds are right).
    real / non  WER on the prompt words that are / aren't in the read-back word list.
    sooner      for words right in both today's and read-back output: how much earlier read-back spoke them
                (median, seconds; positive = sooner).

PART 2 -- REAL WORDS FROM TOM'S OWN GESTURES ("spliced")
    Tom's recordings are random sounds, so they can't show what happens with real words. But trackpad gestures are
    separate touches (unlike speech sounds, which blend into each other), so his recorded gestures can be cut out
    and re-assembled into real-word sentences:
      1. CTC forced alignment: each held-out recording's prompt is aligned to its fold model's output, which says
         when each symbol (sound or space move) happened; every touch stroke is assigned to the nearest symbol.
      2. A bank of Tom's gestures per symbol, plus the real gaps between them (sound->sound inside a word,
         sound->space move, space move->next sound).
      3. Sentences of 2-4 real words (phonemes.make_word_prompt: common words more often, weight 1/sqrt(rank)),
         each sound played by a random recorded gesture for that sound from the SAME fold (never trained on).
    Then today vs read-back on those, plus "early": % of words spoken before their space move.
    Caveat: splicing removes any effect of context on how Tom makes a gesture, and the sentences aren't
    grammatical (prompts are words drawn independently), so only word frequency can help, not context.

HOW TO READ THE RESULT
    ~270 prompt words, of which only ~1 in 5 is a real English word. So this mainly tests that read-back does no
    harm on non-words; the real-word numbers are small and noisy. The decisive test needs recordings of real-word
    prompts: `python collect.py --words`.

HOW TO RUN
    for i in 0 1 2 3 4; do .venv/bin/python train.py --fold $i/5 --epochs 300 --out ablations/results/readback_real/fold$i.pt; done
    .venv/bin/python ablations/readback_real.py          # all folds; or e.g. "1,2,3" for some
    .venv/bin/python ablations/readback_real.py --spliced-only   # part 2 only

RESULTS
    2026-10-01 -- 101 recordings (2026-09-28 to 2026-10-01), 5-fold cross-validation. Folds 0 and 4 stuck in the
    CTC all-blank plateau at seed 0 and were retrained (seed 1: 12.6% and 26.1% phoneme error); fold phoneme
    errors 12.6 / 14.0 / 3.8 / 9.8 / 26.1%.

    PART 1 -- the recordings as they are (random-sound prompts; 298 words, 72 of them real English words):
        today                         30.5% WER
        U=-3 T=0.9 a=0 b=0 guard      30.5%   (best read-back: equal)     spoken sooner: median +0.00 s
        U=-6 T=0.9 a=0 b=2 guard      33.6%
        U=-3 T=0.9 a=1 b=0 guard      35.2%   (frequency prior: 45.8% on the real words!)
        U=-6 T=0.9 a=1 b=2 (no guard) 58.7%   (no guard: "a", "i" spoken before the word is over)
      No read-back setting speaks anything sooner: random sound strings are unpredictable by construction, and
      any of them could still continue, so nothing is certain before its space move. The prefix guard is
      essential; a word-frequency prior hurts because random prompts don't follow English word frequencies.

    PART 2 -- real-word sentences spliced from Tom's gestures (240 sentences, 721 words; fold 4 skipped: its
    recordings couldn't be aligned well enough to cut gestures from):
                                      WER    median delay  90th pct  spoken before space move
        today                         31.9%     +0.70 s    +1.12 s     5%
        U=-6 T=0.8 a=0.5 b=2 guard    24.5%     +0.56 s    +1.04 s    42%
        U=-6 T=0.9 a=0.5 b=2 guard    25.2%     +0.60 s    +1.06 s    36%   <- streaming.make_readback default
        U=-6 T=0.95 a=0.5 b=2 guard   25.7%     +0.66 s    +1.06 s    24%
        U=-6 T=0.9 a=0 b=2 guard      26.5%     +0.70 s    +1.12 s     5%   (no prior: no early words)
        U=-6 T=0.9 a=1 b=2 guard      33.6%     +0.60 s    +1.03 s    34%   (prior sharper than the prompts')
        U=-3 T=0.9 a=0.5 b=0 guard    34.3%     +0.70 s    +1.12 s     5%   (Part 1's best: wrong for words)
      With real words, read-back is more accurate (-6.7 points) AND sooner (a third of words before the space
      move). The gain needs a prior that matches what's said: a=0.5 matches how these prompts were drawn
      (1/sqrt(rank)); a=1 over-favours "to"/"a" and hurts. Settings were picked on these same sentences
      (11 configurations tried), so the best numbers are slightly optimistic.
    Bottom line: use read-back when saying real words, not for calibrating with random sounds.
"""

import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablations.common import RESULTS  # noqa: E402
from lexicon import makeable_words  # noqa: E402
from model import load_model  # noqa: E402
from phonemes import WORD_BREAK, load_inventory  # noqa: E402
from signals import BIN_S, featurize, load_trials  # noqa: E402
from streaming import StreamingDecoder, make_readback  # noqa: E402
from train import in_fold  # noqa: E402

OUT = os.path.join(RESULTS, "readback_real")
FOLDS = 5
# (name, unknown-word penalty U, threshold T, prior weight alpha, prefix guard, word bonus beta)
CONDITIONS = [("today", None, None, None, None, None)] + \
             [("read-back", u, 0.9, a, True, b) for a in (0.0, 1.0) for u in (-3.0, -6.0) for b in (0.0, 2.0, 4.0)] + \
             [("read-back", -6.0, 0.9, a, False, 2.0) for a in (0.0, 1.0)] + \
             [("read-back", -6.0, t, a, True, 2.0) for a in (0.0, 1.0) for t in (0.8, 0.95)]


def prompt_words(trial):
    words, cur = [], []
    for tok in trial["prompt"]:
        if tok == WORD_BREAK:
            if cur:
                words.append(tuple(cur))
            cur = []
        else:
            cur.append(tok)
    return words + ([tuple(cur)] if cur else [])


def run(model, vocab, features, x, unknown, threshold, inventory, alpha=1.0, guard=True, beta=3.0):
    """Push one trial through the live decoder; returns [(sounds, time spoken)]."""
    rb = make_readback(vocab, inventory, unknown, alpha, guard, beta) if unknown is not None else None
    dec = StreamingDecoder(model, vocab, keep_words=True, readback=rb, threshold=threshold or 0.9)
    said = []
    for b, row in enumerate(x):
        said += [(tuple(v), (b + 1) * BIN_S) for kind, v in dec.push(row) if kind == "word" and v]
    said += [(tuple(v), len(x) * BIN_S) for kind, v in dec.flush() if v]
    return said


def ctc_align(logp, labels):
    """Viterbi CTC alignment: for each label, the steps where it is being emitted. logp [steps, symbols]."""
    ext = [0]
    for lab in labels:
        ext += [lab, 0]
    S, T = len(ext), len(logp)
    NEGI = -1e18
    dp = np.full((T, S), NEGI)
    back = np.zeros((T, S), int)
    dp[0, 0], dp[0, 1] = logp[0, ext[0]], logp[0, ext[1]]
    for t in range(1, T):
        for s_ in range(S):
            cands = [(dp[t - 1, s_], s_)]
            if s_ >= 1:
                cands.append((dp[t - 1, s_ - 1], s_ - 1))
            if s_ >= 2 and ext[s_] != 0 and ext[s_] != ext[s_ - 2]:
                cands.append((dp[t - 1, s_ - 2], s_ - 2))
            v, b = max(cands)
            dp[t, s_], back[t, s_] = v + logp[t, ext[s_]], b
    s_ = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    steps = [[] for _ in labels]
    for t in range(T - 1, -1, -1):
        if ext[s_] != 0:
            steps[(s_ - 1) // 2].append(t)
        s_ = back[t, s_]
    return steps


def strokes(x):
    """Runs of bins with any touch (gaps of <= 2 bins merged: a finger briefly lifting)."""
    on, runs, i = x.any(1), [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if runs and i - runs[-1][1] <= 2:
                runs[-1] = (runs[-1][0], j)
            else:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def gesture_bank(model, vocab, trials, features):
    """{symbol: [bins of one gesture, ...]} and {transition: [gap in bins, ...]} from aligned recordings."""
    index = {s: i for i, s in enumerate(vocab)}
    bank, gaps = {}, {"in word": [], "to space": [], "after space": []}
    for t in trials:
        x = featurize(t, features)
        with torch.no_grad():
            logp = model(model.patch(torch.from_numpy(x)[None]))[0][0].log_softmax(-1).numpy()
        steps = ctc_align(logp, [index[s] for s in t["prompt"]])
        if any(not st for st in steps):
            continue
        centres = [(4 * int(np.mean(st)) + 7) for st in steps]          # a step's window is centred 7 bins in
        owner = {}
        for a, b in strokes(x):                                          # each stroke -> the nearest symbol
            owner.setdefault(int(np.argmin([abs((a + b) / 2 - c) for c in centres])), []).append((a, b))
        if len(owner) != len(t["prompt"]):
            continue                                                     # some symbol got no stroke: skip trial
        spans = [(min(a for a, _ in owner[k]), max(b for _, b in owner[k])) for k in range(len(t["prompt"]))]
        if any(spans[k][1] > spans[k + 1][0] for k in range(len(spans) - 1)):
            continue                                                     # out of order: alignment unreliable
        for sym, (a, b) in zip(t["prompt"], spans):
            bank.setdefault(sym, []).append(x[a:b])
        for k in range(len(spans) - 1):
            kind = "after space" if t["prompt"][k] == WORD_BREAK else \
                   "to space" if t["prompt"][k + 1] == WORD_BREAK else "in word"
            gaps[kind].append(spans[k + 1][0] - spans[k][1])
    return bank, gaps


def splice(words, prons, bank, gaps, rng):
    """A word sentence played with recorded gestures. Returns bins, and when each word's last sound and its
    space move end (in seconds)."""
    width = next(iter(bank.values()))[0].shape[1]
    parts, t, ends = [np.zeros((rng.randint(25, 60), width), np.float32)], 0, []
    t = len(parts[0])
    for w in words:
        sounds = list(prons[w])
        for i, s_ in enumerate(sounds + [WORD_BREAK]):
            g = bank[s_][rng.randrange(len(bank[s_]))]
            parts.append(g)
            t += len(g)
            if i == len(sounds) - 1:
                last_sound_end = t
            kind = "to space" if i == len(sounds) - 1 else "after space" if s_ == WORD_BREAK else "in word"
            gap = rng.choice(gaps[kind])
            parts.append(np.zeros((gap, width), np.float32))
            t += gap
        ends.append((last_sound_end * BIN_S, (t - gap) * BIN_S))
    parts.append(np.zeros((40, width), np.float32))
    return np.concatenate(parts), ends


def align_pairs(truth, said):
    from ablations.realtime_readback import align
    return align(truth, said)


def main():
    torch.set_num_threads(1)
    inventory = load_inventory()
    real = set(makeable_words(inventory)[0].values())
    trials = load_trials()
    out = {c: [] for c in CONDITIONS}                    # condition -> [(truth, said)]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folds = [int(f) for f in args[0].split(",")] if args else list(range(FOLDS))
    trials = [t for t in trials if any(in_fold(t, f, FOLDS) for f in folds)]
    if "--spliced-only" in sys.argv:
        return spliced(folds, trials, inventory)
    for fold in folds:
        # Small-data CTC training sometimes never leaves the all-blank plateau; such folds were retrained with other
        # seeds (fold<i>_s<seed>.pt) and the run that trained best is used.
        paths = sorted(glob.glob(os.path.join(OUT, f"fold{fold}.pt")) + glob.glob(os.path.join(OUT, f"fold{fold}_s*.pt")))
        path = min(paths, key=lambda p: load_model(p)[2]["held_out_per"])
        model, vocab, cfg = load_model(path)
        test = [t for t in trials if in_fold(t, fold, FOLDS)]
        print(f"fold {fold}: {len(test)} trials, phoneme error {100 * cfg['held_out_per']:.1f}% "
              f"({os.path.basename(path)})")
        for t in test:
            x = featurize(t, cfg.get("features", "basic"))
            truth = prompt_words(t)
            for c in CONDITIONS:
                out[c].append((truth, run(model, vocab, cfg.get("features", "basic"), x, c[1], c[2], inventory,
                                          c[3], c[4], c[5])))

    n_words = sum(len(truth) for truth, _ in out[CONDITIONS[0]])
    n_real = sum(w in real for truth, _ in out[CONDITIONS[0]] for w in truth)
    print(f"\n{len(trials)} trials, {n_words} prompt words, {n_real} of them real English words "
          f"(one word = {100 / n_words:.1f} points of WER)\n")
    print(f"{'condition':<32}{'WER':>7}{'real':>8}{'non':>8}{'sooner (median)':>18}{'sooner, real words':>20}")
    today = out[CONDITIONS[0]]
    for c in CONDITIONS:
        edits = 0
        err = {True: [0, 0], False: [0, 0]}
        gain, gain_real = [], []
        for (truth, said), (_, said0) in zip(out[c], today):
            e, pairs = align_pairs(truth, [w for w, _ in said])
            edits += e
            ok = dict(pairs)
            _, pairs0 = align_pairs(truth, [w for w, _ in said0])
            ok0 = dict(pairs0)
            for i, w in enumerate(truth):
                err[w in real][0] += i not in ok
                err[w in real][1] += 1
                if i in ok and i in ok0:
                    g = said0[ok0[i]][1] - said[ok[i]][1]
                    gain.append(g)
                    if w in real:
                        gain_real.append(g)
        name = c[0] if c[1] is None else f"U={c[1]:g} T={c[2]} a={c[3]:g} b={c[5]:g}{' guard' if c[4] else ''}"
        pct = lambda k: 100 * err[k][0] / max(err[k][1], 1)
        print(f"{name:<32}{100 * edits / n_words:6.1f}%{pct(True):7.1f}%{pct(False):7.1f}%"
              f"{np.median(gain) if gain else 0:+17.2f}s{np.median(gain_real) if gain_real else 0:+19.2f}s")
    spliced(folds, trials, inventory)


SPLICED = [("today", None, None, None, None, None)] + \
          [("read-back", -6.0, t, a, True, 2.0) for a in (0.0, 0.5, 1.0) for t in (0.8, 0.9, 0.95)] + \
          [("read-back", -3.0, 0.9, 0.5, True, 0.0), ("read-back", -6.0, 0.9, 0.5, False, 2.0)]


def spliced(folds, trials, inventory, per_fold=60):
    import random
    from lexicon import makeable_words
    from phonemes import make_word_prompt
    prons, _ = makeable_words(inventory)
    rng = random.Random(0)
    results = {c: [] for c in SPLICED}
    print("\nPART 2: real-word sentences spliced from Tom's own recorded gestures")
    for fold in folds:
        paths = sorted(glob.glob(os.path.join(OUT, f"fold{fold}.pt")) + glob.glob(os.path.join(OUT, f"fold{fold}_s*.pt")))
        path = min(paths, key=lambda p: load_model(p)[2]["held_out_per"])
        model, vocab, cfg = load_model(path)
        features = cfg.get("features", "basic")
        bank, gaps = gesture_bank(model, vocab, [t for t in trials if in_fold(t, fold, FOLDS)], features)
        missing = [s_ for s_ in inventory + [WORD_BREAK] if s_ not in bank]
        print(f"  fold {fold}: gestures for {len(bank)} symbols ({sum(map(len, bank.values()))} cut out)"
              f"{', missing ' + ' '.join(missing) if missing else ''}; median gaps (s): "
              + ", ".join(f"{k} {np.median(v) * BIN_S:.2f}" if v else f"{k} -" for k, v in gaps.items()))
        if WORD_BREAK not in bank or any(not v for v in gaps.values()):
            print(f"    (fold {fold} skipped: too few recordings could be aligned to cut gestures from)")
            continue
        n = tries = 0
        while n < per_fold and tries < 100 * per_fold:
            tries += 1
            _, words = make_word_prompt(inventory, rng)
            if any(s_ not in bank for w in words for s_ in prons[w]):
                continue                                                 # needs a sound this fold has no gesture for
            x, ends = splice(words, prons, bank, gaps, rng)
            truth = [tuple(prons[w]) for w in words]
            for c in SPLICED:
                results[c].append((truth, ends, run(model, vocab, features, x, c[1], c[2], inventory,
                                                    c[3], c[4], c[5])))
            n += 1
    n_words = sum(len(tr) for tr, _, _ in results[SPLICED[0]])
    print(f"\n  {len(results[SPLICED[0]])} sentences, {n_words} words. Delay = time spoken - end of the word's "
          f"last sound; the space move ends a median {np.median([b - a for _, e, _ in results[SPLICED[0]] for a, b in e]):.2f} s later.")
    print(f"  {'condition':<32}{'WER':>7}{'median delay':>14}{'90th pct':>10}{'before space move':>19}")
    for c in SPLICED:
        edits, delay, early = 0, [], []
        for truth, ends, said in results[c]:
            e, pairs = align_pairs(truth, [w for w, _ in said])
            edits += e
            for i, j in pairs:
                delay.append(said[j][1] - ends[i][0])
                early.append(said[j][1] < ends[i][1])
        name = c[0] if c[1] is None else f"U={c[1]:g} T={c[2]} a={c[3]:g} b={c[5]:g}{' guard' if c[4] else ''}"
        print(f"  {name:<32}{100 * edits / n_words:6.1f}%{np.median(delay):+13.2f}s{np.percentile(delay, 90):+9.2f}s"
              f"{100 * np.mean(early):17.0f}%")


if __name__ == "__main__":
    main()
