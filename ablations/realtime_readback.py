#!/usr/bin/env python3
"""EXPERIMENT: can words be spoken as they're decoded, instead of after the whole sentence?

QUESTION
    The published speech BCIs decode phonemes continuously, but the final words come from rescoring the whole
    sentence with a large language model once the sentence is over -- so speech output waits for the sentence.
    Our live.py speaks each word when a pause ends it, with no word knowledge at all. Is there a middle ground: use
    word and language knowledge, but speak each word the moment the decoder is confident enough of it -- sometimes
    even before the person has finished making it?

    Pipeline under test:  signal -> phoneme probabilities (model.py, every 80 ms) -> words (readback.WordBeamSearch:
    lexicon prefix tree + trigram language model, context = the words already spoken) -> commit policy -> speech.

HYPOTHESIS
    1. Most words become certain at or before their last sound: the lexicon narrows the candidates as sounds arrive
       (by "S AH M TH" only "something" is left), and the context predicts the rest ("thank" -> "you"). So a
       confidence threshold should speak most words within ~0.3 s of the finger lifting from their last sound --
       several seconds sooner than waiting for the sentence end -- for a small cost in accuracy.
    2. The cost comes from words whose ending is genuinely ambiguous until later ("go" vs "going"; "i" vs "ice").
       Waiting for the sentence end can fix those; committing early can't.
    3. A learned commit policy (Tom's idea: decide from the signal, the sound probabilities and the context whether
       to speak) beats a plain probability threshold, because it can learn when the beam is over-confident -- e.g.
       a finger is still on the pad, so more sounds may be coming.

WHAT CHANGES (the conditions are commit policies; the decoder and its weights are identical in all of them)
    pause (now)       live.py today: greedy phonemes; speak the sounds when a word break is decoded. No lexicon.
    pause + snap      same timing; the sounds are replaced by the nearest real word (fewest sound edits). No LM.
    instant phonemes  speak every sound the moment it's decoded (the "instantaneous" extreme: no words at all).
    word break        lexicon + LM beam search; speak word k once most of the beam has finished it (seen its break).
    threshold T       speak word k as soon as P(word k = w) >= T, finished or not. T swept 0.5 ... 0.995.
    agreement N       speak when the same word has been the most likely for N steps in a row (and P >= 0.5) --
                      the "local agreement" rule used to stream Whisper. N swept.
    learned T         logistic regression predicts P(correct if spoken now) from 12 features of the beam, the
                      phoneme probabilities, the raw signal and the context; speak when >= T. Trained on the
                      validation sentences only.
    sentence end      no commits; when the sentence is over, speak the single best word sequence (with the LM's
                      end-of-sentence probability). Stands in for LLM rescoring -- an LLM would be more accurate
                      AND add its own computing delay, so treat this as a latency floor for that approach.

WHAT'S HELD FIXED
    - Participants: simulated (simulate.py, all 39 sounds), so word timing is known exactly. Three levels of
      sloppiness -- tap jitter 8, 11 and 14 mm ("j8", "j11", "j14") -- chosen so the phoneme error rates (4.5%,
      14.3%, 21.5%) bracket Tom's real recordings (14-27%). (Jitter 3 and 6 mm gave 0.2% and 2.8%: too clean for
      the language knowledge to have anything to fix.)
    - Phoneme decoder: one per participant, trained with train.py on 2000 RANDOM sound prompts -- exactly how Tom
      calibrates. It has never seen an English sentence, so all English knowledge comes from the lexicon + LM.
    - Sentences: english/corpus.txt split per sentence (lexicon.split_of): "lm" sentences train the language model,
      "val" sentences (x3 performances) tune alpha/beta and train the learned policy, "test" sentences (x3
      performances) are scored. Lexicon = all 350 pronunciations (closed vocabulary: every test word is known, but
      many test word pairs/triples were never seen by the LM).
    - Beam 48, alpha/beta tuned once per participant on val (sentence-end WER) and then frozen for all policies.

METRICS
    WER          word error rate of what was SPOKEN (edits / words said). Spoken words can't be taken back, so
                 this is the error rate a listener experiences. Homophones count as correct (they sound the same).
    latency      for each correctly spoken word: time it was spoken - time the finger lifted from its LAST sound.
                 Negative = spoken before the word was finished (predicted). Reported as median and 90th percentile.
    early        % of correct words spoken before their last sound was finished.
    saved        % of all sounds that were never needed: the word had already been spoken correctly before the
                 finger started them. The person could stop there and move on (like word completion on a phone
                 keyboard). Potential only: the decoder here still expects the rest of the word.
    For instant phonemes, the error rate is per sound (PER) and latency is per sound.

HOW TO READ THE RESULT
    Plot WER against median latency: each policy setting is one point, and the best policies hug the bottom-left.
    The test set is ~285 sentences / ~1200 words, so one word is ~0.08 points of WER; differences under ~1 point
    are noise. The headline is the gap between "threshold" points and "sentence end": how much accuracy is bought
    by waiting for the whole sentence, and how many seconds that costs.

CAVEATS
    - Simulated participants make independent, unbiased slips; real people drift, hesitate and make correlated
      errors. Real recordings need prompts made of real words (needs Tom's sound inventory to cover English).
    - The language model learns from ~320 hand-written sentences (test perplexity ~40 over 350 words). A real LM
      (or an LLM queried word by word) would predict better -- which helps the early-commit policies most.
    - The simulator always pauses between words, and the phoneme decoder outputs that pause as a word break.
    - "sentence end" uses the same trigram LM, not an LLM, so its accuracy advantage here is a lower bound.

HOW TO RUN (from the project folder)
    .venv/bin/python ablations/realtime_readback.py prepare   # simulate + train 3 decoders (~25 min, 3 cores each)
    .venv/bin/python ablations/realtime_readback.py analyze   # decode with every policy (~25 min), print tables, plot

RESULTS
    2026-10-01 -- 95 test sentences x 3 performances = 1104 words; 60-epoch decoders; beam 48. Full write-up with the
    plot: docs/realtime_readback.md. Spoken WER @ median delay after the word's last sound:

        phoneme error         4.5% (j8)          14.3% (j11)         21.5% (j14)
        pause (now)           22.1% @ +0.70 s    49.0% @ +0.88 s     66.1% @ +0.88 s
        pause + snap          11.2% @ +0.70 s    31.2% @ +0.88 s     47.4% @ +0.88 s
        word break             2.0% @ +0.70 s     6.9% @ +0.86 s     13.3% @ +0.88 s
        threshold 0.9          3.4% @ +0.00 s     6.5% @ +0.12 s     12.0% @ +0.04 s
        threshold 0.95         2.3% @ +0.03 s     6.2% @ +0.15 s     11.1% @ +0.10 s
        threshold 0.98         2.1% @ +0.07 s     6.2% @ +0.21 s     10.4% @ +0.19 s
        agreement 8           10.3% @ +0.11 s    12.1% @ +0.15 s     22.8% @ +0.15 s
        learned 0.9            5.5% @ -0.05 s     7.2% @ +0.03 s     10.9% @ +0.13 s
        learned 0.95           2.4% @ +0.01 s     5.8% @ +0.13 s     10.5% @ +1.13 s
        sentence end           2.1% @ +3.82 s     5.8% @ +3.90 s     10.4% @ +3.95 s
        (instant phonemes: 8.1 / 20.0 / 30.6% per-sound error @ +0.05 s)

    - Hypothesis 1 confirmed: a 0.95-0.98 threshold matches sentence-end accuracy (within ~0.4 points) while speaking
      the median word ~0.1-0.2 s after its last sound instead of ~3.9 s; 23-43% of words come out before they're
      finished. Caveat: "sentence end" uses the same trigram LM; an LLM rescorer would gain more from waiting.
    - Hypothesis 2 confirmed: the 90th-percentile delay grows at high thresholds -- a minority of words (prefixes of
      other words: go/going, i/ice) stay uncertain until the pause or later.
    - Hypothesis 3 NOT confirmed: the learned policy is on the same curve as the threshold (slightly better at 14%
      phoneme error, slightly worse at 4.5%). Its weights are dominated by the beam's own confidence; "finger on pad
      now" got ~0 weight. The beam probability is already well calibrated.
    - LocalAgreement-style "agreement" is clearly worse: a word can stay the favourite for several 80 ms steps
      while still wrong.
"""

import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablations.common import RESULTS, ROOT  # noqa: E402
from lexicon import Lexicon, NgramLM, load_pronunciations, read_sentences, split_of  # noqa: E402
from model import BLANK, edit_distance, load_model  # noqa: E402
from phonemes import WORD_BREAK  # noqa: E402
from readback import WordBeamSearch  # noqa: E402
from signals import BIN_S, featurize  # noqa: E402

OUT = os.path.join(RESULTS, "realtime_readback")
PARTICIPANTS = {"j8": 8.0, "j11": 11.0, "j14": 14.0}
PERFORMANCES = 3          # times each val/test sentence is performed
SEED = 1                  # the simulated participant (same seed = same invented gestures)
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995]
AGREEMENT = [2, 3, 5, 8]
LEARNED = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]


# ---------------------------------------------------------------------------------------------------------------
# prepare: simulate participants, train phoneme decoders, simulate sentence performances
# ---------------------------------------------------------------------------------------------------------------

def prepare():
    import simulate
    from signals import save_trial

    pron = load_pronunciations()
    sentences = read_sentences()
    procs = []
    for name, jitter in PARTICIPANTS.items():
        d = os.path.join(OUT, f"sim_{name}")
        model_path = os.path.join(OUT, f"decoder_{name}.pt")
        if not os.path.exists(os.path.join(d, f"train_{name}.jsonl")):
            subprocess.run([sys.executable, os.path.join(ROOT, "simulate.py"), "-n", "2000", "--all-phonemes",
                            "--jitter", str(jitter), "--seed", str(SEED), "--name", f"train_{name}", "--out", d],
                           check=True)
        lang = json.load(open(os.path.join(d, f"train_{name}_language.json")))
        rng = random.Random(100 + SEED)
        for part in ("val", "test"):
            path = os.path.join(OUT, f"{part}_{name}.jsonl")
            if os.path.exists(path):
                os.remove(path)
            for words in [s for s in sentences if split_of(s) == part] * PERFORMANCES:
                prompt = [tok for w in words for tok in (*pron[w], WORD_BREAK)]
                frames, duration, timing = simulate.perform(prompt, lang, rng, jitter)
                save_trial(path, prompt, frames, duration, simulate.PAD, simulated=True, timing=timing, words=words)
        if os.path.exists(model_path):
            continue
        env = dict(os.environ, OMP_NUM_THREADS="3")
        log = open(os.path.join(OUT, f"decoder_{name}.log"), "w")
        procs.append(subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "train.py"), d, "--epochs", "60",
                                       "--out", model_path],
                                      stdout=log, stderr=subprocess.STDOUT, env=env))
    print("Training phoneme decoders (logs in", OUT, ")")
    for p in procs:
        p.wait()


# ---------------------------------------------------------------------------------------------------------------
# decoding one sentence under a policy
# ---------------------------------------------------------------------------------------------------------------

def load_sentence_trials(part, name):
    from signals import load_trials
    return load_trials(os.path.join(OUT, f"{part}_{name}.jsonl"))


def phoneme_probs(model, trials):
    """Run the phoneme decoder over each trial: log-probabilities per 80 ms step, the time each step's data ends,
    and how much touch signal arrived during each step (the raw "neural data" the learned policy may use)."""
    import torch
    out = []
    with torch.no_grad():
        for t in trials:
            x = featurize(t)
            # Only the silence after the last word can be longer than signals.MAX_QUIET_S here (simulated pauses
            # are shorter), so capping never shifts a step's time.
            logp = model(model.patch(torch.from_numpy(x)[None]))[0][0].log_softmax(-1).numpy()
            ends = (np.arange(len(logp)) * model.patch_stride + model.patch_len) * BIN_S
            act = np.array([np.abs(x[max(0, int(e / BIN_S) - model.patch_stride):int(e / BIN_S)]).sum() for e in ends])
            out.append((logp, ends, act))
    return out


def word_ends(trial, lex, pron):
    """Time the finger lifted from each word's last sound."""
    ends, i = [], 0
    for w in trial["words"]:
        i += len(pron[w])
        ends.append(trial["timing"][i - 1][1])
    return ends


def word_starts(trial, pron):
    """For each word, when the finger started each of its sounds."""
    out, i = [], 0
    for w in trial["words"]:
        out.append([t0 for t0, _ in trial["timing"][i:i + len(pron[w])]])
        i += len(pron[w])
    return out


def features(dec, post, done, logp, act, since, lm_prior):
    """What the learned policy sees for the next word to speak. 12 numbers."""
    lex = dec.lex
    order = np.argsort(post)[::-1]
    top, p1, p2 = order[0], post[order[0]], post[order[1]]
    k = len(dec.committed)
    # how much of the top word's sounds have been seen, averaged over the beam
    tot = seen = complete = 0.0
    best_total = max(np.logaddexp(*v) for v in dec.hyps.values())
    for (words, node), v in dec.hyps.items():
        m = math.exp(np.logaddexp(*v) - best_total)
        tot += m
        if len(words) > k:
            seen += m * (words[k] == top)
            complete += m * (words[k] == top)
        elif len(words) == k and node in dec.on_path[top]:
            seen += m * lex.depth[node] / len(lex.prons[top])
            complete += m * (lex.unit_at[node] == top)
    probs = np.exp(logp)
    sym = dec.sym
    clip = lambda p: math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))
    return [clip(p1), p1 - p2, done, complete / tot, seen / tot, math.log(lm_prior[top] + 1e-9),
            probs[sym[BLANK]], probs[sym[WORD_BREAK]], float(act > 0), min(since, 3.0), len(lex.prons[top]),
            float(-(post[post > 0] * np.log(post[post > 0])).sum())]


FEATURE_NAMES = ["logit P(top)", "P(top) - P(2nd)", "beam fraction past word break", "beam fraction at top's end",
                 "fraction of top's sounds seen", "log LM prior of top", "P(blank) now", "P(word break) now",
                 "finger on pad now", "seconds since last word spoken", "top's length (sounds)", "entropy"]


class Learned:
    def __init__(self, w, mu, sd):
        self.w, self.mu, self.sd = w, mu, sd

    def prob(self, f):
        z = np.dot((np.asarray(f) - self.mu) / self.sd, self.w[1:]) + self.w[0]
        return 1 / (1 + math.exp(-z))


def decode(args):
    """Decode one sentence with one policy. Returns [(unit, time spoken)], plus training rows if collecting."""
    (logp_steps, ends, act), truth, policy, param, ctx = args
    lex, lm, symbols, alpha, beta, learned = ctx["lex"], ctx["lm"], ctx["symbols"], ctx["alpha"], ctx["beta"], ctx.get("learned")
    dec = WordBeamSearch(lex, lm, symbols, alpha=alpha, beta=beta, continuous=ctx.get("continuous", False))
    spoken, rows, streak, last_top, last_time = [], [], 0, None, 0.0
    for logp, t, a in zip(logp_steps, ends, act):
        dec.step(logp)
        if policy == "sentence end":
            continue
        for _ in range(4):                       # a step can complete more than one word (rare)
            post, done = dec.posterior()
            top = int(post.argmax())
            streak = streak + 1 if top == last_top else 1
            last_top = top
            if policy == "word break":
                go = done > 0.5
            elif policy == "threshold":
                go = post[top] >= param
            elif policy == "agreement":
                go = streak >= param and post[top] >= 0.5
            else:                                # learned, or "collect" (teacher-forced, for training data)
                prior = lm.probs([*dec.committed])[:len(lex)]
                f = features(dec, post, done, logp, a, t - last_time, prior)
                if policy == "collect":
                    k = len(dec.committed)
                    if k >= len(truth):
                        break
                    rows.append((f, int(top == truth[k])))
                    go, top = done > 0.95, truth[k]
                else:
                    go = learned.prob(f) >= param
            if not go:
                break
            dec.commit(top)
            spoken.append((top, t))
            last_time, streak, last_top = t, 0, None
    final = dec.best()
    for u in final[len(spoken):]:               # sentence over: speak whatever is left (ends at Enter)
        spoken.append((u, ctx["duration"]))
    return spoken, rows


def greedy(logp_steps, ends, symbols, duration):
    """Today's live.py: best symbol per step; a word is spoken when its word break is decoded."""
    words, cur, prev, sounds = [], [], None, []
    for logp, t in zip(logp_steps, ends):
        i = int(logp.argmax())
        s = symbols[i]
        if i != prev and s != BLANK:
            if s == WORD_BREAK:
                if cur:
                    words.append((tuple(cur), t))
                cur = []
            else:
                cur.append(s)
                sounds.append((s, t))
        prev = i
    if cur:
        words.append((tuple(cur), duration))
    return words, sounds


def align(truth, said):
    """Levenshtein alignment -> (edits, [(truth index, said index)] for exact matches)."""
    n, m = len(truth), len(said)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (truth[i - 1] != said[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (truth[i - 1] != said[j - 1]):
            if truth[i - 1] == said[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return int(d[n, m]), pairs


def score(results):
    """results: [(truth items, truth end times, [(item, time spoken)], optional per-word sound start times)]
    -> WER, latencies of correct words, and the share of sounds that never needed making ("saved": the word was
    already spoken correctly before the finger started them -- the person could have stopped there)."""
    edits = total = saved = sounds = 0
    lat = []
    for truth, ends, said, *starts in results:
        e, pairs = align(truth, [s for s, _ in said])
        edits += e
        total += len(truth)
        lat += [said[j][1] - ends[i] for i, j in pairs]
        if starts:
            sounds += sum(len(st) for st in starts[0])
            saved += sum(sum(t0 > said[j][1] for t0 in starts[0][i]) for i, j in pairs)
    lat = np.array(lat)
    return {"wer": 100 * edits / total, "median": float(np.median(lat)), "p90": float(np.percentile(lat, 90)),
            "early": float(100 * (lat < 0).mean()), "saved": 100 * saved / sounds if sounds else 0.0,
            "n_words": total}


# ---------------------------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------------------------

def run_policy(data, policy, param, ctx):
    out = [decode((probs, d["truth"], policy, param, {**ctx, "duration": d["duration"]})) for probs, d in data]
    results = [(d["truth"], d["ends"], spoken, d["word_starts"]) for (spoken, _), (_, d) in zip(out, data)]
    return results, [r for _, rows in out for r in rows]


def fit_logistic(rows, l2=1e-2, steps=3000, lr=0.5):
    X = np.array([f for f, _ in rows], dtype=np.float64)
    y = np.array([lab for _, lab in rows], dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return Learned(w, mu, sd)


def corpus_lm(lex, pron, parts=("lm",)):
    """Trigram LM over pronunciations, learned from the corpus sentences in `parts`."""
    return NgramLM([lex.units([pron[w] for w in s]) for s in read_sentences() if split_of(s) in parts], range(len(lex)))


def sentence_data(model, lex, pron, part, name):
    """Phoneme probabilities + ground truth for every performance of the val or test sentences."""
    trials = load_sentence_trials(part, name)
    return [(p, {"truth": lex.units([pron[w] for w in t["words"]]), "prons": [pron[w] for w in t["words"]],
                 "sounds": [s for s in t["prompt"] if s != WORD_BREAK], "ends": word_ends(t, lex, pron),
                 "duration": t["duration"], "sound_ends": [e for _, e in t["timing"]],
                 "word_starts": word_starts(t, pron)})
            for p, t in zip(phoneme_probs(model, trials), trials)]


def evaluate(data, ctx, settings, baselines=True, verbose=True):
    """Tune alpha/beta on val, train the learned policy on val, then score every policy setting on test.
    data: {"val": [...], "test": [...]}. Returns [(policy, setting, scores)], the learned policy, alpha, beta."""
    best = None
    for alpha in (0.4, 0.7, 1.0, 1.4):
        for beta in (0.0, 1.5, 3.0):
            res, _ = run_policy(data["val"], "sentence end", None, {**ctx, "alpha": alpha, "beta": beta})
            s = score(res)
            if verbose:
                print(f"  tune alpha={alpha:<4} beta={beta:<4} val WER {s['wer']:5.1f}%")
            if best is None or s["wer"] < best[0]:
                best = (s["wer"], alpha, beta)
    ctx = {**ctx, "alpha": best[1], "beta": best[2]}
    print(f"  -> alpha={best[1]}, beta={best[2]}")

    _, rows = run_policy(data["val"], "collect", None, ctx)
    learned = fit_logistic(rows)
    ctx["learned"] = learned
    print(f"  learned policy: {len(rows)} training rows from val, {100 * np.mean([r[1] for r in rows]):.0f}% "
          f"'top word is right'. Weights (standardized inputs; + means 'speak sooner'):")
    for n_, w in sorted(zip(FEATURE_NAMES, learned.w[1:]), key=lambda x: -abs(x[1])):
        print(f"      {w:+6.2f}  {n_}")

    test, out = data["test"], []
    if baselines:
        symbols, lex = ctx["symbols"], ctx["lex"]
        g = [greedy(p[0], p[1], symbols, d["duration"]) for p, d in test]
        out.append(("pause (now)", "", score([(d["prons"], d["ends"], w) for (w, _), (_, d) in zip(g, test)])))
        snap = lambda w: min(lex.prons, key=lambda p: edit_distance(list(p), list(w)))
        out.append(("pause + snap", "", score([(d["prons"], d["ends"], [(snap(w), t) for w, t in words])
                                              for (words, _), (_, d) in zip(g, test)])))
        out.append(("instant phonemes", "per sound",
                    score([(d["sounds"], d["sound_ends"], sounds) for (_, sounds), (_, d) in zip(g, test)])))
    for policy, param in settings:
        res, _ = run_policy(test, policy, param, ctx)
        out.append((policy, "" if param is None else str(param), score(res)))
    return out, learned, ctx["alpha"], ctx["beta"]


def print_table(rows, test):
    n = rows[0][2]["n_words"]
    print(f"\n  TEST: {len(test)} sentences, {n} words (one word = {100 / n:.2f} points of WER), "
          f"mean sentence {np.mean([d['duration'] for _, d in test]):.1f} s")
    print(f"  {'policy':<18}{'setting':>10}{'WER':>8}{'median latency':>16}{'90th pct':>10}{'early':>8}"
          f"{'saved':>8}")
    for policy, param, s in rows:
        print(f"  {policy:<18}{param:>10}{s['wer']:7.1f}%{s['median']:+15.2f}s{s['p90']:+9.2f}s{s['early']:7.0f}%"
              f"{s.get('saved', 0):7.0f}%")


ALL_SETTINGS = [("word break", None)] + [("threshold", t) for t in THRESHOLDS] + \
               [("agreement", n_) for n_ in AGREEMENT] + [("learned", t) for t in LEARNED] + [("sentence end", None)]


def analyze():
    import torch
    torch.set_num_threads(4)
    pron = load_pronunciations()
    lex = Lexicon(pron)
    lm = corpus_lm(lex, pron)
    report = {}
    for name in PARTICIPANTS:
        model, symbols, cfg = load_model(os.path.join(OUT, f"decoder_{name}.pt"))
        print(f"\n=== participant {name}: phoneme decoder held-out PER {100 * cfg['held_out_per']:.1f}% "
              f"(random prompts) ===")
        data = {part: sentence_data(model, lex, pron, part, name) for part in ("val", "test")}
        ctx = {"lex": lex, "lm": lm, "symbols": symbols}
        rows, learned, alpha, beta = evaluate(data, ctx, ALL_SETTINGS)
        print_table(rows, data["test"])
        report[name] = {"alpha": alpha, "beta": beta, "per": cfg["held_out_per"], "rows": rows,
                        "learned_weights": dict(zip(FEATURE_NAMES, learned.w[1:].tolist()))}
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    plot(report)


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(report), figsize=(6 * len(report), 4.5), sharey=False)
    colors = {"threshold": "C0", "learned": "C1", "agreement": "C2", "word break": "C3", "sentence end": "k",
              "pause (now)": "C7", "pause + snap": "C5"}
    for ax, (name, r) in zip(np.atleast_1d(axes), report.items()):
        by = defaultdict(list)
        for policy, param, s in r["rows"]:
            if policy in colors:
                by[policy].append((s["median"], s["wer"], param))
        for policy, pts in by.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-" if len(pts) > 1 else "s",
                    color=colors[policy], label=policy)
        ax.axvline(0, color="0.8", lw=1)
        ax.set_xlabel("median delay after the word's last sound (s)")
        ax.set_ylabel("spoken word error rate (%)")
        ax.set_title(f"participant {name} (phoneme error {100 * r['per']:.0f}%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "latency_vs_wer.png")
    fig.savefig(path, dpi=130)
    print("plot ->", path)


if __name__ == "__main__":
    {"prepare": prepare, "analyze": analyze}[sys.argv[1] if len(sys.argv) > 1 else "analyze"]()
