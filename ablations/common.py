"""Shared machinery for ablation studies.

An ablation changes ONE thing between conditions (e.g. the input features) and keeps everything else identical:
the same frozen copy of the data, the same held-out sentences, the same training settings, the same random seeds.
Each condition is trained with several seeds, because one training run is noisy -- the score can swing several
points between epochs of a single run -- so only differences larger than the spread between seeds mean anything.

Each experiment is its own script in this folder (see size_features.py): a note at the top explaining what's being
tested, plus a list of conditions. Outputs go to ablations/results/<experiment>/ (not committed: the snapshots and
models are built from personal recordings).
"""

import glob
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import greedy_decode, load_model  # noqa: E402
from signals import DATA_DIR, featurize, load_trials  # noqa: E402
from train import in_test_set  # noqa: E402

RESULTS = os.path.join(ROOT, "ablations", "results")
EPOCH_LINE = re.compile(r"epoch\s+(\d+)\s+loss\s+[\d.]+\s+held-out phoneme error rate\s+([\d.]+)%")


def snapshot(name, src=DATA_DIR):
    """Freeze a copy of the recordings, so recording during a long run can't change the experiment."""
    dst = os.path.join(RESULTS, name, "data")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in glob.glob(os.path.join(src, "*.jsonl")):
        shutil.copy2(f, dst)
    return dst


def run(name, conditions, seeds=3, epochs=300, threads_per_run=1):
    """Train every condition x seed in parallel. conditions: {"label": [extra train.py args]}."""
    out = os.path.join(RESULTS, name)
    data = snapshot(name)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads_per_run))  # leave CPU free for the rest of the Mac
    procs = []
    for label, extra in conditions.items():
        for seed in range(seeds):
            stem = os.path.join(out, f"{label}_s{seed}")
            cmd = [sys.executable, "-u", os.path.join(ROOT, "train.py"), data, *extra,
                   "--seed", str(seed), "--epochs", str(epochs), "--out", stem + ".pt"]
            procs.append(subprocess.Popen(cmd, stdout=open(stem + ".log", "w"), stderr=subprocess.STDOUT, env=env))
    print(f"Training {len(procs)} runs ({len(conditions)} conditions x {seeds} seeds) into {out}")
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"Warning: {sum(c != 0 for c in codes)} runs failed -- see the .log files")


def _align(truth, decoded):
    """Levenshtein alignment: for each sound that should have been decoded, was it? Also the total edits."""
    n, m = len(truth), len(decoded)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (truth[i - 1] != decoded[j - 1]))
    ok, i, j = [False] * n, n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (truth[i - 1] != decoded[j - 1]):
            ok[i - 1] = truth[i - 1] == decoded[j - 1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return ok, int(d[n, m])


def has_angle(trial):
    """True if this recording saved finger angles (recordings from 2026-10-01 on)."""
    return any(len(tc) > 7 for _, touches in trial["frames"] for tc in touches)


def analyze(name, conditions, test_filter=None):
    """Score every run on the same held-out sentences: overall error, and errors broken down by sound.
    test_filter: optionally score only some held-out sentences (e.g. has_angle)."""
    import torch

    out = os.path.join(RESULTS, name)
    test = [t for t in load_trials(os.path.join(out, "data")) if in_test_set(t) and (not test_filter or test_filter(t))]
    n_symbols = sum(len(t["prompt"]) for t in test)
    print(f"Held-out test set: {len(test)} sentences, {n_symbols} symbols "
          f"(one symbol = {100 / n_symbols:.1f} points of error rate)\n")

    rows, per_sound = [], {c: defaultdict(lambda: [0, 0]) for c in conditions}
    for log in sorted(glob.glob(os.path.join(out, "*.log"))):
        stem = log[:-4]
        label, seed = os.path.basename(stem).rsplit("_s", 1)
        if label not in conditions or not os.path.exists(stem + ".pt"):
            continue
        epochs = [(int(e), float(p)) for e, p in EPOCH_LINE.findall(open(log).read())]
        model, vocab, cfg = load_model(stem + ".pt")
        edits = 0
        with torch.no_grad():
            for t in test:
                x = torch.from_numpy(featurize(t, cfg.get("features", "basic")))[None]
                decoded = greedy_decode(model(model.patch(x))[0][0].argmax(-1).tolist(), vocab)
                ok, e = _align(t["prompt"], decoded)
                edits += e
                for sym, good in zip(t["prompt"], ok):
                    per_sound[label][sym][0] += not good
                    per_sound[label][sym][1] += 1
        # The epoch after which error stays under 90% for good. (An untrained network sometimes scores under 90% by
        # luck at the very start, before collapsing into the all-blank plateau -- that doesn't count.)
        escape = next((e for k, (e, _) in enumerate(epochs) if all(p < 90 for _, p in epochs[k:])), None)
        rows.append((label, int(seed), 100 * edits / n_symbols, epochs[-1][1] if epochs else float("nan"), escape))

    print(f"{'condition':<12}{'seed':>5}{'best':>9}{'final':>9}{'left plateau':>15}")
    for label, seed, best, final, esc in rows:
        print(f"{label:<12}{seed:>5}{best:>8.1f}%{final:>8.1f}%{('epoch ' + str(esc)) if esc else 'never':>15}")
    print()
    for label in conditions:
        best = [r[2] for r in rows if r[0] == label]
        final = [r[3] for r in rows if r[0] == label]
        if best:
            print(f"{label:<12} best: mean {np.mean(best):5.1f}% (range {min(best):.1f}-{max(best):.1f})   "
                  f"final epoch: mean {np.mean(final):5.1f}%")

    print("\nErrors by sound, summed over seeds (wrong / times it appeared in the test sentences):")
    labels = list(conditions)
    sounds = sorted({s for c in labels for s in per_sound[c]}, key=lambda s: -per_sound[labels[0]][s][1])
    print(f"{'sound':<7}" + "".join(f"{c:>12}" for c in labels))
    for s in sounds:
        print(f"{s:<7}" + "".join(f"{f'{per_sound[c][s][0]}/{per_sound[c][s][1]}':>12}" for c in labels))
