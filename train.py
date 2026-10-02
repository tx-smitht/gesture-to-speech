#!/usr/bin/env python3
"""Train the decoder on calibration trials with CTC.

    python train.py                     # your sessions in data/
    python train.py data_sim/           # simulated participant
    python train.py --epochs 100 --out models/mine.pt
    python train.py --test-session 20260929-114512   # train on the other sessions, test on this whole one
    python train.py --fold 2/5          # cross-validation: hold out the 3rd of 5 fixed slices of the trials

About 15% of trials are held out and never trained on; accuracy is only ever measured on those. Which trials are held
out is fixed per trial, so adding new sessions never reshuffles the test set -- scores stay comparable over time.
"""

import argparse
import os
import random
import time
import zlib

import numpy as np
import torch
import torch.nn as nn

from model import BLANK, Decoder, edit_distance, greedy_decode, load_model, save_model
from signals import BIN_S, FEATURE_SETS, HERE, featurize, load_trials, n_channels


def in_test_set(trial, test_frac=0.15):
    """A fixed coin flip per trial: the same trial is always test (or always train), however much data you add."""
    return zlib.crc32(f"{trial['session']}:{trial['index']}".encode()) % 1000 < test_frac * 1000


def in_fold(trial, fold, n_folds):
    """Cross-validation: which of n_folds fixed slices a trial belongs to (a different hash from in_test_set)."""
    return zlib.crc32(f"fold:{trial['session']}:{trial['index']}".encode()) % n_folds == fold


def batches(items, size, shuffle):
    items = list(items)
    if shuffle:
        random.shuffle(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def pad_batch(batch, noise=0.0, max_lead_s=0.0, max_trail_s=0.0):
    """Stack variable-length trials into one padded tensor (zeros after each trial ends).

    Augmentation (training only):
      noise       -- random static, so the network can't memorise exact values
      lead/trail  -- random extra silence before/after each trial, so the network can't use
                     "the sentence always starts ~1 s in" as a shortcut. Live decoding has no such regularity.
    """
    leads = [random.randint(0, int(max_lead_s / BIN_S)) for _ in batch]
    trails = [random.randint(0, int(max_trail_s / BIN_S)) for _ in batch]
    lengths = torch.tensor([lead + len(x) + trail for (x, _), lead, trail in zip(batch, leads, trails)])
    X = torch.zeros(len(batch), max(int(lengths.max()), 14), batch[0][0].shape[1])
    for i, ((x, _), lead) in enumerate(zip(batch, leads)):
        X[i, lead:lead + len(x)] = torch.from_numpy(x)
    if noise:
        X = X + noise * torch.randn_like(X)
    return X, lengths


def evaluate(model, data, vocab, show=0, lead_s=0.0):
    """Phoneme error rate on held-out trials: total edits needed / total sounds in the prompts.
    lead_s adds that much silence before every trial -- a stress test for live use, where nobody
    tells the decoder when a sentence is about to start."""
    model.eval()
    edits = total = 0
    examples = []
    if lead_s:
        silence = np.zeros((int(lead_s / BIN_S), data[0][0].shape[1]), dtype=np.float32)
        data = [(np.concatenate([silence, x]), target) for x, target in data]
    with torch.no_grad():
        for batch in batches(data, 32, shuffle=False):
            X, lengths = pad_batch(batch)
            best = model(model.patch(X))[0].argmax(-1)
            steps = model.steps(lengths)
            for i, (_, target) in enumerate(batch):
                decoded = greedy_decode(best[i, :steps[i]].tolist(), vocab)
                truth = [vocab[t] for t in target]
                edits += edit_distance(decoded, truth)
                total += len(truth)
                if len(examples) < show:
                    examples.append((truth, decoded))
    model.train()
    return edits / max(total, 1), examples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="*", help="session files or folders (default: data/)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--noise", type=float, default=0.05, help="augmentation: random static added while training")
    p.add_argument("--features", choices=FEATURE_SETS, default="basic",
                   help="basic = one activity map; shape = adds contact size and orientation maps")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--test-session", metavar="NAME",
                   help="hold out every trial of the session whose file name contains NAME (a 'new day' test)")
    p.add_argument("--fold", metavar="I/N", help="cross-validation: hold out slice I (0-based) of N instead")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "models", "decoder.pt"))
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    trials = load_trials(*args.data)
    if len(trials) < 10:
        raise SystemExit(f"Only {len(trials)} trials found -- record more with collect.py (or try simulate.py).")

    # Output symbols: blank + every sound/word-break that appears in the prompts
    vocab = [BLANK] + sorted({tok for t in trials for tok in t["prompt"]})
    index = {s: i for i, s in enumerate(vocab)}

    t0 = time.time()
    data = [(featurize(t, args.features), [index[tok] for tok in t["prompt"]]) for t in trials]
    print(f"{len(trials)} trials, {sum(len(x) for x, _ in data) * BIN_S / 60:.1f} min of signal, "
          f"symbols: {' '.join(vocab[1:])}  (features: {args.features}, {n_channels(args.features)} channels, "
          f"featurized in {time.time() - t0:.1f}s)")

    def held_out(t):
        if args.fold:
            i, n = map(int, args.fold.split("/"))
            return in_fold(t, i, n)
        if args.test_session:
            return args.test_session in t["session"]
        return in_test_set(t, args.test_frac)

    train, test_by_session = [], {}
    for t, item in zip(trials, data):
        if held_out(t):
            test_by_session.setdefault(t["session"], []).append(item)
        else:
            train.append(item)
    test = [item for items in test_by_session.values() for item in items]
    if not train or not test:
        raise SystemExit(f"Nothing to {'train' if not train else 'test'} on -- check --test-session.")
    n_symbols = sum(len(tgt) for _, tgt in test)
    print(f"train on {len(train)}, test on {len(test)} held-out trials ({n_symbols} symbols)\n")

    model = Decoder(n_channels(args.features), len(vocab), args.hidden, args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * -(-len(train) // 16))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    config = {"n_in": n_channels(args.features), "features": args.features, "hidden": args.hidden, "layers": args.layers, "bin_s": BIN_S,
              "patch_len": model.patch_len, "patch_stride": model.patch_stride}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_per = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in batches(train, 16, shuffle=True):
            X, lengths = pad_batch(batch, noise=args.noise, max_lead_s=3.0, max_trail_s=1.0)
            targets = torch.tensor([s for _, tgt in batch for s in tgt])
            target_lengths = torch.tensor([len(tgt) for _, tgt in batch])
            logits, _ = model(model.patch(X))
            log_probs = logits.log_softmax(-1).transpose(0, 1)  # CTCLoss wants [time, batch, symbols]
            loss = ctc(log_probs, targets, model.steps(lengths), target_lengths)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        if epoch % 5 == 0 or epoch == args.epochs:
            per, _ = evaluate(model, test, vocab)
            per_quiet, _ = evaluate(model, test, vocab, lead_s=2.0)
            score = (per + per_quiet) / 2
            mark = ""
            if score < best_per:
                best_per = score
                save_model(args.out, model, vocab, {**config, "held_out_per": per, "epoch": epoch,
                                                    "n_train": len(train), "n_test": len(test)})
                mark = "  <- best, saved"
            print(f"epoch {epoch:3d}   loss {np.mean(losses):6.3f}   held-out phoneme error rate {per:6.1%}"
                  f"   (with 2 s extra silence: {per_quiet:6.1%}){mark}")

    model, vocab, _ = load_model(args.out)
    per, examples = evaluate(model, test, vocab, show=6)
    per_quiet, _ = evaluate(model, test, vocab, lead_s=2.0)
    print(f"\nBest model: {per:.1%} phoneme error rate on held-out trials "
          f"({per_quiet:.1%} with 2 s extra silence) -> {args.out}\n")
    if len(test_by_session) > 1:
        print("Held-out error by session:")
        for session, items in sorted(test_by_session.items()):
            per_s, _ = evaluate(model, items, vocab)
            print(f"  {session}: {per_s:6.1%}   ({len(items)} trials, {sum(len(t) for _, t in items)} symbols)")
        print()
    for truth, decoded in examples:
        print(f"  said:    {' '.join(truth)}\n  decoded: {' '.join(decoded)}\n")


if __name__ == "__main__":
    main()
