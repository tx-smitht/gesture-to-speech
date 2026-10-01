#!/usr/bin/env python3
"""ABLATION: does giving the network contact SIZE as its own input map make decoding more accurate?

QUESTION
    The decoder's input is a 16x10 grid of "virtual electrodes". In the original ("basic") features, each cell holds
    ONE number: blob strength x pressure x contact area. Size is therefore mixed together with pressure -- a big,
    light contact and a small, firm one can produce the same value -- and the blob's width doesn't change with
    contact size. The "size" feature set adds a second 16x10 map that holds contact area on its own, so the network
    can tell a thumb (about 14 x 8 mm) from a fingertip (about 9 x 8 mm) even where pressure is similar.

HYPOTHESIS
    Size helps, mainly on sounds whose moves involve a thumb or flat finger, and hardly at all on fingertip-only
    sounds. Against it: the extra map doubles the input layer's weights (about 287k -> 573k) while the data is small,
    which raises the risk of overfitting -- memorising the training sentences instead of learning the moves.

WHAT CHANGES (the only difference between conditions)
    basic   train.py --features basic   1 map,  160 channels: activity
    size    train.py --features size    2 maps, 320 channels: activity + contact size

WHAT'S HELD FIXED
    - Data: a frozen copy of data/ taken when the run starts (ablations/results/size_features/data/).
    - Test set: the same held-out ~15% of sentences for every run (train.in_test_set, a fixed per-sentence rule).
    - Training: same model size, learning rate, augmentation and number of epochs.
    - Seeds 0, 1, 2 for each condition: the seed changes the starting weights and the order of training batches. One
      run is noisy, so each condition gets three and we compare averages and spread.

METRICS
    best    phoneme error rate (%) of the saved model on the held-out sentences. train.py saves the epoch that
            scored best on those same sentences, so this number is slightly optimistic -- equally for both
            conditions, so the comparison is still fair.
    final   error rate at the last epoch (no picking the best epoch, so no optimism).
    left plateau   the first epoch below 90% error: how quickly the network escaped the CTC "all blank" plateau.
    errors by sound   which sounds each condition gets wrong, summed over the three seeds.

HOW TO READ THE RESULT
    The test set is small, so a single symbol is worth 1-2 points of error rate. Treat a difference as real only if
    it's larger than the spread between seeds of the same condition AND it shows up in both "best" and "final".
    Otherwise the honest answer is "no detectable difference at this amount of data" -- which is still useful: it
    says size isn't the bottleneck yet. Check the by-sound table for where any gain comes from: a real size effect
    should concentrate on the thumb / flat-finger sounds.

CAVEATS
    - Small data (81 sentences when first run). The answer may change as you record more: richer inputs usually need
      more examples before they pay off.
    - Recordings before 2026-10-01 have no finger angle, so orientation isn't tested here (see FEATURE_SETS "shape" in
      signals.py; it needs its own ablation once there are enough new recordings).

HOW TO RUN (from the project folder)
    .venv/bin/python ablations/size_features.py run        # ~15-25 min; trains 6 models, 1 CPU core each
    .venv/bin/python ablations/size_features.py analyze    # prints the tables

RESULTS
    2026-10-01 -- 81 sentences (6 sessions, recorded 2026-09-28 to 2026-10-01); test set 10 sentences / 64 symbols;
    300 epochs; seeds 0-2.

        condition  seed   best   final  left plateau
        basic      0     10.9%   14.1%    epoch 90
        basic      1     10.9%   14.1%    epoch 85
        basic      2     10.9%   15.6%    epoch 90
        size       0     29.7%   31.2%    epoch 60     <- stuck: training loss stopped at 0.64 (others reach ~0.01)
        size       1     12.5%   15.6%    epoch 90
        size       2      7.8%   14.1%    epoch 80

    Verdict: no evidence that size helps at this amount of data -- keep "basic" as the default.
    - The two size runs that trained normally landed within about 2 symbols of basic (best 7.8% / 12.5% vs 10.9%;
      final 14.1% / 15.6% vs 14.1-15.6%), inside the noise of a 64-symbol test.
    - One size run in three got stuck in a poor solution and never fit even its training sentences; no basic run
      did. Twice the input weights from the same small data made training less reliable.
    - The by-sound table showed no gain concentrated on particular sounds (size was slightly worse on most, better
      only on T: 0/9 vs 2/9 wrong).
    Revisit with more sentences, and together with orientation once enough recordings have finger angles.
"""

import argparse

from common import analyze, run

NAME = "size_features"
CONDITIONS = {
    "basic": ["--features", "basic"],
    "size": ["--features", "size"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["run", "analyze"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()
    if args.step == "run":
        run(NAME, CONDITIONS, seeds=args.seeds, epochs=args.epochs)
    analyze(NAME, CONDITIONS)


if __name__ == "__main__":
    main()
