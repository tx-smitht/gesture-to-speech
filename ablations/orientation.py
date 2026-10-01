#!/usr/bin/env python3
"""ABLATION: does giving the network finger ORIENTATION (the angle of each contact) make decoding more accurate?

QUESTION
    Each finger touches the trackpad as an oval, and the trackpad reports which way the oval points. A fingertip
    lands almost round (about 9 x 8 mm), so its angle means little; a thumb or flat finger lands as a long, tilted
    oval (about 14 x 8 mm), so its angle is real information. A move that twists a finger changes the angle without
    moving the finger. The "orientation" feature set adds two 16x10 maps holding each contact's angle, so the
    network can use it.

HOW ANGLE IS ENCODED (signals.frame_activity)
    orient_cos = cos(2 x angle) x elongation,  orient_sin = sin(2 x angle) x elongation, spread over nearby cells.
    The angle is doubled because an oval pointing at 10 degrees and one at 190 degrees are the same shape: angles
    repeat every 180 degrees. Splitting it into cos and sin avoids a jump where 179 and 1 degrees would look far
    apart when they're nearly identical. elongation = (1 - minor/major) / 0.45 is 0 for a round contact and about 1
    for a typical thumb, so a round fingertip (whose angle is noise) contributes almost nothing and a tilted thumb
    contributes about as much as the activity map does. (Without the / 0.45 a thumb only reached ~0.18 -- small
    enough that training's 0.05 augmentation noise would have partly drowned it, biasing this test against angle.)

HYPOTHESIS
    Orientation helps if your moves differ by finger angle or involve the thumb / flat fingers; it does little if
    your moves are fingertip taps that differ only in position. Against it: the extra maps add about 287k input
    weights from small data -- the size ablation (size_features.py) found that made training less reliable.

WHAT CHANGES (the only difference between conditions)
    basic         train.py --features basic         1 map,  160 channels: activity
    orientation   train.py --features orientation   3 maps, 480 channels: activity + orient_cos + orient_sin

WHAT'S HELD FIXED
    - Data: a frozen copy of data/ taken when the run starts (ablations/results/orientation/data/).
    - Training data: ALL sentences, for both conditions. Sentences recorded before 2026-10-01 have no angle, so
      their orientation maps are zeros -- that's also what happens if you train on all your data in real use.
    - Test set: the usual fixed held-out ~15% (train.in_test_set), but SCORED ONLY ON SENTENCES THAT HAVE ANGLE
      DATA -- orientation can only help where it exists. Both conditions are scored on exactly the same sentences.
    - Training settings, epochs, and seeds 0, 1, 2 for each condition.

METRICS
    best    phoneme error rate (%) of the saved model on the scored sentences. train.py picks the epoch that did best
            on its whole test set, so this is slightly optimistic -- equally for both conditions.
    final   error rate at the last epoch (no picking, no optimism).
    left plateau   the epoch after which error stays below 90% (escaping the CTC "all blank" phase).
    errors by sound   which sounds each condition gets wrong, summed over the three seeds.

HOW TO READ THE RESULT
    With few angle sentences the test is small: check the "one symbol = X points" line. Treat a difference as real
    only if it's larger than the spread between seeds of the same condition AND shows in both "best" and "final".
    A real orientation effect should concentrate on the sounds whose moves use a thumb, a flat finger or a twist.
    Also watch for runs that get stuck (training loss stays high) -- that's the cost of more input weights.

CAVEATS
    - Needs enough recordings with angle: the script refuses to run with fewer than 40, and the result is far more
      trustworthy with 80+. Recordings made by a server started before 2026-10-01 08:25 have no angle, even if they
      were recorded later -- the server must be restarted to pick up new code.
    - The training data mixes sentences with and without angle; the network may learn to rely on orientation less
      than it would if every sentence had it. Re-run once most of your data has angle.

HOW TO RUN (from the project folder)
    uv run ablations/orientation.py run        # ~15-30 min; trains 6 models, 1 CPU core each
    uv run ablations/orientation.py analyze    # prints the tables

RESULTS
    (pending -- needs recordings with finger angles first)
"""

import argparse
import sys

from common import analyze, has_angle, run
from signals import DATA_DIR, load_trials
from train import in_test_set

NAME = "orientation"
CONDITIONS = {
    "basic": ["--features", "basic"],
    "orientation": ["--features", "orientation"],
}
MIN_ANGLE_SENTENCES = 40


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["run", "analyze"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()

    if args.step == "run":
        trials = load_trials(DATA_DIR)
        with_angle = [t for t in trials if has_angle(t)]
        scored = [t for t in with_angle if in_test_set(t)]
        print(f"{len(trials)} sentences, {len(with_angle)} with finger angles, {len(scored)} of those held out for scoring")
        if len(with_angle) < MIN_ANGLE_SENTENCES or not scored:
            sys.exit(f"Not enough yet: record at least {MIN_ANGLE_SENTENCES} sentences with angle data "
                     f"(a server started after 2026-10-01 08:25), then run this again.")
        run(NAME, CONDITIONS, seeds=args.seeds, epochs=args.epochs)
    analyze(NAME, CONDITIONS, test_filter=has_angle)


if __name__ == "__main__":
    main()
