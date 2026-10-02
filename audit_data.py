#!/usr/bin/env python3
"""Audit calibration recordings: find trials whose touches don't match their prompt.

    python audit_data.py                     # audits data/, prints a report, saves audit.png
    python audit_data.py data/ --plot out.png

Run it after each recording session, before training. No model is needed: it works from the touches alone.

How it works
  1. Each recording is split into strokes: from the first finger landing until every finger has lifted
     (a lift shorter than 40 ms doesn't end a stroke).
  2. Each sound takes a typical number of strokes (in Tom's language AY and EY are two quick taps, everything
     else one). These are found automatically: the counts that make the most trials add up.
  3. If a trial has exactly as many strokes as its prompt needs, stroke k belongs to prompt symbol k -- no
     guessing. Those trials give:
       - each session's HAND OFFSET: how far the whole hand sat from its usual place that day (it drifts);
       - each sound's TEMPLATE: the median stroke for that sound once the offset is removed (fingers, centre,
         movement, duration, contact size). AY has two templates (AY.1, AY.2), one per tap.
  4. Flags
       stroke count   more or fewer strokes than the prompt needs: a skipped sound, an extra or stray touch
       wrong gesture  a stroke that fits another sound's template much better than its own
       odd gesture    a stroke that fits no template well
       fingers        a stroke using a number of fingers that sound rarely uses (under 30% of its examples;
                      for 3+ finger gestures only a difference of 2 or more counts)
       made differently  a sound whose examples in one session sit consistently away from its template (the
                      gesture for that sound changed that day)
       hand drift     a session whose whole hand position is far from usual (the decoder reads absolute
                      positions, so a shifted hand looks like different gestures)
     A stroke's "fit" is its distance from a template measured in that template's typical spread
     (1 = a typical example, 3+ = unusual).

What to do with flagged trials: re-record (or delete) the ones flagged for stroke count or wrong gestures;
for hand drift, record a few more sentences on the day so the decoder sees the new position.
"""

import argparse
import collections
import os

import numpy as np

from signals import CONTACT_STATES, DATA_DIR, load_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["fingers", "x", "y", "dx", "dy", "dur", "size"]
# The smallest spread allowed per feature, so a sound that happens to be very consistent doesn't make every tiny
# difference look huge: fingers, mm, mm, mm, mm, seconds, mm
MIN_SPREAD = {"fingers": 0.5, "x": 3, "y": 3, "dx": 4, "dy": 4, "dur": 0.05, "size": 0.7}


def strokes(trial, gap=0.04):
    """[[(time, [touches]), ...], ...] -- runs of frames with at least one finger down."""
    out = []
    for t, touches in trial["frames"]:
        touches = [tc for tc in touches if tc[1] in CONTACT_STATES]
        if not touches:
            continue
        if out and t - out[-1][-1][0] <= gap:
            out[-1].append((t, touches))
        else:
            out.append([(t, touches)])
    return out


def describe(stroke):
    """A stroke -> fingers, centre (mm), movement (mm, averaged over fingers), duration (s), contact size (mm)."""
    points = np.array([[tc[2], tc[3]] for _, touches in stroke for tc in touches])
    paths = collections.defaultdict(list)
    for _, touches in stroke:
        for tc in touches:
            paths[tc[0]].append((tc[2], tc[3]))
    move = np.mean([np.subtract(p[-1], p[0]) for p in paths.values()], axis=0)
    return {"fingers": max(len(touches) for _, touches in stroke), "x": points[:, 0].mean(), "y": points[:, 1].mean(),
            "dx": move[0], "dy": move[1], "dur": stroke[-1][0] - stroke[0][0],
            "size": float(np.mean([tc[4] for _, touches in stroke for tc in touches])), "t": stroke[0][0]}


def strokes_per_sound(trials, counts):
    """Find how many strokes each sound takes (1 or 2): the choice under which the most trials add up."""
    sounds = sorted({s for t in trials for s in t["prompt"]})
    per = {s: 1 for s in sounds}
    fits = lambda: sum(len(counts[i]) == sum(per[s] for s in t["prompt"]) for i, t in enumerate(trials))
    improved = True
    while improved:
        improved = False
        for s in sounds:
            before = fits()
            per[s] = 3 - per[s]
            if fits() > before:
                improved = True
            else:
                per[s] = 3 - per[s]
    return per


def audit(trials):
    all_strokes = [[describe(st) for st in strokes(t)] for t in trials]
    per = strokes_per_sound(trials, all_strokes)
    labels = [[f"{s}.{k + 1}" if per[s] > 1 else s for s in t["prompt"] for k in range(per[s])] for t in trials]
    matched = [i for i, t in enumerate(trials) if len(all_strokes[i]) == len(labels[i])]

    # hand offset per session, then templates on offset-corrected positions
    units = [(i, lab, st) for i in matched for lab, st in zip(labels[i], all_strokes[i])]
    raw_centre = {lab: np.median([[st["x"], st["y"]] for _, l, st in units if l == lab], axis=0)
                  for lab in {l for _, l, _ in units}}
    offset = {}
    for session in sorted({t["session"] for t in trials}):
        diffs = [np.subtract([st["x"], st["y"]], raw_centre[lab]) for i, lab, st in units
                 if trials[i]["session"] == session]
        offset[session] = np.median(diffs, axis=0) if diffs else np.zeros(2)
    for i, t in enumerate(trials):
        for st in all_strokes[i]:
            st["x"] -= offset[t["session"]][0]
            st["y"] -= offset[t["session"]][1]
    template = {}
    for lab in sorted(raw_centre):
        rows = [st for _, l, st in units if l == lab]
        centre = {f: np.median([r[f] for r in rows]) for f in FEATURES}
        spread = {f: max(MIN_SPREAD[f], 1.4826 * np.median([abs(r[f] - centre[f]) for r in rows])) for f in FEATURES}
        template[lab] = (centre, spread, len(rows))

    def fit(st, lab):
        centre, spread, _ = template[lab]
        return float(np.sqrt(np.mean([min(abs(st[f] - centre[f]) / spread[f], 10) ** 2 for f in FEATURES])))

    def nearest(st):
        return min(template, key=lambda lab: fit(st, lab))

    def reads_as(st):
        best = nearest(st)
        return best if fit(st, best) < 3 else "stray?"     # fits nothing well: likely an accidental touch

    finger_counts = {lab: collections.Counter(st["fingers"] for _, l, st in units if l == lab) for lab in template}
    report = []
    for i, t in enumerate(trials):
        issues = []
        n, need = len(all_strokes[i]), len(labels[i])
        if i in matched:
            for k, (lab, st) in enumerate(zip(labels[i], all_strokes[i])):
                own, best = fit(st, lab), nearest(st)
                st["fit"] = own
                if best.split(".")[0] != lab.split(".")[0] and own > 2.5 and fit(st, best) < 0.6 * own:
                    issues.append(("wrong gesture", f"stroke {k + 1} was prompted {lab} but looks like {best}", st))
                elif own > 4:
                    issues.append(("odd gesture", f"stroke {k + 1} ({lab}) fits no template (fit {own:.1f})", st))
                usual = finger_counts[lab]
                mode = usual.most_common(1)[0][0]
                # a brief extra fingertip on a 3-5 finger gesture is harmless; on a 1-2 finger gesture it's a
                # different gesture
                if abs(st["fingers"] - mode) >= (1 if mode <= 2 else 2) and \
                        usual[st["fingers"]] < 0.3 * sum(usual.values()):
                    issues.append(("fingers", f"stroke {k + 1} ({lab}) used {st['fingers']} finger(s); {lab} is "
                                   f"usually {usual.most_common(1)[0][0]}", st))
        else:
            reading = " ".join(reads_as(st) for st in all_strokes[i])
            issues.append(("stroke count", f"{n} strokes, prompt needs {need} ({n - need:+d}); the strokes read as: "
                           f"{reading}", None))
        report.append(issues)

    # a sound made differently for a whole session: its examples there fit its template badly on average
    changed = []
    for session in sorted({t["session"] for t in trials}):
        for lab in template:
            fits = [st["fit"] for i in matched if trials[i]["session"] == session
                    for l, st in zip(labels[i], all_strokes[i]) if l == lab]
            if len(fits) >= 3 and np.median(fits) > 2.0:
                rows = [st for i in matched if trials[i]["session"] == session
                        for l, st in zip(labels[i], all_strokes[i]) if l == lab]
                changed.append((session, lab, len(fits), float(np.median(fits)),
                                {f: float(np.median([r[f] for r in rows])) for f in FEATURES}))
    return dict(changed=changed, trials=trials, strokes=all_strokes, labels=labels, per=per, matched=matched, offset=offset,
                template=template, report=report)


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials, template = result["trials"], result["template"]
    pad = trials[0]["pad"]
    sounds = sorted({lab.split(".")[0] for lab in template})
    colors = {s: plt.cm.tab20(i % 20) for i, s in enumerate(sounds)}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [2.3, 1]})

    # left: every matched stroke (offset-corrected), each sound's template, and flagged strokes
    for i in result["matched"]:
        for lab, st in zip(result["labels"][i], result["strokes"][i]):
            ax.plot(st["x"], st["y"], ".", color=colors[lab.split(".")[0]], alpha=0.25, ms=5)
    for lab, (c, _, n) in template.items():
        col = colors[lab.split(".")[0]]
        ax.plot(c["x"], c["y"], "o", color=col, ms=9 + 2 * c["fingers"], mec="black", mew=0.6)
        if np.hypot(c["dx"], c["dy"]) > 5:
            ax.annotate("", (c["x"] + c["dx"], c["y"] + c["dy"]), (c["x"], c["y"]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.annotate(f"{lab}  ({c['fingers']:.0f}f{', thumb' if c['size'] > 11 else ''})", (c["x"], c["y"]),
                    xytext=(7, 7), textcoords="offset points", fontsize=9, weight="bold")
    for issues in result["report"]:
        for kind, _, st in issues:
            if st is not None:
                ax.plot(st["x"], st["y"], "x", color="red", ms=10, mew=2)
    ax.set_xlim(0, pad[0])
    ax.set_ylim(0, pad[1])
    ax.set_aspect("equal")
    ax.set_title("Each sound's typical stroke (big dot; arrow = swipe), every example (small dots),\n"
                 "red x = flagged stroke. Positions corrected for each session's hand offset.", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # right: hand offset per session
    sessions = sorted(result["offset"])
    offs = np.array([result["offset"][s] for s in sessions])
    ax2.axhline(0, color="0.8", lw=1)
    ax2.axvline(0, color="0.8", lw=1)
    ax2.plot(offs[:, 0], offs[:, 1], "-", color="0.7")
    for (x, y), s in zip(offs, sessions):
        ax2.plot(x, y, "o", color="tab:blue")
        ax2.annotate(s.replace("session_", "").replace(".jsonl", ""), (x, y), xytext=(5, 4),
                     textcoords="offset points", fontsize=8)
    ax2.set_title("Where the whole hand sat each session\n(offset from usual, mm)", fontsize=10)
    ax2.set_xlabel("x offset (mm)")
    ax2.set_ylabel("y offset (mm)")
    lim = max(10, np.abs(offs).max() + 5)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="*", help="session files or folders (default: data/)")
    p.add_argument("--plot", default=os.path.join(HERE, "audit.png"))
    args = p.parse_args()

    trials = load_trials(*(args.data or [DATA_DIR]))
    r = audit(trials)
    print(f"{len(trials)} trials. Strokes per sound: " + ", ".join(f"{s} {n}" for s, n in r["per"].items() if n > 1)
          + " (all others 1)")
    print(f"{len(r['matched'])} trials have exactly the strokes their prompt needs.\n")

    print("Hand offset per session (mm from usual; > 10 mm is a big shift):")
    for s, (x, y) in sorted(r["offset"].items()):
        print(f"  {s:<32} x {x:+6.1f}   y {y:+6.1f}{'   <- shifted' if np.hypot(x, y) > 10 else ''}")

    print("\nSound templates (after removing hand offset):")
    for lab, (c, sp, n) in r["template"].items():
        move = f"swipe ({c['dx']:+.0f},{c['dy']:+.0f}) mm" if np.hypot(c["dx"], c["dy"]) > 5 else "tap"
        print(f"  {lab:<5} n={n:<4} {c['fingers']:.0f} finger(s){' (thumb)' if c['size'] > 11 else '':<8} "
              f"at ({c['x']:4.0f},{c['y']:4.0f}) mm  {move:<22} {c['dur']:.2f} s")

    print("\nSounds made differently in one session (median fit > 2 over its examples there):")
    for session, lab, n, med, c in r["changed"] or [("", "", 0, 0, {})]:
        if not n:
            print("  none")
            break
        tc = r["template"][lab][0]
        print(f"  {session:<32} {lab:<5} n={n:<3} median fit {med:.1f}: {c['fingers']:.0f} finger(s) at "
              f"({c['x']:.0f},{c['y']:.0f}), usual {tc['fingers']:.0f} at ({tc['x']:.0f},{tc['y']:.0f})")

    flagged = [(t, issues) for t, issues in zip(r["trials"], r["report"]) if issues]
    print(f"\nFlagged trials: {len(flagged)} of {len(trials)}")
    for t, issues in flagged:
        print(f"  {t['session']} #{t['index']}   prompt: {' '.join(t['prompt'])}"
              + (f"   ({' '.join(t['words'])})" if t.get("words") else ""))
        for kind, text, _ in issues:
            print(f"      {kind:<14} {text}")
    plot(r, args.plot)
    print(f"\nPicture of every sound's typical gesture -> {args.plot}")


if __name__ == "__main__":
    main()
