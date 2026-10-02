// Small pieces shared by the simple screens.
import { useEffect, type RefObject } from "react";
import { Key } from "../components/bits";
import { GridCanvas } from "../components/ElectrodeGrid";
import type { State } from "../server";

/** "Next →", fading in slowly once it's time to move on. The right arrow key presses it too. */
export function Next({ show, onClick, label = "Next" }: { show: boolean; onClick: () => void; label?: string }) {
  useEffect(() => {
    if (!show) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") onClick();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [show, onClick]);
  return (
    <button className={`next ${show ? "show" : ""}`} onClick={onClick} tabIndex={show ? 0 : -1} aria-hidden={!show}>
      {label} <span className="next-arrow">→</span>
    </button>
  );
}

/** The keys for recording sentences, in one quiet row. */
export function KeyRow({ className = "" }: { className?: string }) {
  return (
    <div className={`keys-quiet ${className}`}>
      <Key k="Enter">Next sentence</Key>
      <Key k="R">Redo</Key>
      <Key k="Esc">Stop</Key>
    </div>
  );
}

/** Bottom right while the trackpad is in use: a small live map (proof the signal is coming in) and the lock state. */
export function SignalCorner({ state, grid, map }: { state: State; grid: RefObject<Float32Array>; map: boolean }) {
  if (state.mode === "idle") return null;
  const { guard } = state;
  const problem = !guard.available && guard.error && guard.error !== "turned off" ? guard.error : null;
  return (
    <div className="corner-signal">
      {map && <GridCanvas grid={grid} className="corner-pad" />}
      <span className={`corner-status ${guard.locked ? "on" : ""}`}>
        {guard.locked ? "Trackpad locked" : guard.available ? "Locking…"
          : guard.error === "turned off" ? "Trackpad lock off" : "Trackpad not locked"}
      </span>
      {problem && (
        <span className="corner-why">{problem[0].toUpperCase() + problem.slice(1).replace(" -- ", ": ")}</span>
      )}
    </div>
  );
}
