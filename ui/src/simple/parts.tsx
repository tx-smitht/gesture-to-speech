// Small pieces shared by the simple screens.
import { useEffect } from "react";

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

export function KeyList() {
  return (
    <ul className="keylist">
      <li><kbd>Enter</kbd> when you finish a sentence</li>
      <li><kbd>R</kbd> to redo it</li>
      <li><kbd>Esc</kbd> to stop</li>
    </ul>
  );
}
