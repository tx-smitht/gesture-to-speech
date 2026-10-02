// Training, simply: one progress bar and the current error rate. The chart, log and settings are in the advanced view.
import { useEffect, useState } from "react";
import type { EpochPoint, Send, State } from "../server";

export const EPOCHS = 200;

export function Training({ state, history, log, send, pending, clearPending, doneLabel, onDone }: {
  state: State; history: EpochPoint[]; log: string[]; send: Send;
  pending: boolean; clearPending: () => void; // pending: start a run as soon as this shows
  doneLabel: string; onDone: () => void;
}) {
  const { training, summary } = state;
  const [waiting, setWaiting] = useState(false); // asked to start, not running yet

  const start = () => {
    setWaiting(true);
    send("train_start", { epochs: EPOCHS, features: "basic" });
  };
  useEffect(() => {
    if (pending && !training.running) start();
    if (pending) clearPending();
  }, [pending]);
  useEffect(() => {
    if (training.running) setWaiting(false);
  }, [training.running]);

  const last = waiting ? undefined : history[history.length - 1]; // until it starts, the history is the last run's

  if (training.running || waiting) {
    return (
      <div className="step">
        <h2 className="step-title">Teaching the decoder your moves</h2>
        <p className="step-text">
          A small neural network is studying your sentences over and over, learning which touches mean which sound.
          This takes a minute or two.
        </p>
        <div className="meter"><span style={{ width: `${(100 * (last?.epoch ?? 0)) / Math.max(1, training.epochs)}%` }} /></div>
        <div className="stat">
          <p className="step-lead">Current error rate</p>
          <div className="stat-big">{last ? `${Math.round(last.per)}%` : "—"}</div>
          <p className="step-hint">{last ? "the share of sounds it gets wrong in sentences it hasn't studied" : "getting ready…"}</p>
        </div>
      </div>
    );
  }

  // Finished: this run's result, or (after a server restart) the saved model's own score
  const per = training.result?.per ?? (log.length === 0 && summary.model.held_out_per != null
    ? summary.model.held_out_per * 100 : null);
  if (per == null) {
    const reason = [...log].reverse().find((l) => l.trim());
    return (
      <div className="step">
        <h2 className="step-title">Training didn't finish.</h2>
        {reason && <p className="step-hint">{reason}</p>}
        <button className="btn-big" onClick={start}>Try again</button>
      </div>
    );
  }
  return (
    <div className="step">
      <h2 className="step-title">Your decoder is ready.</h2>
      <div className="stat">
        <p className="step-lead">Error rate</p>
        <div className="stat-big">{Math.round(per)}%</div>
        <p className="step-hint">the share of sounds it gets wrong in sentences it hasn't studied</p>
      </div>
      <p className="step-text">
        {per < 20 ? "That's a strong start."
          : per < 50 ? "A decent start. Recording more sentences will make it better."
          : "Still rough. It learns from more examples — record more sentences, then train again."}
      </p>
      <button className="btn-big" onClick={onDone}>{doneLabel} <span className="next-arrow">→</span></button>
    </div>
  );
}
