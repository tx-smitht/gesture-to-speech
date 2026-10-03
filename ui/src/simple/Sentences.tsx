// Recording in earnest: a sentence at a time, the prompt updating in place. No playback here -- that was only
// for the practice moves. The server does the work (the same Copy Task as the advanced Record tab).
import { useEffect, useState, type ReactNode } from "react";
import { Prompt } from "../components/bits";
import type { Send, State } from "../server";
import { KeyRow } from "./parts";

/** A batch of sentences: done once the recorded total reaches `goal` (it was `start` when the batch began).
 * Counting from the saved total lets a batch survive stopping with Esc and picking it up again. */
export interface Goal {
  start: number;
  goal: number;
}

export function Sentences({ state, send, goal, setGoal, count, intro, actions, onComplete }: {
  state: State; send: Send; goal: Goal | null; setGoal: (g: Goal | null) => void; count: number;
  intro: ReactNode; actions?: ReactNode; onComplete: () => void;
}) {
  const [waiting, setWaiting] = useState(false); // asked the server to start, not recording yet
  const rec = state.recording;
  const recording = state.mode === "recording" && rec;
  const total = state.summary.total_trials;
  const n = goal ? goal.goal - goal.start : count;
  const saved = goal ? total - goal.start : 0;
  const complete = !!goal && !recording && saved >= n;

  useEffect(() => {
    if (recording) setWaiting(false);
  }, [recording]);
  useEffect(() => {
    if (!waiting) return;
    const t = window.setTimeout(() => setWaiting(false), 3000); // the server didn't start (busy?)
    return () => clearTimeout(t);
  }, [waiting]);
  useEffect(() => {
    if (complete) onComplete();
  }, [complete]);

  const record = (k: number) => {
    setWaiting(true);
    send("record_start", { n: k, kind: "mix" });
  };

  if (recording) {
    const done = n - (rec.n - rec.saved); // sentences finished in this batch
    return (
      <>
        <div className="step step-wide">
          <div className="count">
            <div className="meter meter-thin"><span style={{ width: `${(100 * done) / n}%` }} /></div>
            <span>{Math.min(done + 1, n)} of {n}</span>
          </div>
          <Prompt tokens={rec.prompt} words={rec.words} arpabet={state.arpabet} wordBreak={state.word_break} />
        </div>
        <KeyRow className="keys-bottom" />
      </>
    );
  }
  if (waiting || complete) return <div className="step" />;

  if (!goal) {
    return (
      <div className="step">
        {intro}
        <div className="btn-row">
          <button className="btn-big" onClick={() => {
            setGoal({ start: total, goal: total + count });
            record(count);
          }}>Start</button>
          {actions}
        </div>
      </div>
    );
  }

  return (
    <div className="step">
      <h2 className="step-title">Paused</h2>
      <p className="step-text">
        {saved} of {n} sentences saved.{total < 10 && " The decoder needs at least 10 to learn from."}
      </p>
      <div className="btn-row">
        <button className="btn-big" onClick={() => record(n - saved)}>Keep going</button>
        {total >= 10 && saved > 0 && (
          <button className="btn-big btn-ghost" onClick={() => setGoal({ start: goal.start, goal: total })}>
            Done for now
          </button>
        )}
      </div>
    </div>
  );
}
