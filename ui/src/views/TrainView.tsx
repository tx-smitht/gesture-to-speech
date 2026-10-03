// Train the decoder on everything recorded, watching the held-out error rate fall.
import { useEffect, useRef, useState } from "react";
import { LineChart } from "../components/bits";
import type { DecoderInfo, EpochPoint, State, Send } from "../server";

const pct = (v: number) => `${v.toFixed(1)}%`;

const FEATURE_INFO: Record<string, string> = {
  basic: "One map: where fingers are, how hard and how big, combined. The tested default.",
  size: "Adds contact size as its own map. No gain yet on current data (ablations/size_features.py).",
  orientation: "Adds finger angle (2 maps). Needs recordings with angle (from 2026-10-01).",
  shape: "Size and angle together (4 maps). Needs the most data.",
};

/** "decoder-20261002-101500 · 28.4% error · 60 sentences" */
function decoderLabel(d: DecoderInfo) {
  const name = d.name.replace(/\.pt$/, "");
  if (!d.exists) return `${name} · not trained yet`;
  return `${name} · ${d.held_out_per != null ? `${pct(d.held_out_per * 100)} error` : "no score"} · ${d.n_trials ?? "?"} sentences`;
}

/** "session_20261002-101500" -> a readable date */
function sinceLabel(since: string) {
  const m = since.match(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  return m ? new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]).toLocaleString() : since;
}

export function TrainView({ state, history, log, send, onStartOver }: {
  state: State; history: EpochPoint[]; log: string[]; send: Send; onStartOver: (freshRecordings: boolean) => void;
}) {
  const [epochs, setEpochs] = useState(300);
  const [features, setFeatures] = useState("basic");
  const [confirming, setConfirming] = useState(false);
  const [fresh, setFresh] = useState(true);
  const { training, summary } = state;
  const model = summary.model;
  const busy = training.running || state.mode !== "idle";
  const logBox = useRef<HTMLPreElement>(null);
  const last = history[history.length - 1];

  useEffect(() => {
    logBox.current?.scrollTo({ top: logBox.current.scrollHeight });
  }, [log]);

  return (
    <>
      <section className="card">
        <div className="card-head">
          <span className="label">Current model</span>
          {model.exists && model.trained_at && (
            <span className="caption">trained {new Date(model.trained_at).toLocaleString()}</span>
          )}
        </div>
        <div className="row">
          <label className="field">
            <span className="caption">Decoder</span>
            <select value={model.name} disabled={busy} onChange={(e) => send("decoder_select", { name: e.target.value })}>
              {(summary.decoders ?? []).map((d) => (
                <option key={d.name} value={d.name} disabled={!d.exists && d.name !== model.name}>{decoderLabel(d)}</option>
              ))}
            </select>
          </label>
          <button className="btn btn-small" disabled={busy} onClick={() => setConfirming(true)}>Start over…</button>
        </div>
        {confirming && (
          <div className="start-over">
            <p className="lede">
              Make a new, untrained decoder and go through the intro from the beginning, as if you'd never had one.
              Nothing is deleted: your decoders and recordings stay on disk, and you can switch back here any time.
            </p>
            <label className="toggle">
              <input type="checkbox" checked={fresh} onChange={(e) => setFresh(e.target.checked)} />
              Learn only from recordings made from now on (leave out the {summary.total_trials} recorded so far)
            </label>
            <div className="row">
              <button className="btn btn-primary" onClick={() => onStartOver(fresh)}>Start over</button>
              <button className="btn" onClick={() => setConfirming(false)}>Cancel</button>
            </div>
          </div>
        )}
        {model.since && <p className="caption since-note">Learns only from recordings made since {sinceLabel(model.since)}.</p>}
        {model.exists ? (
          <div className="model-summary">
            <div>
              <span className="big-number">
                {model.held_out_per != null ? pct(model.held_out_per * 100) : "—"}
              </span>
              <span className="caption">
                {model.held_out_per != null ? "phoneme error rate on held-out sentences" : "score not saved — retrain to see it"}
              </span>
            </div>
            <div className="model-meta">
              <div className="chips">
                <span className="chip chip-dark">{model.features ?? "basic"} features</span>
                {model.vocab?.map((s) => <span key={s} className="chip">{s === state.word_break ? "space" : s}</span>)}
              </div>
              {model.missing && model.missing.length > 0 && (
                <p className="warn-text">
                  Doesn't know {model.missing.join(", ")} yet — record some sentences with {model.missing.length > 1 ? "them" : "it"}, then retrain.
                </p>
              )}
            </div>
          </div>
        ) : (
          <p className="lede">No model yet. Record at least 20 sentences, then train.</p>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <span className="label">Train</span>
          <span className="caption">{summary.total_trials} sentences · {summary.sessions.length} sessions · ~15% held out for testing</span>
        </div>
        <div className="row">
          <div className="segmented" role="radiogroup" aria-label="Input features">
            {(state.feature_sets ?? ["basic"]).map((f) => (
              <button key={f} className={features === f ? "on" : ""} onClick={() => setFeatures(f)}
                disabled={training.running}>{f}</button>
            ))}
          </div>
          <span className="caption">{FEATURE_INFO[features]}</span>
        </div>
        <div className="row">
          <label className="field">
            <span className="caption">Epochs</span>
            <input type="number" min={5} max={2000} step={50} value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))} disabled={training.running} />
          </label>
          {training.running ? (
            <button className="btn" onClick={() => send("train_stop")}>Stop</button>
          ) : (
            <button className="btn btn-primary" onClick={() => send("train_start", { epochs, features })}
              disabled={summary.total_trials < 10}>Train model</button>
          )}
          {training.running && (
            <span className="caption">epoch {last?.epoch ?? 0} of {training.epochs}{last ? ` · error ${pct(last.per)}` : ""}</span>
          )}
        </div>
        {(training.running || history.length > 0) && (
          <>
            <div className="progress"><span style={{ width: `${(100 * (last?.epoch ?? 0)) / Math.max(1, training.epochs)}%` }} /></div>
            <LineChart
              xMax={training.epochs}
              series={[
                { label: "held-out error", className: "line-blue", points: history.map((p) => [p.epoch, p.per]) },
                { label: "with 2 s extra silence", className: "line-gray", points: history.map((p) => [p.epoch, p.per_quiet]) },
              ]}
            />
            <p className="caption">
              The flat start is the CTC "blank plateau": outputting nothing is the easiest early strategy. The drop
              comes when the network finds its first sound.
            </p>
          </>
        )}
        {training.result && (
          <div className="result">
            <div><span className="big-number">{pct(training.result.per)}</span><span className="caption">best held-out error</span></div>
            {training.result.sessions.length > 0 && (
              <table className="table">
                <thead><tr><th>Session</th><th>Held-out trials</th><th>Error</th></tr></thead>
                <tbody>
                  {training.result.sessions.map((s) => (
                    <tr key={s.name}><td>{s.name.replace("session_", "")}</td><td>{s.trials}</td><td>{pct(s.per)}</td></tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
        {log.length > 0 && <pre className="console" ref={logBox}>{log.join("\n")}</pre>}
      </section>
    </>
  );
}
