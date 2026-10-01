// Live decoding: sounds appear as you make them; the decoder's per-step probabilities show what it "sees".
import { useEffect, useRef, useState } from "react";
import { Key } from "../components/bits";
import type { LiveFrame, State, Send } from "../server";

export function LiveView({ state, live, send }: {
  state: State; live: LiveFrame | null; send: Send;
}) {
  const [speak, setSpeak] = useState(true);
  const [keepWords, setKeepWords] = useState(true);
  const avgMs = useRef(0);
  const model = state.summary.model;
  const running = state.mode === "live";

  if (live?.infer_ms != null) avgMs.current = avgMs.current ? 0.9 * avgMs.current + 0.1 * live.infer_ms : live.infer_ms;

  const transcript = useRef<HTMLDivElement>(null);
  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight });
  }, [live?.words.length]);

  if (!running) {
    return (
      <section className="card">
        <div className="card-head"><span className="label">Live decoding</span></div>
        <h1 className="title">Speak with your hands</h1>
        <p className="lede">
          The decoder reads the trackpad every 20 ms and updates its guess every 80 ms. The pointer is locked while
          it runs — press <kbd>Esc</kbd> to stop.
        </p>
        <div className="toggles">
          <label className="toggle"><input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
            Say each word out loud</label>
          <label className="toggle"><input type="checkbox" checked={keepWords} onChange={(e) => setKeepWords(e.target.checked)} />
            Only my space move ends a word (pausing to think won't split it)</label>
        </div>
        <button className="btn btn-primary" disabled={!model.exists || state.mode !== "idle"}
          onClick={() => send("live_start", { speak, keep_words: keepWords })}>
          Start decoding
        </button>
        {!model.exists && <p className="warn-text">Train a model first.</p>}
        {live && live.words.length > 0 && (
          <div className="transcript transcript-sm">
            {live.words.map((w, i) => <span key={i} className="word">{w.join(" ")}</span>)}
          </div>
        )}
      </section>
    );
  }

  const vocab = model.vocab ?? [];
  const probs = live?.probs ?? {};
  return (
    <section className="card card-focus">
      <div className="card-head">
        <span className="label">Decoding</span>
        <span className="caption">inference {avgMs.current ? avgMs.current.toFixed(1) : "—"} ms per step</span>
      </div>
      <div className="transcript" ref={transcript}>
        {(live?.words ?? []).map((w, i) => (
          <span key={i} className="word" title={w.map((s) => state.arpabet[s]).join(" ")}>{w.join(" ")}</span>
        ))}
        <span className="word word-current">
          {(live?.current ?? []).join(" ")}
          <span className="caret" />
        </span>
      </div>
      <div className="probs" aria-label="Decoder output probabilities">
        {["_", ...vocab].map((s) => (
          <div key={s} className="prob">
            <span className="prob-track">
              <span className={`prob-bar ${s === "_" ? "blank" : ""}`} style={{ height: `${100 * (probs[s] ?? 0)}%` }} />
            </span>
            <span className="prob-sym">{s === "_" ? "blank" : s === state.word_break ? "space" : s}</span>
          </div>
        ))}
      </div>
      <div className="keys">
        <Key k="Esc">stop</Key>
        <Key k="C">clear</Key>
        <button className="btn btn-small push-right" onClick={() => send("stop")}>Stop</button>
      </div>
    </section>
  );
}
