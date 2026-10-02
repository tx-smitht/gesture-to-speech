// Calibration (the "Copy Task"): see a sentence of sounds, perform it on the trackpad, press Enter.
import { useState } from "react";
import { Key, Prompt } from "../components/bits";
import type { PromptKind, State, Send } from "../server";

const KINDS: [PromptKind, string][] = [["mix", "Mix"], ["words", "Real words"], ["sounds", "Random sounds"]];

export function RecordView({ state, send }: { state: State; send: Send }) {
  const [n, setN] = useState(20);
  const [kind, setKind] = useState<PromptKind>("mix");
  const rec = state.recording;
  const { summary } = state;

  if (state.mode === "recording" && rec) {
    return (
      <section className="card card-focus">
        <div className="card-head">
          <span className="label">Sentence {rec.saved + 1} of {rec.n}</span>
          <span className="caption">{rec.session}</span>
        </div>
        <div className="progress"><span style={{ width: `${(100 * rec.saved) / rec.n}%` }} /></div>
        <Prompt tokens={rec.prompt} words={rec.words} arpabet={state.arpabet} wordBreak={state.word_break} />
        <p className="instruction">Perform the sentence on the trackpad, ending each word with your space move.</p>
        <div className="keys">
          <Key k="Enter">save</Key>
          <Key k="R">redo</Key>
          <Key k="⌫">undo last saved</Key>
          <Key k="Esc">finish</Key>
          <button className="btn btn-small push-right" onClick={() => send("stop")}>Finish</button>
        </div>
      </section>
    );
  }

  return (
    <section className="card">
      <div className="card-head">
        <span className="label">Calibration session</span>
      </div>
      <h1 className="title">Teach the decoder your language</h1>
      <p className="lede">
        You'll see sentences made of your sounds. Perform each one on the trackpad, then press <kbd>Enter</kbd>.
        You never label individual moves — the network works out which part of the signal is which sound.
      </p>
      <div className="row">
        <div className="segmented" role="radiogroup" aria-label="Sentences">
          {[10, 20, 40].map((v) => (
            <button key={v} className={n === v ? "on" : ""} onClick={() => setN(v)}>{v} sentences</button>
          ))}
        </div>
        <div className="segmented" role="radiogroup" aria-label="Prompts">
          {KINDS.map(([k, label]) => (
            <button key={k} className={kind === k ? "on" : ""} onClick={() => setKind(k)}
              disabled={k !== "sounds" && !summary.words?.length}>{label}</button>
          ))}
        </div>
        <button className="btn btn-primary" onClick={() => send("record_start", { n, kind })} disabled={state.mode !== "idle"}>
          Start recording
        </button>
      </div>
      <p className="caption balance-note">
        {kind === "sounds" ? "Random sound sentences cover every combination of your sounds. "
          : kind === "words" ? `Real English words your sounds can say (${summary.words?.length ?? 0} of them), for word read-back. `
          : "Alternates real words and random sounds: words for read-back, random sounds to cover every combination. "}
        Prompts favour the sounds with the fewest examples, so new sounds catch up:{" "}
        {[...summary.inventory]
          .sort((a, b) => (summary.counts[a] ?? 0) - (summary.counts[b] ?? 0))
          .slice(0, 3)
          .map((s) => `${s} (${summary.counts[s] ?? 0})`)
          .join(", ")}
      </p>
      <div className="stats">
        <div><span className="stat">{summary.total_trials}</span><span className="caption">sentences recorded</span></div>
        <div><span className="stat">{summary.sessions.length}</span><span className="caption">sessions</span></div>
        <div><span className="stat">{summary.inventory.length}</span><span className="caption">sounds in your language</span></div>
      </div>
      <p className="caption tip">
        Tips: do each sound the same way every time; pauses to think are fine. Record a little on different days —
        it teaches the decoder what stays the same about your moves.
      </p>
    </section>
  );
}
