// Your language's sound inventory, picked from the 39 ARPAbet phonemes. Saved to language.json.
import { useEffect, useState } from "react";
import type { State, Send } from "../server";

const VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"];

export function SoundsView({ state, send }: { state: State; send: Send }) {
  const { inventory, counts, model } = state.summary;
  const [draft, setDraft] = useState<string[]>(inventory);
  useEffect(() => setDraft(inventory), [inventory]);

  const all = Object.keys(state.arpabet);
  const changed = draft.join() !== inventory.join();
  const toggle = (s: string) => setDraft((d) => (d.includes(s) ? d.filter((x) => x !== s) : [...d, s]));

  const group = (title: string, sounds: string[]) => (
    <div className="sound-group">
      <span className="label">{title}</span>
      <div className="sound-grid">
        {sounds.map((s) => {
          const on = draft.includes(s);
          return (
            <button key={s} className={`sound ${on ? "on" : ""}`} onClick={() => toggle(s)} aria-pressed={on}>
              <span className="sound-sym">{s}</span>
              <span className="sound-hint">{state.arpabet[s]}</span>
              {counts[s] ? <span className="sound-n">{counts[s]}</span> : null}
            </button>
          );
        })}
      </div>
    </div>
  );

  return (
    <section className="card">
      <div className="card-head">
        <span className="label">Your sounds</span>
        <span className="caption">{draft.length} selected · number = recorded examples</span>
      </div>
      <p className="lede">
        Pick the sounds you've designed a move for. Prompts only use these. The model can only output sounds it was
        trained on, so retrain after adding one{model.missing?.length ? ` (it doesn't know ${model.missing.join(", ")} yet)` : ""}.
      </p>
      {group("Vowels", all.filter((s) => VOWELS.includes(s)))}
      {group("Consonants", all.filter((s) => !VOWELS.includes(s)))}
      <div className="row">
        <button className="btn btn-primary" disabled={!changed || draft.length === 0}
          onClick={() => send("inventory_set", { phonemes: draft })}>Save sounds</button>
        {changed && <button className="btn" onClick={() => setDraft(inventory)}>Discard changes</button>}
      </div>
    </section>
  );
}
