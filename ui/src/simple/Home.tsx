// After the intro: two tabs. "Record more data" (sentences, then retrain) and "Help me speak" (live decoding,
// nothing on screen but what the decoder heard).
import { useEffect, useMemo, useState } from "react";
import { Key } from "../components/bits";
import type { LiveFrame, Send, State, useServer } from "../server";
import { usePersistent } from "./persist";
import { Sentences, type Goal } from "./Sentences";
import { KeyList } from "./parts";
import { Training } from "./Training";

type HomeTab = "record" | "speak";
const MORE_SENTENCES = 10;

export function Home({ server }: { server: ReturnType<typeof useServer> }) {
  const { state, live, send } = server;
  const [tab, setTab] = usePersistent<HomeTab>("trackpad.home-tab", "speak");
  const mode = state?.mode ?? "idle";

  useEffect(() => { // a mode started elsewhere (e.g. the advanced view) shows on its own tab
    if (mode === "live") setTab("speak");
    if (mode === "recording") setTab("record");
  }, [mode]);

  return (
    <>
      <div className="brand-small">Trackpad</div>
      <nav className="home-tabs" aria-label="Sections">
        <button className={tab === "record" ? "on" : ""} disabled={mode !== "idle"} onClick={() => setTab("record")}>
          Record more data
        </button>
        <button className={tab === "speak" ? "on" : ""} disabled={mode !== "idle"} onClick={() => setTab("speak")}>
          Help me speak
        </button>
      </nav>
      {!state ? (
        <div className="step"><p className="step-hint">Connecting… is <code>server.py</code> running?</p></div>
      ) : tab === "record" ? (
        <RecordMore server={server} state={state} onSpeak={() => setTab("speak")} />
      ) : (
        <Speak state={state} live={live} send={send} />
      )}
    </>
  );
}

function RecordMore({ server, state, onSpeak }: {
  server: ReturnType<typeof useServer>; state: State; onSpeak: () => void;
}) {
  const [goal, setGoal] = usePersistent<Goal | null>("trackpad.more-goal", null);
  const [trainPending, setTrainPending] = usePersistent("trackpad.more-train", false);
  const [training, setTraining] = useState(state.training.running || trainPending);
  const { summary } = state;
  const model = summary.model;
  // Sentences recorded since the decoder was trained (its train + test trials are what it has seen)
  const learned = model.exists ? (model.n_train ?? 0) + (model.n_test ?? 0) : 0;
  const unlearned = model.exists && model.n_train == null ? 0 : Math.max(0, summary.total_trials - learned);

  useEffect(() => {
    if (state.training.running) setTraining(true);
  }, [state.training.running]);

  if (training) {
    return (
      <Training state={state} history={server.history} log={server.log} send={server.send} pending={trainPending}
        clearPending={() => setTrainPending(false)} doneLabel="Start speaking"
        onDone={() => {
          setTraining(false);
          onSpeak();
        }} />
    );
  }
  return (
    <Sentences state={state} send={server.send} goal={goal} setGoal={setGoal} count={MORE_SENTENCES}
      onComplete={() => setGoal(null)}
      intro={
        <>
          <h2 className="step-title">Record more sentences</h2>
          <p className="step-text">Every sentence you record helps the decoder understand your moves.</p>
          <p className="step-hint">
            {summary.total_trials} recorded so far
            {unlearned > 0 && ` · ${unlearned} the decoder hasn't learned from yet`}
          </p>
          <KeyList />
        </>
      }
      actions={unlearned > 0 && summary.total_trials >= 10 && (
        <button className="btn-big btn-ghost" onClick={() => {
          setTrainPending(true);
          setTraining(true);
        }}>Train the decoder</button>
      )} />
  );
}

function Speak({ state, live, send }: { state: State; live: LiveFrame | null; send: Send }) {
  const running = state.mode === "live";
  // Your sounds -> the English word they spell, for words decoded without read-back
  const spellings = useMemo(
    () => new Map((state.summary.words ?? []).map((w) => [w.sounds.join(" "), w.word])),
    [state.summary.words],
  );
  const words = live?.words ?? [];
  const word = (w: string[], i: number) => {
    const text = live?.texts?.[i] ?? spellings.get(w.join(" "));
    return text
      ? <span key={i} className="speak-word">{text}</span>
      : <span key={i} className="speak-word speak-sounds">{w.join(" ")}</span>;
  };

  if (!running) {
    if (!state.summary.model.exists) {
      return (
        <div className="step">
          <h2 className="step-title">No decoder yet</h2>
          <p className="step-text">Record some sentences in “Record more data”, then train the decoder.</p>
        </div>
      );
    }
    return (
      <div className="step">
        <h2 className="step-title">Make your moves. I'll say the words.</h2>
        <button className="btn-big" disabled={state.mode !== "idle"}
          onClick={() => send("live_start", { speak: true, keep_words: true, readback: false })}>Start</button>
        <p className="step-hint">The pointer locks while I listen. Press <kbd>Esc</kbd> to stop.</p>
        {words.length > 0 && <p className="speak-last">{words.slice(-12).map(word)}</p>}
      </div>
    );
  }

  const offset = Math.max(0, words.length - 12);
  const current = live?.current ?? [];
  return (
    <>
      <div className="speak">
        {words.length === 0 && current.length === 0 ? (
          <p className="speak-idle">listening…</p>
        ) : (
          <div className="speak-words">
            {words.slice(offset).map((w, k) => word(w, offset + k))}
            <span className="speak-current">{current.join(" ")}<span className="caret" /></span>
          </div>
        )}
      </div>
      <div className="keys-quiet keys-bottom">
        <Key k="Esc">stop</Key>
        <Key k="C">clear</Key>
      </div>
    </>
  );
}
