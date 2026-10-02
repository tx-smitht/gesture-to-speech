// The simple, guided face of the app. A first visit walks through it one step at a time:
//   welcome -> "this is your trackpad" -> practise one sound -> practise space -> record 10 sentences -> train
// and then lands on two tabs: "Record more data" and "Help me speak". Everything else lives in the advanced view.
import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { GridCanvas } from "../components/ElectrodeGrid";
import { BIN_MS, CHANNELS, type Send, type State, useModeKeys, type useServer } from "../server";
import { Home } from "./Home";
import { KeyList, Next } from "./parts";
import { usePersistent } from "./persist";
import { Sentences, type Goal } from "./Sentences";
import { Training } from "./Training";
import "./simple.css";

type Stage = "welcome" | "practice-sound" | "practice-space" | "sentences" | "training" | "home";
const FIRST_SENTENCES = 10;

export function SimpleApp({ server, onAdvanced }: { server: ReturnType<typeof useServer>; onAdvanced: () => void }) {
  const { state, notices, grid, practice, clearPractice, send } = server;
  const [stage, setStage] = usePersistent<Stage>("trackpad.stage", "welcome");
  const [goal, setGoal] = usePersistent<Goal | null>("trackpad.intro-goal", null);
  const [trainPending, setTrainPending] = usePersistent("trackpad.intro-train", false);
  const [heroUp, setHeroUp] = useState(stage !== "welcome");
  const mode = state?.mode ?? "idle";
  useModeKeys(mode, send);

  // ?intro in the address bar replays the intro from the start
  useEffect(() => {
    if (new URLSearchParams(location.search).has("intro")) {
      history.replaceState(null, "", location.pathname);
      setGoal(null);
      setTrainPending(false);
      setHeroUp(false);
      setStage("welcome");
    }
  }, []);

  const go = (next: Stage) => {
    setHeroUp(true);
    setStage(next);
  };

  let body: ReactNode;
  if (stage === "welcome") {
    body = <Welcome grid={grid} onRise={() => setHeroUp(true)} onNext={() => go("practice-sound")} />;
  } else if (!state) {
    body = <div className="step"><p className="step-hint">Connecting… is <code>server.py</code> running?</p></div>;
  } else if (stage === "practice-sound") {
    body = (
      <Practice key="sound" state={state} send={send} grid={grid} practice={practice} clearPractice={clearPractice}
        lead="Make the motion for the sound" big="ah" sub="as in “father”" onNext={() => go("practice-space")} />
    );
  } else if (stage === "practice-space") {
    body = (
      <Practice key="space" state={state} send={send} grid={grid} practice={practice} clearPractice={clearPractice}
        lead="Now make your move for" big="space" sub="the break at the end of every word" onNext={() => go("sentences")} />
    );
  } else if (stage === "sentences") {
    body = (
      <Sentences state={state} send={send} goal={goal} setGoal={setGoal} count={FIRST_SENTENCES}
        onComplete={() => {
          setGoal(null);
          setTrainPending(true);
          go("training");
        }}
        intro={
          <>
            <h2 className="step-title">Now let's record for real.</h2>
            <p className="step-text">
              I'll show you a sentence. Make the movements, ending each word with your space move, and I'll record them.
            </p>
            <KeyList />
          </>
        } />
    );
  } else if (stage === "training") {
    body = (
      <Training state={state} history={server.history} log={server.log} send={send} pending={trainPending}
        clearPending={() => setTrainPending(false)} doneLabel="Start speaking" onDone={() => go("home")} />
    );
  } else {
    body = <Home server={server} />;
  }

  const onboarding = stage !== "home";
  return (
    <div className="simple">
      {(heroUp || !onboarding) && <div className="topband" />}
      {onboarding && (
        <div className={`hero ${heroUp ? "up" : ""}`}>
          <div className="hero-welcome">welcome to</div>
          <div className="hero-title">Trackpad</div>
          <div className="hero-sub">gesture to speech</div>
        </div>
      )}

      {state && state.stale_code.length > 0 && (
        <div className="stale">The app's code changed — restart <code>server.py</code> to use it.</div>
      )}

      {body}

      <button className="advanced-link" onClick={onAdvanced}>Advanced settings</button>
      {onboarding && state?.summary.model.exists && mode === "idle" && (
        <button className="skip-link" onClick={() => go("home")}>Skip intro</button>
      )}

      <div className="notices" aria-live="polite">
        {notices.map((n) => <div key={n.id} className={`notice ${n.level}`}>{n.text}</div>)}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- welcome

function Welcome({ grid, onRise, onNext }: { grid: RefObject<Float32Array>; onRise: () => void; onNext: () => void }) {
  const [phase, setPhase] = useState(0); // 0: the welcome alone, 1: the trackpad appears, 2: "Next" appears
  useEffect(() => {
    const timers = [
      window.setTimeout(onRise, 2000), // the welcome slides up to the top
      window.setTimeout(() => setPhase(1), 2900),
      window.setTimeout(() => setPhase(2), 5600),
    ];
    return () => timers.forEach(clearTimeout);
  }, []);

  if (phase === 0) return null;
  return (
    <>
      <div className="step">
        <h2 className="step-title">This is your trackpad.</h2>
        <GridCanvas grid={grid} className="pad" />
        <p className="step-hint">Touch it — the squares light up under your fingers.</p>
      </div>
      <Next show={phase >= 2} onClick={onNext} />
    </>
  );
}

// ---------------------------------------------------------------- practice: one move, then "here's what we saw"

/** Plays a recorded move back into a grid ref at its real speed. */
function usePlayback(bins: number[][] | null) {
  const frame = useRef(new Float32Array(CHANNELS));
  const bar = useRef<HTMLSpanElement>(null);
  const [run, setRun] = useState(0);

  useEffect(() => {
    if (!bins?.length) return;
    let raf = 0;
    let shownUpTo = -1;
    const t0 = performance.now();
    const step = (now: number) => {
      const i = Math.min(bins.length - 1, Math.floor((now - t0) / BIN_MS));
      // Every bin since the last frame (a slow frame mustn't skip a quick tap): take the strongest value per cell
      frame.current.fill(0);
      for (let b = shownUpTo + 1; b <= i; b++) {
        for (let c = 0; c < CHANNELS; c++) frame.current[c] = Math.max(frame.current[c], bins[b][c]);
      }
      shownUpTo = i;
      if (bar.current) bar.current.style.width = `${(100 * (i + 1)) / bins.length}%`;
      if (i < bins.length - 1) raf = requestAnimationFrame(step);
      else frame.current.fill(0);
    };
    raf = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(raf);
      frame.current.fill(0);
    };
  }, [bins, run]);

  return { frame, bar, replay: () => setRun((r) => r + 1) };
}

function Practice({ state, send, grid, practice, clearPractice, lead, big, sub, onNext }: {
  state: State; send: Send; grid: RefObject<Float32Array>; practice: number[][] | null; clearPractice: () => void;
  lead: string; big: string; sub: string; onNext: () => void;
}) {
  const [started, setStarted] = useState(false);
  const playback = usePlayback(started ? practice : null);
  const recording = state.mode === "practice";
  const saw = started && !recording && practice !== null;

  useEffect(clearPractice, []); // a playback from an earlier step isn't this one's

  const record = () => {
    clearPractice();
    setStarted(true);
    send("practice_start");
  };

  let controls: ReactNode;
  if (recording) {
    controls = (
      <p className="recording"><span className="rec-dot live" />Recording — make the motion, then press <kbd>Esc</kbd></p>
    );
  } else if (saw && practice.length > 0) {
    controls = (
      <>
        <p className="step-text">Here's what we saw.</p>
        <div className="btn-row">
          <button className="btn-big btn-ghost" onClick={playback.replay}>Play again</button>
          <button className="btn-big btn-ghost" onClick={record}><span className="rec-dot" />Record again</button>
        </div>
      </>
    );
  } else if (saw) {
    controls = (
      <>
        <p className="step-text">We didn't see any touches. Try again?</p>
        <button className="btn-big" onClick={record}><span className="rec-dot" />Record</button>
      </>
    );
  } else if (!started) {
    controls = (
      <>
        <button className="btn-big" onClick={record}><span className="rec-dot" />Record</button>
        <p className="step-hint">Press Record, make the motion on the trackpad, then press <kbd>Esc</kbd>.</p>
      </>
    );
  }

  return (
    <>
      <div className="step">
        <p className="step-lead">{lead}</p>
        <div className="step-big">{big}</div>
        <p className="step-sub">{sub}</p>
        <div className="pad-wrap">
          <GridCanvas grid={saw ? playback.frame : grid} className="pad" />
          <div className={`pad-progress ${saw && practice.length > 0 ? "" : "hidden"}`}><span ref={playback.bar} /></div>
        </div>
        {controls}
      </div>
      <Next show={saw && practice.length > 0} onClick={onNext} />
    </>
  );
}
