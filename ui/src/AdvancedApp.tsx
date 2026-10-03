// The full app: every control, chart and signal view. The simple guided flow (SimpleApp) links here.
import { useEffect, useState } from "react";
import { SoundBalance } from "./components/bits";
import { ElectrodeGrid } from "./components/ElectrodeGrid";
import { useModeKeys, type useServer } from "./server";
import { LiveView } from "./views/LiveView";
import { RecordView } from "./views/RecordView";
import { SignalsView } from "./views/SignalsView";
import { SoundsView } from "./views/SoundsView";
import { TrainView } from "./views/TrainView";

type Tab = "record" | "train" | "live" | "signals" | "sounds";
const TABS: { id: Tab; label: string }[] = [
  { id: "record", label: "Record" },
  { id: "train", label: "Train" },
  { id: "live", label: "Live" },
  { id: "signals", label: "Signals" },
  { id: "sounds", label: "Sounds" },
];

export function AdvancedApp({ server, onSimple, onStartOver }: {
  server: ReturnType<typeof useServer>; onSimple: () => void; onStartOver: (freshRecordings: boolean) => void;
}) {
  const { state, connected, live, history, log, notices, grid, signal, markers, send } = server;
  const [tab, setTab] = useState<Tab>("record");
  const mode = state?.mode ?? "idle";

  // Tabs you can be on while a mode runs (the pointer may be locked); entering a mode jumps to its tab if needed
  const allowed: Record<string, Tab[]> = {
    recording: ["record", "signals"], practice: ["record", "signals"], live: ["live", "signals"],
  };
  useEffect(() => {
    if (mode !== "idle") setTab((t) => (allowed[mode].includes(t) ? t : allowed[mode][0]));
  }, [mode]);

  useModeKeys(mode, send);

  const guard = state?.guard;
  const locked = mode !== "idle";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          Trackpad gesture to speech
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}
              disabled={locked && !allowed[mode].includes(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="status">
          {locked && guard?.locked && <span className="pill pill-blue">Trackpad locked · Esc to release</span>}
          {locked && guard && !guard.available && <span className="pill">Pointer not locked</span>}
          <span className={`conn ${connected ? "ok" : ""}`}>{connected ? "connected" : "connecting…"}</span>
          <button className="simple-link" onClick={onSimple}>Simple view</button>
        </div>
      </header>

      {state && (state.stale_code?.length ?? 0) > 0 && (
        <div className="banner banner-strong">
          The app's code has changed since the server started ({state.stale_code.join(", ")}). Restart it to use the
          changes: press <kbd>Ctrl</kbd>+<kbd>C</kbd> in its terminal, then run <code>uv run server.py</code> again.
        </div>
      )}

      {guard && !guard.available && mode === "idle" && (
        <div className="banner">
          {guard.error === "turned off" ? (
            <>Trackpad lock is off (replay or <code>--no-lock</code> mode).</>
          ) : (
            <>
              Trackpad lock unavailable ({guard.error}). To enable it: System Settings → Privacy &amp; Security →
              Accessibility → turn on the app you ran <code>server.py</code> from (e.g. Terminal), then restart the
              server. Everything else works without it.
            </>
          )}
        </div>
      )}

      {!state ? (
        <main className="empty">Connecting to the decoder… is <code>python server.py</code> running?</main>
      ) : (
        <main className={`layout ${tab === "signals" ? "layout-wide" : ""}`}>
          <div className="main-col">
            {tab === "record" && <RecordView state={state} send={send} />}
            {tab === "train" && <TrainView state={state} history={history} log={log} send={send} onStartOver={onStartOver} />}
            {tab === "live" && <LiveView state={state} live={live} send={send} />}
            {tab === "signals" && <SignalsView state={state} signal={signal} markers={markers} send={send} />}
            {tab === "sounds" && <SoundsView state={state} send={send} />}
          </div>
          {tab !== "signals" && (
            <aside className="side-col">
              <ElectrodeGrid grid={grid} active={connected} />
              <SoundBalance inventory={state.summary.inventory} counts={state.summary.counts} arpabet={state.arpabet} />
            </aside>
          )}
        </main>
      )}

      <div className="notices" aria-live="polite">
        {notices.map((n) => <div key={n.id} className={`notice ${n.level}`}>{n.text}</div>)}
      </div>
    </div>
  );
}
