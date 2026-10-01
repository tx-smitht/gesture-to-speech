// The "neural signals" raster: one row per electrode, time scrolling right to left, decoded sounds marked on top.
import { useEffect, useRef, useState, type RefObject } from "react";
import { BIN_MS, CHANNELS, GRID_COLS, GRID_ROWS, type Marker, type Send, SignalBuffer, type State } from "../server";

const LABEL_H = 28; // strip above the raster for decoded-sound labels
const AXIS_H = 22; // strip below for the time axis
const BASELINE = 0.012; // spike view: spontaneous firing probability per bin, like a resting neuron
const GREEN = [96, 255, 136];

// Display order: top band = top strip of the trackpad, left to right within each band
const ROW_TO_CHANNEL = Array.from({ length: CHANNELS }, (_, r) => {
  const band = Math.floor(r / GRID_COLS);
  return (GRID_ROWS - 1 - band) * GRID_COLS + (r % GRID_COLS);
});

/** A fixed pseudo-random number per (bin, channel): dots stay put as they scroll instead of flickering. */
function noise(bin: number, ch: number) {
  let h = Math.imul(bin, 374761393) ^ Math.imul(ch + 1, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

export function SignalsView({ state, signal, markers, send }: {
  state: State; signal: RefObject<SignalBuffer>; markers: RefObject<Marker[]>; send: Send;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [spikes, setSpikes] = useState(true);
  const [seconds, setSeconds] = useState(10);
  const [frozen, setFrozen] = useState<number | null>(null); // bin number the view is frozen at
  const settings = useRef({ spikes, seconds, frozen });
  settings.current = { spikes, seconds, frozen };
  const live = state.mode === "live";

  useEffect(() => {
    let frame = 0;
    const off = document.createElement("canvas");
    off.height = CHANNELS;
    let image: ImageData | null = null;

    const draw = () => {
      frame = requestAnimationFrame(draw);
      const c = canvas.current;
      const buf = signal.current;
      if (!c || !buf) return;
      const { spikes, seconds, frozen } = settings.current;
      const bins = Math.round((seconds * 1000) / BIN_MS);
      const end = frozen ?? buf.latest; // newest bin shown, at the right edge
      const start = end - bins + 1;

      // 1. One pixel per (bin, channel) into a small offscreen image
      if (!image || image.width !== bins) {
        off.width = bins;
        image = new ImageData(bins, CHANNELS);
      }
      const px = image.data;
      for (let col = 0; col < bins; col++) {
        const values = buf.get(start + col);
        for (let row = 0; row < CHANNELS; row++) {
          const i = (row * bins + col) * 4;
          const ch = ROW_TO_CHANNEL[row];
          const v = values ? Math.min(1, values[ch] / 1.2) : 0;
          let a: number;
          if (spikes) {
            // Rate coding: fire with a probability set by the activity, plus a little spontaneous firing
            a = values && noise(start + col, ch) < BASELINE + 0.85 * v ? 1 : 0;
          } else {
            a = v;
          }
          px[i] = GREEN[0] * a;
          px[i + 1] = GREEN[1] * a;
          px[i + 2] = GREEN[2] * a;
          px[i + 3] = 255;
        }
      }
      off.getContext("2d")!.putImageData(image, 0, 0);

      // 2. Scale it up onto the visible canvas
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth;
      const h = c.clientHeight;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
      }
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, w, h);
      const top = LABEL_H;
      const rh = h - LABEL_H - AXIS_H;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, top, w, rh);

      // Faint lines between the 10 bands (one per horizontal strip of the trackpad)
      ctx.fillStyle = "rgba(255,255,255,0.07)";
      for (let band = 1; band < GRID_ROWS; band++) ctx.fillRect(0, top + (rh * band) / GRID_ROWS, w, 1);
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.strokeRect(0.5, top + 0.5, w - 1, rh - 1);

      // 3. Decoded sounds: a line at the bin each was decoded in, its label above
      const xOf = (n: number) => ((n - start + 0.5) / bins) * w;
      ctx.font = "600 11px ui-monospace, SF Mono, Menlo, monospace";
      ctx.textAlign = "center";
      let lastLabelX = -Infinity;
      for (const m of markers.current ?? []) {
        if (m.n < start || m.n > end) continue;
        const x = Math.round(xOf(m.n)) + 0.5;
        ctx.strokeStyle = m.kind === "break" ? "rgba(31,91,255,0.95)" : "rgba(255,255,255,0.85)";
        ctx.lineWidth = m.kind === "break" ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, top + rh);
        ctx.stroke();
        if (m.kind === "sound") {
          const y = x - lastLabelX < 22 ? 11 : 20; // stagger labels that would overlap
          ctx.fillStyle = "#fff";
          ctx.fillText(m.label, x, y);
          lastLabelX = x;
        }
      }

      // 4. Time axis: a tick every second, "now" at the right
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.font = "11px -apple-system, SF Pro Text, Inter, sans-serif";
      const perSec = 1000 / BIN_MS;
      for (let s = 0; s <= seconds; s++) {
        const x = w - (s * perSec / bins) * w;
        ctx.fillRect(Math.round(x) - 1, top + rh, 1, 4);
        ctx.textAlign = s === 0 ? "right" : s === seconds ? "left" : "center"; // keep edge labels inside
        ctx.fillText(s === 0 ? (frozen != null ? "frozen" : "now") : `−${s}s`, s === 0 ? w - 2 : Math.max(2, x), h - 5);
      }
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [signal, markers]);

  return (
    <section className="card card-dark">
      <div className="card-head">
        <span className="label">Neural signals · 160 electrodes</span>
        <span className="caption">{live ? "decoding live — sounds marked as they're decoded" : "touch the trackpad"}</span>
      </div>
      <div className="raster-wrap">
        <span className="raster-axis">electrodes</span>
        <canvas ref={canvas} className="raster" />
      </div>
      <div className="row raster-controls">
        <div className="segmented segmented-dark" role="radiogroup" aria-label="Display">
          <button className={spikes ? "on" : ""} onClick={() => setSpikes(true)}>Spikes</button>
          <button className={!spikes ? "on" : ""} onClick={() => setSpikes(false)}>Exact values</button>
        </div>
        <div className="segmented segmented-dark" role="radiogroup" aria-label="Window">
          {[5, 10, 20].map((s) => (
            <button key={s} className={seconds === s ? "on" : ""} onClick={() => setSeconds(s)}>{s} s</button>
          ))}
        </div>
        <button className="btn btn-dark" onClick={() => setFrozen(frozen == null ? signal.current?.latest ?? 0 : null)}>
          {frozen == null ? "Freeze" : "Resume"}
        </button>
        {!live && state.mode === "idle" && (
          <button className="btn btn-primary push-right" disabled={!state.summary.model.exists}
            onClick={() => send("live_start", { speak: false, keep_words: true })}>Decode live</button>
        )}
        {live && <button className="btn btn-dark push-right" onClick={() => send("stop")}>Stop</button>}
      </div>
      <p className="caption raster-note">
        {spikes
          ? "Spike view: each electrode fires at random with a rate set by its activity, plus a little spontaneous " +
            "firing — the way real neurons carry a signal (rate coding). The decoder itself reads the exact values."
          : "Exact values: the brightness of each dot is the activity the decoder actually reads."}
        {" "}Rows are grouped in 10 bands of 16, one band per horizontal strip of the trackpad, top of the pad at the top.
      </p>
    </section>
  );
}
