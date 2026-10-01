// The virtual electrode array: 16 x 10 cells, each lit by the touch activity it picks up (see signals.py).
import { useEffect, useRef, type RefObject } from "react";
import { GRID_COLS, GRID_ROWS } from "../server";

const BLUE = [31, 91, 255];
const AFTERGLOW = 0.93; // per frame: a tap lasts ~100 ms, so let it fade over ~0.5 s instead of flashing

export function ElectrodeGrid({ grid, active }: { grid: RefObject<Float32Array>; active: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let frame = 0;
    const shown = new Float32Array(GRID_COLS * GRID_ROWS);
    const draw = () => {
      const c = canvas.current;
      const values = grid.current;
      if (c && values) {
        for (let i = 0; i < shown.length; i++) shown[i] = Math.max(values[i], shown[i] * AFTERGLOW);
        const dpr = window.devicePixelRatio || 1;
        const w = c.clientWidth;
        const h = c.clientHeight;
        if (c.width !== w * dpr) {
          c.width = w * dpr;
          c.height = h * dpr;
        }
        const ctx = c.getContext("2d")!;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        const gap = 3;
        const cw = (w - gap * (GRID_COLS + 1)) / GRID_COLS;
        const ch = (h - gap * (GRID_ROWS + 1)) / GRID_ROWS;
        for (let row = 0; row < GRID_ROWS; row++) {
          for (let col = 0; col < GRID_COLS; col++) {
            const v = Math.min(1, shown[row * GRID_COLS + col] / 1.2);
            const x = gap + col * (cw + gap);
            const y = h - (gap + (row + 1) * (ch + gap)) + gap; // row 0 is the bottom edge of the trackpad
            ctx.fillStyle = "#f2f2f2";
            ctx.fillRect(x, y, cw, ch);
            if (v > 0.02) {
              ctx.fillStyle = `rgba(${BLUE.join(",")},${0.15 + 0.85 * v})`;
              ctx.fillRect(x, y, cw, ch);
            }
          }
        }
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [grid]);

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="label">Electrode array</span>
        <span className={`dot ${active ? "dot-on" : ""}`} />
      </div>
      <canvas ref={canvas} className="grid-canvas" />
      <p className="caption">160 channels · 20 ms bins · each cell ≈ 10 mm of trackpad</p>
    </div>
  );
}
