// Small shared pieces: prompts, key hints, sound balance, line chart.
import type { ReactNode } from "react";

export function Prompt({ tokens, arpabet, wordBreak, size = "lg" }: {
  tokens: string[]; arpabet: Record<string, string>; wordBreak: string; size?: "lg" | "md";
}) {
  return (
    <div className={`prompt prompt-${size}`}>
      {tokens.map((t, i) =>
        t === wordBreak ? (
          <span key={i} className="token-break" title="word break: your space move">space</span>
        ) : (
          <span key={i} className="token">
            <span className="token-sym">{t}</span>
            <span className="token-hint">{arpabet[t]}</span>
          </span>
        ),
      )}
    </div>
  );
}

export function Key({ k, children }: { k: string; children: ReactNode }) {
  return (
    <span className="key-hint">
      <kbd>{k}</kbd>
      {children}
    </span>
  );
}

/** How many examples of each sound are in the recorded data -- the network needs plenty of every one. */
export function SoundBalance({ inventory, counts, arpabet }: {
  inventory: string[]; counts: Record<string, number>; arpabet: Record<string, string>;
}) {
  const max = Math.max(1, ...inventory.map((s) => counts[s] ?? 0));
  const target = 40;
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="label">Examples per sound</span>
        <span className="caption">aim for {target}+ each</span>
      </div>
      <div className="balance">
        {inventory.map((s) => {
          const n = counts[s] ?? 0;
          return (
            <div key={s} className="balance-row" title={`${s} as in "${arpabet[s]}"`}>
              <span className="balance-sym">{s}</span>
              <span className="balance-track">
                <span className={`balance-bar ${n < target ? "low" : ""}`} style={{ width: `${(100 * n) / max}%` }} />
              </span>
              <span className="balance-n">{n}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** A small SVG line chart: x = epoch, y = percent (0-100). */
export function LineChart({ series, xMax, height = 200 }: {
  series: { points: [number, number][]; className: string; label: string }[]; xMax: number; height?: number;
}) {
  const w = 600;
  const h = height;
  const pad = { l: 36, r: 12, t: 12, b: 24 };
  const x = (v: number) => pad.l + ((w - pad.l - pad.r) * v) / Math.max(1, xMax);
  const y = (v: number) => pad.t + ((h - pad.t - pad.b) * (100 - v)) / 100;
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="chart" preserveAspectRatio="none" role="img">
        {[0, 25, 50, 75, 100].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={w - pad.r} y1={y(v)} y2={y(v)} className="chart-grid" />
            <text x={pad.l - 6} y={y(v) + 4} className="chart-tick" textAnchor="end">{v}%</text>
          </g>
        ))}
        <text x={w - pad.r} y={h - 6} className="chart-tick" textAnchor="end">epoch {xMax}</text>
        {series.map((s) =>
          s.points.length > 0 && (
            <polyline key={s.label} className={s.className}
              points={s.points.map(([a, b]) => `${x(a)},${y(b)}`).join(" ")} />
          ),
        )}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.label} className="legend-item"><span className={`legend-swatch ${s.className}`} />{s.label}</span>
        ))}
      </div>
    </div>
  );
}
