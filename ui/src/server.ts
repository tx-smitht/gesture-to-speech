// The WebSocket connection to server.py and the shapes of everything it sends.
import { useCallback, useEffect, useRef, useState } from "react";

export type Mode = "idle" | "recording" | "live";

export interface ModelInfo {
  exists: boolean;
  vocab?: string[];
  features?: string; // which input maps it was trained on
  held_out_per?: number | null; // fraction, 0.138 = 13.8%
  epoch?: number | null;
  n_train?: number | null;
  n_test?: number | null;
  trained_at?: string;
  missing?: string[]; // sounds in your language the model can't output yet
}

export interface Summary {
  inventory: string[];
  counts: Record<string, number>;
  total_trials: number;
  sessions: { name: string; trials: number }[];
  model: ModelInfo;
}

export interface EpochPoint {
  epoch: number;
  loss: number;
  per: number; // percent
  per_quiet: number; // percent, with 2 s of extra silence
}

export interface TrainResult {
  per: number;
  sessions: { name: string; per: number; trials: number }[];
}

export interface State {
  mode: Mode;
  summary: Summary;
  recording: { prompt: string[]; n: number; saved: number; session: string } | null;
  live: { words: string[][]; texts: string[]; current: string[]; speak: boolean; keep_words: boolean; readback: boolean } | null;
  training: { running: boolean; epochs: number; history: EpochPoint[]; result: TrainResult | null; log: string[] };
  guard: { available: boolean; locked: boolean; error: string | null };
  arpabet: Record<string, string>;
  word_break: string;
  stale_code: string[]; // code files changed since the server started: it needs a restart to use them
  feature_sets: string[];
}

export interface LiveFrame {
  words: string[][];
  texts?: string[]; // read-back: how each word is spelled ("see/sea"; unknown words end in "?")
  current: string[];
  probs?: Record<string, number>;
  infer_ms?: number;
}

export interface Notice {
  id: number;
  text: string;
  level: "info" | "warn";
}

export type Send = (cmd: string, extra?: object) => void;

export const CHANNELS = 160;
export const GRID_COLS = 16;
export const GRID_ROWS = 10;
export const BIN_MS = 20;

/** The last 30 s of the signal, one 160-channel bin every 20 ms, in a fixed-size ring that new bins overwrite. */
export class SignalBuffer {
  readonly cap = 1500;
  readonly data = new Float32Array(this.cap * CHANNELS);
  latest = -1; // number of the newest bin (from the server), -1 before any arrive

  push(n: number, values: number[]) {
    if (this.latest >= 0 && n > this.latest + 1) {
      // missed bins: blank them rather than show stale data
      for (let b = Math.max(this.latest + 1, n - this.cap); b < n; b++) this.bin(b).fill(0);
    }
    this.data.set(values, (n % this.cap) * CHANNELS);
    this.latest = n;
  }

  /** The 160 values of bin n, or null if it's not in the buffer. */
  get(n: number): Float32Array | null {
    if (n > this.latest || n <= this.latest - this.cap || n < 0) return null;
    return this.bin(n);
  }

  private bin(n: number) {
    const i = (n % this.cap) * CHANNELS;
    return this.data.subarray(i, i + CHANNELS);
  }
}

/** A decoded sound (or word break) and the bin it was decoded in. */
export interface Marker {
  n: number;
  label: string;
  kind: "sound" | "break";
}

export function useServer() {
  const [state, setState] = useState<State | null>(null);
  const [connected, setConnected] = useState(false);
  const [live, setLive] = useState<LiveFrame | null>(null);
  const [history, setHistory] = useState<EpochPoint[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [notices, setNotices] = useState<Notice[]>([]);
  // The signal arrives 50 times a second: kept in refs and drawn straight to canvases, not React state
  const grid = useRef<Float32Array>(new Float32Array(CHANNELS));
  const signal = useRef(new SignalBuffer());
  const markers = useRef<Marker[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const noticeId = useRef(0);

  useEffect(() => {
    let closed = false;
    let retry: number | undefined;

    const connect = () => {
      const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.current = socket;
      socket.onopen = () => setConnected(true);
      socket.onclose = () => {
        setConnected(false);
        if (!closed) retry = window.setTimeout(connect, 1000);
      };
      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "grid":
            grid.current.set(msg.v);
            signal.current.push(msg.n, msg.v);
            break;
          case "state":
            setState(msg as State);
            setHistory(msg.training.history);
            setLog(msg.training.log);
            if (msg.live) setLive((prev) => ({ ...prev, words: msg.live.words, texts: msg.live.texts, current: msg.live.current }));
            break;
          case "live":
            setLive((prev) => ({ ...prev, ...msg }));
            for (const [kind, value] of msg.events as [string, string | string[]][]) {
              markers.current.push(kind === "sound"
                ? { n: msg.n, label: value as string, kind: "sound" }
                : { n: msg.n, label: "", kind: "break" });
            }
            if (markers.current.length > 400) markers.current = markers.current.slice(-300);
            break;
          case "train_line":
            setLog((prev) => [...prev, msg.line].slice(-400));
            if (msg.epoch) setHistory((prev) => [...prev, msg.epoch]);
            break;
          case "notice": {
            const id = ++noticeId.current;
            setNotices((prev) => [...prev, { id, text: msg.text, level: msg.level }]);
            window.setTimeout(() => setNotices((prev) => prev.filter((n) => n.id !== id)), 3200);
            break;
          }
        }
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      ws.current?.close();
    };
  }, []);

  const send: Send = useCallback((cmd: string, extra: object = {}) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ cmd, ...extra }));
  }, []);

  return { state, connected, live, history, log, notices, grid, signal, markers, send };
}
