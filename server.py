#!/usr/bin/env python3
"""Backend for the web app: calibration recording, training and live decoding, streamed over one WebSocket.

    python server.py              # opens http://localhost:8765
    python server.py --no-browser --port 9000

The trackpad sensor stays open the whole time, so the browser always shows the live electrode grid. Only one mode
runs at a time: idle, recording (the Copy Task) or live (decoding). In recording and live mode the pointer is
locked via bin/trackpad-guard; Esc always unlocks it.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from live import speak
from model import load_model
from phonemes import ARPABET, LANGUAGE_PATH, WORD_BREAK, count_sounds, load_inventory, make_prompt
from signals import BIN_S, CONTACT_STATES, DATA_DIR, HERE, RawTouches, bin_vector, load_trials, remove_last_trial, \
    save_trial
from streaming import StreamingDecoder
from trackpad_guard import TrackpadGuard

UI_DIST = os.path.join(HERE, "ui", "dist")
EPOCH_LINE = re.compile(r"epoch\s+(\d+)\s+loss\s+([\d.]+)\s+held-out phoneme error rate\s+([\d.]+)%"
                        r"\s+\(with 2 s extra silence:\s+([\d.]+)%\)")
BEST_LINE = re.compile(r"Best model: ([\d.]+)% phoneme error rate")
SESSION_LINE = re.compile(r"(session_[\w-]+)\.jsonl:\s+([\d.]+)%\s+\((\d+) trials")


class ReplayTouches:
    """Stands in for RawTouches: replays recorded trials in real time, looping, with a pause between them.
    For demos and for testing the whole pipeline without touching the trackpad (python server.py --replay FILE)."""

    def __init__(self, path, gap_s=2.0):
        self.trials = load_trials(path)
        self.pad = self.trials[0]["pad"]
        self.schedule, t = [], 0.5
        for tr in self.trials:
            self.schedule += [(t + ft, touches) for ft, touches in tr["frames"]]
            t += tr["duration"] + gap_s
        self.length, self.t0, self.i = t, time.monotonic(), 0

    def take(self):
        now = time.monotonic() - self.t0
        if now > self.length:  # loop
            self.t0, self.i, now = time.monotonic(), 0, 0.0
        out = []
        while self.i < len(self.schedule) and self.schedule[self.i][0] <= now:
            out.append((self.t0 + self.schedule[self.i][0], self.schedule[self.i][1]))
            self.i += 1
        return out

    def __exit__(self, *exc):
        pass


class App:
    def __init__(self, data_dir, model_path, replay=None, lock=True):
        self.data_dir, self.model_path = data_dir, model_path
        self.clients = set()
        self.loop = None
        self.rt = ReplayTouches(replay) if replay else RawTouches().__enter__()
        # Replays don't read the trackpad, so they never lock it
        self.guard = TrackpadGuard(on_event=self._guard_event, enabled=lock and not replay)
        self.mode = "idle"
        self.tick = 0          # number of the current 20 ms bin, so the browser can line events up with the signal
        self.rec = None        # recording state
        self.live = None       # live-decoding state
        self.train = {"running": False, "epochs": 0, "history": [], "result": None, "log": []}
        self._train_proc = None
        self._model_cache = (None, None)
        self.summary = self._summarize()

    # ------------------------------------------------------------------ state sent to the browser

    def _summarize(self):
        trials = load_trials(self.data_dir) if os.path.isdir(self.data_dir) else []
        counts, sessions = {}, {}
        for t in trials:
            for tok in t["prompt"]:
                counts[tok] = counts.get(tok, 0) + 1
            sessions[t["session"]] = sessions.get(t["session"], 0) + 1
        inventory = load_inventory()
        model = {"exists": os.path.exists(self.model_path)}
        if model["exists"]:
            ck = torch.load(self.model_path, weights_only=False)
            c = ck["config"]
            vocab = ck["vocab"][1:]
            model.update(vocab=vocab, held_out_per=c.get("held_out_per"), epoch=c.get("epoch"),
                         n_train=c.get("n_train"), n_test=c.get("n_test"),
                         trained_at=datetime.fromtimestamp(os.path.getmtime(self.model_path)).isoformat(),
                         missing=[s for s in inventory if s not in vocab])
        return {"inventory": inventory, "counts": counts, "total_trials": len(trials),
                "sessions": [{"name": k, "trials": v} for k, v in sorted(sessions.items())], "model": model}

    def state(self):
        rec = None
        if self.rec:
            rec = {k: self.rec[k] for k in ("prompt", "n", "session")} | {"saved": len(self.rec["saved"])}
        live = None
        if self.live:
            live = {k: self.live[k] for k in ("words", "current", "speak", "keep_words")}
        guard = {"available": self.guard.available, "locked": self.guard.locked, "error": self.guard.error}
        return {"type": "state", "mode": self.mode, "summary": self.summary, "recording": rec, "live": live,
                "training": self.train, "guard": guard, "arpabet": ARPABET, "word_break": WORD_BREAK}

    async def send(self, msg, ws=None):
        text = json.dumps(msg)
        for client in [ws] if ws else list(self.clients):
            try:
                await client.send_text(text)
            except Exception:
                self.clients.discard(client)

    async def push_state(self):
        await self.send(self.state())

    async def notice(self, text, level="info"):
        await self.send({"type": "notice", "text": text, "level": level})

    # ------------------------------------------------------------------ trackpad lock

    def _guard_event(self, event):
        # Called from the guard's reader thread
        if self.loop and event in ("escape", "watchdog", "blocked", "allowed"):
            asyncio.run_coroutine_threadsafe(self._on_guard(event), self.loop)

    async def _on_guard(self, event):
        if event in ("escape", "watchdog") and self.mode != "idle":
            await self.stop()
            await self.notice("Trackpad unlocked (Esc)" if event == "escape" else "Trackpad unlocked (watchdog)")
        else:
            await self.push_state()

    # ------------------------------------------------------------------ the 20 ms loop

    async def tick_loop(self):
        next_t = time.monotonic()
        while True:
            next_t += BIN_S
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))
            frames = self.rt.take()
            x = bin_vector(frames, self.rt.pad)
            self.tick += 1
            if self.rec is not None:
                self.rec["frames"] += frames
            if self.live is not None:
                await self._live_step(x)
            if self.clients:  # every bin, numbered: the signal raster draws one column per bin
                await self.send({"type": "grid", "n": self.tick, "v": [round(float(v), 2) for v in x]})

    # ------------------------------------------------------------------ recording (the Copy Task)

    async def record_start(self, n):
        if self.mode != "idle":
            return
        self.rec = {"session": f"session_{datetime.now():%Y%m%d-%H%M%S}", "n": max(1, int(n)), "saved": [],
                    "inventory": load_inventory(), "frames": [], "t_start": time.monotonic()}
        self.rec["prompt"] = self._next_prompt()
        self.mode = "recording"
        self.guard.block()
        await self.push_state()

    def _next_prompt(self):
        """Favour sounds with the fewest examples, counting this session's saved trials too."""
        counts = dict(self.summary["counts"])
        for tok, n in count_sounds(self.rec["saved"]).items():
            counts[tok] = counts.get(tok, 0) + n
        return make_prompt(self.rec["inventory"], counts=counts)

    def _restart_trial(self):
        self.rec["frames"], self.rec["t_start"] = [], time.monotonic()

    async def record_accept(self):
        rec = self.rec
        if not rec:
            return
        t_end = time.monotonic()
        frames = [(t - rec["t_start"], touches) for t, touches in rec["frames"] if t >= rec["t_start"]]
        if not any(tc[1] in CONTACT_STATES for _, touches in frames for tc in touches):
            self._restart_trial()
            await self.notice("No touches recorded -- try again", "warn")
            return
        path = os.path.join(self.data_dir, rec["session"] + ".jsonl")
        save_trial(path, rec["prompt"], frames, t_end - rec["t_start"], self.rt.pad)
        rec["saved"].append(rec["prompt"])
        if len(rec["saved"]) >= rec["n"]:
            await self.stop()
            await self.notice(f"Session complete: {rec['n']} sentences saved")
            return
        rec["prompt"] = self._next_prompt()
        self._restart_trial()
        await self.push_state()

    async def record_redo(self):
        if self.rec:
            self._restart_trial()
            await self.notice("Redo -- same prompt")

    async def record_undo(self):
        rec = self.rec
        if not rec:
            return
        if not rec["saved"]:
            await self.notice("Nothing saved yet to undo", "warn")
            return
        remove_last_trial(os.path.join(self.data_dir, rec["session"] + ".jsonl"))
        rec["prompt"] = rec["saved"].pop()
        self._restart_trial()
        await self.notice(f"Removed sentence {len(rec['saved']) + 1} -- do it again")
        await self.push_state()

    # ------------------------------------------------------------------ live decoding

    def _model(self):
        mtime = os.path.getmtime(self.model_path)
        if self._model_cache[0] != mtime:
            self._model_cache = (mtime, load_model(self.model_path))
        return self._model_cache[1]

    async def live_start(self, speak_words, keep_words):
        if self.mode != "idle":
            return
        if not os.path.exists(self.model_path):
            await self.notice("No model yet -- train one first", "warn")
            return
        model, vocab, _ = self._model()
        with torch.no_grad():  # warm-up: PyTorch's first call is slow; do it now, not on your first touch
            model(torch.zeros(1, 1, model.inp[0].in_features))
        self.live = {"decoder": StreamingDecoder(model, vocab, keep_words=keep_words), "vocab": vocab,
                     "words": [], "current": [], "speak": speak_words, "keep_words": keep_words}
        self.mode = "live"
        self.guard.block()
        await self.push_state()

    async def _live_step(self, x):
        lv = self.live
        dec = lv["decoder"]
        events = dec.push(x)
        for kind, value in events:
            if kind == "sound":
                lv["current"].append(value)
            else:
                if value:
                    lv["words"].append(value)
                    if lv["speak"]:
                        speak(value)
                lv["current"] = []
        stepped = dec.stepped
        if events or stepped:
            msg = {"type": "live", "n": self.tick, "events": events, "words": lv["words"][-40:],
                   "current": lv["current"]}
            if stepped:
                msg |= {"probs": dict(zip(lv["vocab"], (round(float(p), 3) for p in dec.probs))),
                        "infer_ms": round(dec.infer_ms, 2)}
            await self.send(msg)

    async def live_clear(self):
        if self.live:
            self.live["words"], self.live["current"] = [], []
            await self.push_state()

    # ------------------------------------------------------------------ stopping any mode

    async def stop(self):
        self.guard.allow()
        if self.live:
            for _, word in self.live["decoder"].flush():
                self.live["words"].append(word)
                if self.live["speak"]:
                    speak(word)
        was_recording = self.rec is not None
        self.mode, self.rec, self.live = "idle", None, None
        if was_recording:
            self.summary = self._summarize()
        await self.push_state()

    # ------------------------------------------------------------------ training

    async def train_start(self, epochs):
        if self.train["running"]:
            return
        epochs = max(5, int(epochs))
        self.train = {"running": True, "epochs": epochs, "history": [], "result": None, "log": []}
        await self.push_state()
        self._train_proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", os.path.join(HERE, "train.py"), self.data_dir, "--epochs", str(epochs),
            "--out", self.model_path, cwd=HERE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        asyncio.create_task(self._train_watch(self._train_proc))

    async def _train_watch(self, proc):
        result = {"sessions": []}
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            self.train["log"] = (self.train["log"] + [line])[-400:]
            msg = {"type": "train_line", "line": line}
            if m := EPOCH_LINE.search(line):
                point = {"epoch": int(m[1]), "loss": float(m[2]), "per": float(m[3]), "per_quiet": float(m[4])}
                self.train["history"].append(point)
                msg["epoch"] = point
            elif m := BEST_LINE.search(line):
                result["per"] = float(m[1])
            elif m := SESSION_LINE.search(line):
                result["sessions"].append({"name": m[1], "per": float(m[2]), "trials": int(m[3])})
            await self.send(msg)
        code = await proc.wait()
        self.train["running"] = False
        self.train["result"] = result if code == 0 and "per" in result else None
        self._train_proc = None
        self.summary = self._summarize()
        await self.push_state()
        await self.notice("Training finished" if code == 0 else "Training stopped", "info" if code == 0 else "warn")

    async def train_stop(self):
        if self._train_proc and self._train_proc.returncode is None:
            self._train_proc.terminate()

    # ------------------------------------------------------------------ sound inventory

    async def set_inventory(self, phonemes):
        phonemes = [p for p in phonemes if p in ARPABET]
        if not phonemes:
            await self.notice("Keep at least one sound", "warn")
            return
        with open(LANGUAGE_PATH) as f:
            lang = json.load(f)
        lang["phonemes"] = phonemes
        with open(LANGUAGE_PATH, "w") as f:
            json.dump(lang, f, indent=2)
        self.summary = self._summarize()
        await self.push_state()

    # ------------------------------------------------------------------ commands from the browser

    async def handle(self, msg):
        cmd = msg.get("cmd")
        if cmd == "record_start":
            await self.record_start(msg.get("n", 20))
        elif cmd == "accept":
            await self.record_accept()
        elif cmd == "redo":
            await self.record_redo()
        elif cmd == "undo":
            await self.record_undo()
        elif cmd == "stop":
            await self.stop()
        elif cmd == "live_start":
            await self.live_start(bool(msg.get("speak")), bool(msg.get("keep_words")))
        elif cmd == "live_clear":
            await self.live_clear()
        elif cmd == "train_start":
            await self.train_start(msg.get("epochs", 200))
        elif cmd == "train_stop":
            await self.train_stop()
        elif cmd == "inventory_set":
            await self.set_inventory(msg.get("phonemes", []))
        elif cmd == "refresh":
            self.summary = self._summarize()
            await self.push_state()

    async def disconnected(self):
        # Never leave the trackpad locked with nobody watching
        await asyncio.sleep(2.0)
        if not self.clients and self.mode != "idle":
            await self.stop()

    def close(self):
        self.guard.close()
        self.rt.__exit__(None, None, None)


def create_app(core):
    @asynccontextmanager
    async def lifespan(_):
        core.loop = asyncio.get_running_loop()
        ticker = asyncio.create_task(core.tick_loop())
        yield
        ticker.cancel()
        core.close()

    api = FastAPI(lifespan=lifespan)

    @api.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        core.clients.add(ws)
        await core.send(core.state(), ws)
        try:
            while True:
                await core.handle(json.loads(await ws.receive_text()))
        except WebSocketDisconnect:
            pass
        finally:
            core.clients.discard(ws)
            asyncio.create_task(core.disconnected())

    if os.path.isdir(UI_DIST):
        api.mount("/assets", StaticFiles(directory=os.path.join(UI_DIST, "assets")), name="assets")

        @api.get("/")
        async def index():
            return FileResponse(os.path.join(UI_DIST, "index.html"))
    else:
        @api.get("/")
        async def missing():
            return HTMLResponse("<p>The UI isn't built yet. Run <code>./setup.sh</code>.</p>")

    return api


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--data", default=DATA_DIR, help="folder of session files (default: data/)")
    p.add_argument("--model", default=os.path.join(HERE, "models", "decoder.pt"))
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--replay", metavar="FILE", help="demo/testing: replay a recorded session instead of the trackpad "
                                                   "(never locks the trackpad)")
    p.add_argument("--no-lock", action="store_true", help="never lock the trackpad pointer")
    args = p.parse_args()

    os.makedirs(args.data, exist_ok=True)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    torch.set_num_threads(1)  # live inference is one tiny step every 80 ms; training runs in its own process
    core = App(args.data, args.model, replay=args.replay, lock=not args.no_lock)
    if args.replay:
        print(f"Replaying {args.replay} instead of reading the trackpad")
    if core.guard.error:
        print(f"Trackpad lock unavailable: {core.guard.error}")
    url = f"http://localhost:{args.port}"
    print(f"Open {url}")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(core), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
