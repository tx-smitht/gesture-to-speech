"""Python side of bin/trackpad-guard (guard/TrackpadGuard.swift): lock and unlock the trackpad's pointer.

    guard = TrackpadGuard(on_event=print)   # on_event receives "escape", "watchdog", "blocked", ...
    guard.block(); ...; guard.allow(); guard.close()

If the helper can't run (not built, or no Accessibility permission), `available` is False and `error` says why;
block()/allow() then do nothing, so recording still works -- the cursor just isn't frozen.
"""

import os
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "bin", "trackpad-guard")


class TrackpadGuard:
    def __init__(self, on_event=None, enabled=True):
        self.on_event = on_event or (lambda e: None)
        self.available, self.locked, self.error = False, False, None
        self._ready = threading.Event()
        self._proc = None
        if not enabled:
            self.error = "turned off"
            return
        if not os.path.exists(BINARY):
            self.error = "not built -- run ./setup.sh"
            return
        self._proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._ping, daemon=True).start()
        self._ready.wait(timeout=3)
        if not self.available and not self.error:
            self.error = "helper did not start"

    def _read(self):
        for line in self._proc.stdout:
            line = line.strip()
            if line == "ready":
                self.available = True
                self._ready.set()
            elif line.startswith("error:"):
                self.error = line[len("error:"):].strip()
                self._ready.set()
            elif line in ("escape", "watchdog", "allowed"):
                self.locked = False
            elif line == "blocked":
                self.locked = True
            self.on_event(line)
        self.available = self.locked = False

    def _ping(self):
        while self._proc and self._proc.poll() is None:
            self._send("ping")
            time.sleep(1.0)

    def _send(self, cmd):
        try:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, AttributeError):
            pass

    def block(self):
        if self.available:
            self._send("block")

    def allow(self):
        if self.available:
            self._send("allow")

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._send("quit")
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
