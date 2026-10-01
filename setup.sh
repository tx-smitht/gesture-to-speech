#!/bin/bash
# One-time setup for the web app: Python packages, the trackpad lock (Swift), the UI (TypeScript)
# and the pronunciation dictionary.
# Each step shows its latest 10 lines of output, indented, while it runs.
set -e
cd "$(dirname "$0")"

WINDOW=10      # lines of live output shown per step
INDENT="    "

# step "title" command...  Run a command, showing a rolling window of its output under the title.
# The window collapses when the step succeeds; on failure it stays, and the full log is kept.
step() {
  local title=$1; shift
  echo "$title"
  local log; log=$(mktemp -t setup)
  local start=$SECONDS status

  if [ ! -t 1 ]; then                      # not a terminal (CI, piped): plain indented output
    set +e; "$@" 2>&1 | tee "$log" | sed "s/^/$INDENT/"; status=${PIPESTATUS[0]}; set -e
  else
    local cols; cols=$(tput cols 2>/dev/null) || cols=80
    local width=$(( cols - ${#INDENT} - 1 )) shown=0 pid
    draw() {
      local lines n=0
      # \r-style progress bars become separate lines; long lines are cut so each takes one row
      lines=$(tr '\r' '\n' < "$log" | grep -v '^[[:space:]]*$' | tail -n "$WINDOW" | cut -c1-"$width")
      [ "$shown" -gt 0 ] && printf '\033[%dA' "$shown"
      if [ -n "$lines" ]; then
        while IFS= read -r l; do printf '\033[K\033[2m%s%s\033[0m\n' "$INDENT" "$l"; n=$((n + 1)); done <<< "$lines"
      fi
      shown=$n
    }
    "$@" > "$log" 2>&1 & pid=$!
    trap 'kill $pid 2>/dev/null; echo; exit 130' INT TERM
    while kill -0 "$pid" 2>/dev/null; do draw; sleep 0.1; done
    set +e; wait "$pid"; status=$?; set -e
    trap - INT TERM
    if [ "$status" -eq 0 ]; then
      [ "$shown" -gt 0 ] && printf '\033[%dA\033[J' "$shown"      # collapse the window
    else
      draw
    fi
  fi

  if [ "$status" -ne 0 ]; then
    echo "${INDENT}Failed (exit $status). Full output: $log"
    exit "$status"
  fi
  echo "${INDENT}done ($((SECONDS - start))s)"
  rm -f "$log"
}

if ! command -v uv > /dev/null; then
  echo "uv is not installed. Install it, then re-run ./setup.sh:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh     (or: brew install uv)"
  exit 1
fi

build_ui() { cd ui && npm install --no-fund --no-audit && npm run build; }

# --no-cache: torch is large and the wheel isn't worth keeping in uv's cache after install
step "1/4  Python packages (uv, .venv)" uv sync --no-cache
step "2/4  Trackpad lock (Swift)"       bash -c 'mkdir -p bin && swiftc -O guard/TrackpadGuard.swift -o bin/trackpad-guard'
step "3/4  Web UI (TypeScript)"         build_ui
step "4/4  Pronunciation dictionary + word-frequency list (english/, ~3.7 MB)" \
  bash -c '[ -f english/cmudict.dict ] && [ -f english/common_words.txt ] && echo "already downloaded" || uv run pronounce.py --download'
command -v espeak-ng > /dev/null || echo "${INDENT}optional: brew install espeak-ng  (pronounces words the dictionary doesn't have)"

echo
echo "Done. Start the app with:  uv run server.py"
