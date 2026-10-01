# Gesture-to-Speech

### TL;DR 
With Gesture-to-Speech, you can turn gestures on your trackpad into speech. Here's how it works:
- Record gestures on you trackpad that correspond to certain sounds
- Train a decoder on your gesture data
- Decode gestures into speech live

The current design draws from the published BrainGate speech
neuroprosthesis ([paper](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132) and [repo](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/main)).

### Backstory
A while back, my wife and I were bored on a flight. Naturally, we decided to create a way to "talk" to each other with just movements you can make with one hand. This app gives the user a way to create their own set of movements and decode those into speech.

I've been interested in Brain Computer Interfaces (BCIs) lately, and I wanted to build a project to practice decoding signals into speech. Because I don't have a brain implant, trackpad motion was a simple substitute. 

---


### Quick start (macOS, Apple Silicon or Intel; needs [uv](https://docs.astral.sh/uv/), Node, and the Xcode command-line tools; uv fetches Python 3.10 itself):

```bash
./setup.sh                      # Python env, the Swift trackpad lock, the web UI
uv run server.py                 # record, train and decode at http://localhost:8765
```

**Not in this repo:** recordings (`data/`, `recordings/`), taught gestures (`dictionary.json`), trained models
(`models/`) and your sound list (`language.json`) are gitignored and stay on your machine; they describe your private
gesture language. A default `language.json` is created on first run. **API keys** go in `.env` (copy `.env.example`),
which is gitignored too.


## Web app: record, train and decode in the browser

```bash
./setup.sh                      # once: Python packages, the Swift trackpad lock, the TypeScript UI
uv run server.py                 # opens http://localhost:8765
```

| Tab | What it does |
|---|---|
| **Record** | The calibration Copy Task. Keys: Enter = save · R = redo · ⌫ = undo last saved · Esc = finish |
| **Train** | Trains on everything in `data/` and shows the held-out error rate falling live, with a per-session breakdown |
| **Live** | Decodes as you go: transcript, the network's probability for each symbol every 80 ms, and inference time per step |
| **Signals** | A "neural signals" raster: one row per electrode, time scrolling, decoded sounds marked as white lines (word breaks in blue). *Spikes* shows each channel firing at a rate set by its activity (rate coding); *Exact values* shows what the decoder reads |
| **Sounds** | Choose your language's sounds from the 39 ARPAbet phonemes (saved to `language.json`) |

The electrode-array panel always shows the live 16×10 "virtual electrode" signal.

**Trackpad lock.** While recording or decoding, `bin/trackpad-guard` (Swift, `guard/TrackpadGuard.swift`) stops the
trackpad from moving the pointer, clicking, scrolling or triggering gestures, so your touches only become data. The
raw touch signal comes from a lower layer of macOS, so recording is unaffected. It needs **Accessibility** permission:
System Settings → Privacy & Security → Accessibility → turn on the app you start `server.py` from (Terminal, or
the Claude app if you use its terminal), then restart the server. **Esc always unlocks**, even if the browser is
closed, and the lock also releases itself if the server stops responding.

**Stack:** TypeScript + React (`ui/`) for the frontend; Python
(`server.py`, FastAPI + WebSocket) for the backend, reusing the decoder; Swift for the OS integration.

**Demo without a trackpad:** `uv run server.py --replay data/<session>.jsonl` replays a recorded session
through the whole pipeline in real time. Replays never lock the trackpad; `--no-lock` turns the lock off for live use too.

**UI development:** run `uv run server.py --no-browser`, then `cd ui && npm run dev` for hot reload.

## v2: Neural decoder trained on sentences (the BCI approach)

Modelled on the published BrainGate speech decoders ([Card et al., NEJM 2024](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132),
[code](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text)). You never label individual moves: you "say"
whole prompted sentences, and the network works out which part of the signal is which sound.

Everything here runs in the project's own Python environment (`.venv`, managed by uv). Activate it once per terminal
so plain `python` uses it, or prefix each command with `uv run`:

```bash
source .venv/bin/activate
```

**1. List your sounds** in `language.json`, using ARPAbet symbols (the full list with example words is in
`phonemes.py`). It starts with the five letter-name vowels: `EY IY AY OW UW` (ate, eat, hide, oat, two).

**2. Calibrate (the "Copy Task"):**

```bash
python collect.py -n 50
```

A prompt like `EY IY | OW   (ate eat / oat)` appears. Perform it on the trackpad, **pausing between words** (`|`), then
press Enter. Type `r` + Enter to redo, `q` + Enter to stop. Every trial is saved immediately to `data/`.

**3. Train:**

```bash
python train.py
```

This reports the phoneme error rate on held-out sentences it never trained on, plus a stress test with 2 s of extra
silence. The best model is saved to `models/decoder.pt`.

`--features` chooses the input maps: `basic` (default, one map), `size` (+ contact size), `orientation` (+ finger
angle) or `shape` (both: four 16×10 maps). The model remembers its feature set, so live decoding always matches; the
web app's Train page has the same choice. Experiments comparing them live in `ablations/`.

**4. Decode live:**

```bash
python live.py --speak
```

Sounds appear as you make them. With `--speak`, each word is spoken aloud when you pause after it.

**Try it without recording anything:** `python simulate.py -n 300` creates a simulated participant with an invented
language and writes 300 trials to `data_sim/`. Then run `python train.py data_sim/`.

| File | Job | BCI equivalent |
|---|---|---|
| `phonemes.py` | sound inventory, random prompts | the prompt corpus |
| `collect.py` | prompted calibration sessions | the Copy Task |
| `signals.py` | touches → 16×10 grid of "electrodes", 20 ms bins | electrode array, binned activity |
| `model.py` | 280 ms windows every 80 ms → GRU → sound scores; CTC decoding | the RNN phoneme decoder |
| `train.py` | CTC training, noise and silence augmentation, held-out evaluation | decoder training |
| `live.py` | streaming decoding, speech output | real-time use + text-to-speech |
| `simulate.py` | fake participant for testing the pipeline | simulated users |

## Experimental: real-time read-back (speak each word as it's decoded)

Speech BCIs usually wait for the whole sentence, then rescore it with a language model before speaking.
`readback.py` instead speaks each word the moment it's confident enough. It turns the sound probabilities into words
using a pronunciation tree (`lexicon.py`, `english/`) and a trigram language model conditioned on the words already
spoken. On simulated participants it matched the accuracy of waiting for the sentence, while speaking each word
~0.1-0.2 s after its last sound instead of ~4 s later. With a personal language model, most words come out before
they're finished. Write-up, results and next steps: [docs/realtime_readback.md](docs/realtime_readback.md).
Experiments: `ablations/realtime_readback.py`, `ablations/readback_variants.py`, `ablations/uniqueness_point.py`.

**On your own decoder:** `python live.py --readback` (or the Live tab's "Read back real words early") speaks the
common English words your sounds can say as soon as they're certain; anything else comes out as its sounds.
`python collect.py --words` records prompts made of those words. On real-word sentences built from the recorded
gestures it cut word errors from 32% to 25% and spoke a third of words before their space move (§7 of the write-up).

**Pronunciations for any word:** `python pronounce.py water tonight` looks words up in the CMU Pronouncing Dictionary
(downloaded by `./setup.sh`), falling back to `espeak-ng` (`brew install espeak-ng`) for words it doesn't have.
`--add` appends them to `english/pronunciations.txt`; `--check` compares that file with the dictionary.

## v1: Dictionary of gestures → symbols (template matching)

A **gesture** is everything from the first finger landing until every finger has lifted. Fingers down at the same time
form one chord. Teach each symbol by example, then decode:

```bash
python3 teach.py A        # do your "A" gesture 5-10 times, Ctrl+C when done
python3 teach.py B
python3 teach.py --list   # see what's taught
python3 decode.py         # live decoding; add --speak to hear it, -v to see measurements
```

- While you teach, each example prints what was measured: where each finger landed, how far it moved, how long it
  stayed down, contact size and pressure. If the new example looks like a symbol you already taught, you get a warning.
- Made a mistake? `python3 teach.py --undo A` drops the latest example. `--delete A` forgets A entirely.
- A symbol called `space` becomes a space in the decoded text.
- `python3 decode.py --file recordings/X.csv` decodes a saved recording, which is handy for comparing tweaks on the same data.

**How matching works** (all in `gestures.py`): each gesture becomes a few numbers per finger (landing x/y, movement
dx/dy, duration, contact ellipse size, peak pressure). A new gesture is compared against every stored example with the
same number of fingers, and the closest symbol wins, unless nothing is within `MATCH_THRESHOLD`, in which case you get `?`.
`SCALES` sets how much each measurement matters, so tune it as you learn what separates your symbols. Examples are
stored raw in `dictionary.json`, so changing the features or scales re-interprets every example you've already taught.

Things the trackpad can tell apart: a thumb makes a noticeably larger contact than a fingertip (about 12 mm long vs.
about 8 mm). The physical *click* isn't part of the touch data, but pressure goes up when you press harder.

## CSV columns

| column | meaning |
|---|---|
| `t` | seconds since recording started |
| `frame` | sensor frame number; all fingers seen at the same instant share one |
| `touch_id` | stays the same from when a finger lands until it lifts, so one touch-and-swipe is one ID |
| `state` | `hover_in_range` → `make_touch` (landed) → `touching` → `break_touch` (lifting) → `out_of_range` |
| `x`, `y` | position 0–1, origin at the **bottom-left** |
| `x_mm`, `y_mm` | the same position in millimetres |
| `vx`, `vy` | velocity the trackpad itself reports (normalized units/sec) |
| `size`, `pressure` | contact size, and roughly how hard/flat the finger is pressed |
| `angle`, `major_axis`, `minor_axis` | the contact is an ellipse: its orientation (radians) and axis lengths (≈ mm) |
| `finger_id`, `hand_id` | the trackpad's own guess at which finger/hand; not very reliable |

Rows with `hover_in_range` are fingers (often a thumb or palm) that are near the surface or only grazing it, not
pressing on it.

## How it works

`record.py` uses macOS's private `MultitouchSupport` framework (the same data the OS uses for gestures) through Python
`ctypes`. It's undocumented, so a future macOS update could break it. It needs no special permissions.
