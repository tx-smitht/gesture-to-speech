"""The sound inventory: which phonemes your language has moves for, and the prompts used to calibrate.

Sounds are written in ARPAbet, the standard used by the CMU Pronouncing Dictionary and by the BrainGate speech
decoders. `|` marks a word break (a pause between words -- the "silence" token of the speech BCIs).

Your inventory lives in language.json. Add a sound there once you've designed a move for it.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
LANGUAGE_PATH = os.path.join(HERE, "language.json")
WORD_BREAK = "|"

# All 39 ARPAbet phonemes, each with an example word containing that sound.
ARPABET = {
    "AA": "father", "AE": "at", "AH": "hut", "AO": "ought", "AW": "cow", "AY": "hide", "EH": "Ed", "ER": "hurt",
    "EY": "ate", "IH": "it", "IY": "eat", "OW": "oat", "OY": "toy", "UH": "hood", "UW": "two",
    "B": "be", "CH": "cheese", "D": "dee", "DH": "thee", "F": "fee", "G": "green", "HH": "he", "JH": "gee",
    "K": "key", "L": "lee", "M": "me", "N": "knee", "NG": "ping", "P": "pee", "R": "read", "S": "sea",
    "SH": "she", "T": "tea", "TH": "theta", "V": "vee", "W": "we", "Y": "yield", "Z": "zee", "ZH": "seizure",
}

DEFAULT_LANGUAGE = {
    "phonemes": ["EY", "IY", "AY", "OW", "UW", "S", "M", "T"],
    "note": "Sounds you have a move for, in ARPAbet (see phonemes.py for the full list). "
            "Word breaks are a pause between words.",
}


def load_inventory(path=LANGUAGE_PATH):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_LANGUAGE, f, indent=2)
    with open(path) as f:
        try:
            phonemes = json.load(f)["phonemes"]
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} has a typo on line {e.lineno}, column {e.colno}: {e.msg}.\n"
                             f"Common causes: a line break inside quotes, a missing comma, or a comma after the "
                             f"last item. It should look like:\n"
                             f'{{\n  "phonemes": ["EY", "IY", "S"]\n}}')
    unknown = [p for p in phonemes if p not in ARPABET]
    if unknown:
        raise SystemExit(f"{path}: not ARPAbet phonemes: {unknown}. Valid: {' '.join(ARPABET)}")
    return phonemes


BALANCE_SMOOTHING = 20  # weight = 1 / (examples + this): keeps a brand-new sound from taking over every prompt


def make_prompt(inventory, rng=random, counts=None):
    """A random 'sentence' of 2-4 'words', each 1-3 sounds from the inventory, e.g. ['EY', 'IY', '|', 'OW', '|'].
    Every word -- including the last -- ends with a word break, so every word ends the same way.

    counts: examples recorded so far per sound. When given, sounds with fewer examples are picked more often, so
    a newly added sound catches up quickly. Update it as trials are saved and the favouring fades as it catches up.
    """
    weights = [1 / (counts.get(s, 0) + BALANCE_SMOOTHING) for s in inventory] if counts is not None else None
    pick = (lambda: rng.choices(inventory, weights)[0]) if weights else (lambda: rng.choice(inventory))
    tokens = []
    for _ in range(rng.randint(2, 4)):
        tokens += [pick() for _ in range(rng.randint(1, 3))] + [WORD_BREAK]
    return tokens


def count_sounds(prompts):
    """Examples of each sound in a list of prompts."""
    counts = {}
    for prompt in prompts:
        for tok in prompt:
            if tok != WORD_BREAK:
                counts[tok] = counts.get(tok, 0) + 1
    return counts


def show(tokens):
    """'EY IY | OW   (ate eat / oat)' -- the symbols plus example words to read them by."""
    hints = " ".join("/" if t == WORD_BREAK else ARPABET.get(t, t) for t in tokens)
    return f"{' '.join(tokens)}   ({hints})"
