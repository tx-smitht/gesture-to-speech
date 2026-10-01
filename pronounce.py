#!/usr/bin/env python3
"""Text -> ARPAbet sounds, for any English word.

    python pronounce.py water something hungry     # look words up
    python pronounce.py --check                    # compare english/pronunciations.txt with the dictionary
    python pronounce.py --add tonight blanket      # append words to english/pronunciations.txt
    python pronounce.py --download                 # fetch the CMU dictionary (~3.6 MB) + a word-frequency list

Two sources, tried in order:
    1. the CMU Pronouncing Dictionary (english/cmudict.dict): 135k words, hand-checked, the standard that ARPAbet
       and the BrainGate speech decoders use. Stress digits are dropped (AH0 -> AH).
    2. espeak-ng (`brew install espeak-ng`): a speech synthesizer that can pronounce ANY spelling, including
       names and made-up words, by rule. It outputs IPA, which is converted to ARPAbet here. Less reliable.

(macOS used to do this itself -- NSSpeechSynthesizer.phonemes(from:) -- but that returns nothing on current macOS
for every voice.)
"""

import argparse
import os
import shutil
import subprocess
import urllib.request

from phonemes import ARPABET

HERE = os.path.dirname(os.path.abspath(__file__))
CMUDICT = os.path.join(HERE, "english", "cmudict.dict")
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
# The 10,000 most common English words, most common first (Google Web Trillion Word Corpus via Peter Norvig and
# Josh Kaufman; personal/research use). Used to pick which words the decoder knows and how likely each is.
COMMON_WORDS = os.path.join(HERE, "english", "common_words.txt")
COMMON_WORDS_URL = ("https://raw.githubusercontent.com/first20hours/google-10000-english/master/"
                    "google-10000-english-no-swears.txt")
PRONUNCIATIONS = os.path.join(HERE, "english", "pronunciations.txt")

# IPA (as espeak-ng writes American English) -> ARPAbet. Longest match first, so "aɪ" wins over "a".
IPA = {
    "eɪ": "EY", "aɪ": "AY", "ɔɪ": "OY", "aʊ": "AW", "oʊ": "OW", "əʊ": "OW", "tʃ": "CH", "dʒ": "JH",
    "ɑ": "AA", "ɒ": "AA", "æ": "AE", "a": "AE", "ʌ": "AH", "ə": "AH", "ɐ": "AH", "ɚ": "ER", "ɝ": "ER", "ɜ": "ER",
    "ɔ": "AO", "o": "OW", "ɛ": "EH", "e": "EH", "ɪ": "IH", "ᵻ": "IH", "i": "IY", "ʊ": "UH", "u": "UW",
    "b": "B", "d": "D", "ð": "DH", "f": "F", "ɡ": "G", "g": "G", "h": "HH", "k": "K", "l": "L", "m": "M",
    "n": "N", "ŋ": "NG", "p": "P", "ɹ": "R", "r": "R", "s": "S", "ʃ": "SH", "t": "T", "ɾ": "T", "ʔ": "T",
    "θ": "TH", "v": "V", "w": "W", "j": "Y", "z": "Z", "ʒ": "ZH", "x": "K",
}
IGNORE = set("ˈˌː.ˑ̩ ̩˞-")   # stress, length, syllable marks


def download():
    """Fetch with curl (macOS's own certificates; the python.org Python often has none installed), else urllib."""
    for url, path in ((CMUDICT_URL, CMUDICT), (COMMON_WORDS_URL, COMMON_WORDS)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        if shutil.which("curl"):
            subprocess.run(["curl", "-sSfL", "-o", tmp, url], check=True)
        else:
            urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)  # only replace a good copy once the download has finished
        print(f"{url.rsplit('/', 1)[-1]} -> {path}")


_cmu = None


def cmudict_all():
    """{"read": [("R", "IY", "D"), ("R", "EH", "D")], ...} -- every pronunciation, most common first."""
    global _cmu
    if _cmu is None:
        _cmu = {}
        if os.path.exists(CMUDICT):
            with open(CMUDICT, encoding="latin-1") as f:
                for line in f:
                    word, *sounds = line.split("#")[0].split()
                    if sounds:                            # "word(2)" lines are alternative pronunciations
                        _cmu.setdefault(word.split("(")[0], []).append(tuple(s.rstrip("012") for s in sounds))
    return _cmu


_first = None


def cmudict():
    """{"water": ("W", "AO", "T", "ER"), ...} -- each word's first (most common) pronunciation."""
    global _first
    if _first is None:
        _first = {w: p[0] for w, p in cmudict_all().items()}
    return _first


def ipa_to_arpabet(ipa):
    out, i = [], 0
    while i < len(ipa):
        if ipa[i] in IGNORE:
            i += 1
            continue
        for n in (2, 1):
            if ipa[i:i + n] in IPA:
                out.append(IPA[ipa[i:i + n]])
                i += n
                break
        else:
            raise ValueError(f"no ARPAbet sound for {ipa[i]!r} in {ipa!r}")
    # espeak writes "r-coloured" vowels as vowel + ɹ (and sometimes ɹɹ); ARPAbet uses ER for "uh-r" / "ur", and
    # never repeats a sound inside a word
    merged = []
    for s in out:
        if s == "R" and merged and merged[-1] == "AH":
            merged[-1] = "ER"
        elif not (merged and (s == merged[-1] or (s == "R" and merged[-1] == "ER"))):
            merged.append(s)
    return tuple(merged)


def espeak(word):
    exe = shutil.which("espeak-ng")
    if not exe:
        return None
    ipa = subprocess.run([exe, "-q", "-v", "en-us", "--ipa", word], capture_output=True, text=True).stdout.strip()
    return ipa_to_arpabet(ipa) if ipa else None


def pronounce(word):
    """(sounds, source) for one word; source is "cmudict" or "espeak-ng". None if neither can do it."""
    word = word.lower()
    if word in cmudict():
        return cmudict()[word], "cmudict"
    sounds = espeak(word)
    return (sounds, "espeak-ng") if sounds else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("words", nargs="*")
    p.add_argument("--check", action="store_true", help="compare english/pronunciations.txt with CMUdict")
    p.add_argument("--add", action="store_true", help="append the words to english/pronunciations.txt")
    p.add_argument("--download", action="store_true", help="fetch the CMU Pronouncing Dictionary")
    args = p.parse_args()

    if args.download:
        download()
    if not os.path.exists(CMUDICT):
        print(f"(No {os.path.relpath(CMUDICT, HERE)} -- run `python pronounce.py --download`. Using espeak-ng only.)")

    if args.check:
        from lexicon import load_pronunciations
        mine = load_pronunciations()
        differ = [(w, p) for w, p in sorted(mine.items()) if w in cmudict() and cmudict()[w] != p]
        missing = [w for w in mine if w not in cmudict()]
        for w, p in differ:
            print(f"  {w:<16} mine: {' '.join(p):<28} cmudict: {' '.join(cmudict()[w])}")
        print(f"{len(mine)} words: {len(mine) - len(differ) - len(missing)} match CMUdict, {len(differ)} differ, "
              f"{len(missing)} not in it{': ' + ' '.join(missing) if missing else ''}")

    new = []
    for w in args.words:
        r = pronounce(w)
        if r is None:
            print(f"  {w:<16} ?  (not in CMUdict, and espeak-ng isn't installed)")
            continue
        sounds, source = r
        assert all(s in ARPABET for s in sounds), sounds
        print(f"  {w:<16} {' '.join(sounds):<28} ({source}: {' '.join(ARPABET[s] for s in sounds)})")
        new.append((w.lower(), sounds))
    if args.add and new:
        from lexicon import load_pronunciations
        have = load_pronunciations()
        with open(PRONUNCIATIONS, "a") as f:
            for w, sounds in new:
                if w not in have:
                    f.write(f"{w} {' '.join(sounds)}\n")
        print(f"added {sum(w not in have for w, _ in new)} words to {os.path.relpath(PRONUNCIATIONS, HERE)}")


if __name__ == "__main__":
    main()
