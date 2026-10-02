"""Say decoded sounds out loud.

    speak(["M", "AY"])        # says "my"
    speak(["EY", "T", "OW"])  # no English word sounds like that: says the sounds themselves

macOS `say` used to take phonemes directly ("[[inpt PHON]]mAY"), but current macOS no longer understands that
command and reads the brackets out loud instead. So:
  1. If the sounds are an English word (CMU dictionary; the most common spelling), the Mac voice speaks the word.
     Words that sound the same are the same sounds, so any spelling is right. It's spoken by bin/speaker
     (guard/Speaker.swift), which keeps the voice loaded: a word starts within a few milliseconds, where starting
     `say` for each word took over a second. Without bin/speaker, `say` is used.
  2. Otherwise espeak-ng (`brew install espeak-ng`) speaks the exact sounds from phoneme input -- a more robotic
     voice, but every sound is right. Without espeak-ng, `say` reads a rough respelling ("ay toh").
"""

import os
import queue
import shutil
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

# ARPAbet -> espeak-ng's phoneme mnemonics, used inside [[ ]]
ESPEAK = {
    "AA": "A:", "AE": "a", "AH": "V", "AO": "O:", "AW": "aU", "AY": "aI", "EH": "E", "ER": "3:", "EY": "eI",
    "IH": "I", "IY": "i:", "OW": "oU", "OY": "OI", "UH": "U", "UW": "u:",
    "B": "b", "CH": "tS", "D": "d", "DH": "D", "F": "f", "G": "g", "HH": "h", "JH": "dZ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "N", "P": "p", "R": "r", "S": "s", "SH": "S", "T": "t", "TH": "T", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "Z",
}
# Last resort for `say` without espeak-ng: a rough English respelling of each sound
RESPELL = {
    "AA": "ah", "AE": "a", "AH": "uh", "AO": "aw", "AW": "ow", "AY": "eye", "EH": "eh", "ER": "er", "EY": "ay",
    "IH": "ih", "IY": "ee", "OW": "oh", "OY": "oy", "UH": "u", "UW": "oo",
    "B": "b", "CH": "ch", "D": "d", "DH": "th", "F": "f", "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}

_words = None
_checked = {}


def _word_index():
    """Pronunciation -> candidate spellings to say, best first: common words (most common first), then any
    dictionary word that's also in the system word list (so abbreviations like "cc" aren't used). Only a word's
    main pronunciation counts, because that's how a speech voice reads it ("a" is read "uh", not "ay")."""
    global _words
    if _words is None:
        _words = {}
        try:
            from pronounce import cmudict_all
            entries = cmudict_all()
        except Exception:
            entries = {}
        common = os.path.join(HERE, "english", "common_words.txt")
        real = set()
        if os.path.exists("/usr/share/dict/words"):
            real = {w for w in open("/usr/share/dict/words").read().split() if w.islower()}
        if os.path.exists(common):
            for word in open(common).read().split():
                # very short spellings only if they're real words: a voice may spell "ou" out letter by letter
                if entries.get(word) and (len(word) > 2 or not real or word in real):
                    _words.setdefault(entries[word][0], []).append(word)
        if real:
            for word in sorted({w for w in real if len(w) > 1} & set(entries)):
                if word not in _words.get(entries[word][0], []):
                    _words.setdefault(entries[word][0], []).append(word)
    return _words


def _reads_as(word):
    """How a text-to-speech voice reads this spelling, using espeak-ng's English rules as a stand-in for the Mac
    voice (macOS no longer reports its own pronunciations). None if espeak-ng isn't installed."""
    if not shutil.which("espeak-ng"):
        return None
    from pronounce import espeak
    return espeak(word)


def word_for(sounds):
    """An English spelling a voice will read as exactly these sounds, or None."""
    key = tuple(sounds)
    if key not in _checked:
        _checked[key] = None
        for word in _word_index().get(key, [])[:5]:
            reading = _reads_as(word)
            if reading is None or reading == key:   # no checker available, or it reads right
                _checked[key] = word
                break
    return _checked[key]


def espeak_phonemes(sounds):
    """['EY', 'T', 'OW'] -> "eIt'oU": espeak-ng phoneme input, stressing the first vowel."""
    out, stressed = [], False
    for s in sounds:
        if s in VOWELS and not stressed:
            out.append("'")
            stressed = True
        out.append("@" if s == "AH" and stressed and out[-1] != "'" else ESPEAK[s])
    return "".join(out)


def command(sounds):
    """How to say these sounds: ["say", word] (sent to bin/speaker when it's running), or an espeak-ng command."""
    word = word_for(sounds)
    if word:
        return ["say", word]
    exe = shutil.which("espeak-ng")
    if exe:
        return [exe, "-v", "en-us", f"[[{espeak_phonemes(sounds)}]]"]
    return ["say", " ".join(RESPELL[s] for s in sounds)]


SPEAKER = os.path.join(HERE, "bin", "speaker")
_queue = None
_speaker = None


def _speaker_process():
    """bin/speaker, started once and kept running (None if it isn't built)."""
    global _speaker
    if _speaker is None or _speaker.poll() is not None:
        _speaker = None
        if os.access(SPEAKER, os.X_OK):
            _speaker = subprocess.Popen([SPEAKER], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
    return _speaker


def _say(cmd):
    if cmd[0] == "say":
        proc = _speaker_process()
        if proc is not None:
            try:
                proc.stdin.write(cmd[1].replace("\n", " ") + "\n")
                proc.stdin.flush()
                return
            except (BrokenPipeError, OSError):
                pass
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _worker():
    while True:
        sounds = _queue.get()
        try:  # looking a word up can take a moment the first time; it happens here, not in the decoding loop
            _say(command(sounds))
        except Exception:
            pass


def speak(sounds):
    """Say sounds out loud. Returns immediately; words are spoken in order by a background thread."""
    global _queue
    if not sounds:
        return
    if _queue is None:
        _queue = queue.Queue()
        threading.Thread(target=_worker, daemon=True).start()
    _queue.put(list(sounds))


def warm_up():
    """Load the word list and start the speech voice now, so the first spoken word isn't delayed."""
    threading.Thread(target=lambda: (_speaker_process(), _word_index()), daemon=True).start()
