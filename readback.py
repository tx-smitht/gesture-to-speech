"""Real-time read-back: speak each word as soon as the decoder is sure enough of it, not when the sentence ends.

The phoneme decoder (model.py) outputs, every 80 ms, a probability for each sound. This module turns that stream
into *words*, using two kinds of knowledge the network doesn't have:

    the lexicon         which sound sequences are real words (lexicon.Lexicon, a prefix tree)
    the language model  which words tend to follow which (lexicon.NgramLM)

WordBeamSearch keeps the few dozen most likely explanations of the signal so far ("hypotheses"), e.g.
    [i, need] + sounds "S AH"      (probably "some" -- or "something", or "sun"?)
    [i, need] + sounds "S"
    [i, knee] + sounds "S AH"
and updates them every step. Each hypothesis is a list of finished words plus a position in the prefix tree for the
word in progress. From all hypotheses together it can say how likely each word is to be the next word to speak:
`posterior()`. Once a word is spoken it can't be taken back, so `commit()` throws away every hypothesis that
disagrees, and the remaining hypotheses -- and the language model -- carry on with it as context.

*When* to commit is the job of a policy (see ablations/realtime_readback.py):
    word break     speak a word once the best hypothesis has seen its word break (today's live.py, plus words)
    threshold      speak as soon as one word's probability passes a threshold, even before the word is finished
    learned        a small classifier looks at the probabilities, the signal and the context and predicts
                   "if I speak now, will it be right?"
    sentence end   wait for the whole sentence, then pick the best word sequence (what an LLM rescorer does)

Simplified version of one step (the real code below adds CTC's blank/repeat bookkeeping and the language model):

    for hyp in beam:
        for sound in sounds:
            if the word in progress can continue with sound:      new hyp, score += log p(sound)
        if the word in progress is a whole word:                 finish it on a word break
    beam = the best 48 new hyps
"""

import math

import numpy as np

from model import BLANK
from phonemes import WORD_BREAK

NEG = -1e30


def logadd(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


class WordBeamSearch:
    """CTC prefix beam search constrained to a lexicon, with an n-gram language model and irreversible commits.

    symbols: the phoneme model's output symbols (vocab from model.load_model)
    alpha:   language-model weight (0 = ignore it; larger = trust it more than the signal)
    beta:    bonus per finished word (stops the search preferring fewer, longer words)
    beam:    how many hypotheses to keep
    continuous: words may also end WITHOUT a word break -- the next word's first sound starts straight after the
             last word's last sound, and the lexicon + LM work out where one word stops ("some thing" or
             "something"?). Lets someone skip the pause between words.
    """

    def __init__(self, lexicon, lm, symbols, alpha=0.8, beta=1.0, beam=48, prune=-12.0, continuous=False):
        self.lex, self.lm, self.alpha, self.beta, self.beam, self.prune = lexicon, lm, alpha, beta, beam, prune
        self.continuous = continuous
        self.symbols, self.sym = list(symbols), {s: i for i, s in enumerate(symbols)}
        self.blank, self.brk = self.sym[BLANK], self.sym[WORD_BREAK]
        self.on_path = [set(p) for p in lexicon.path]
        self.reset()

    def reset(self):
        # key = (finished words, node of the word in progress) -> [log p ending in blank, log p ending in a sound]
        self.hyps = {((), 0): [0.0, NEG]}
        self.committed = []          # units already spoken, in order
        self._s, self._L = {}, {}

    # ---- language model, scaled by alpha. s(u | ctx) = P(u | ctx) ** alpha; L(node) = log sum of s under node ----
    def _scaled(self, ctx):
        key = tuple(ctx[-2:])
        if key not in self._s:
            s = self.lm.probs(list(ctx)) ** self.alpha
            self._s[key] = s
            self._L[key] = np.log(self.lex.under @ s[:len(self.lex)] + 1e-300)
        return self._s[key], self._L[key]

    def _lookahead(self, words, node):
        """log of the (scaled) LM probability shared by all words still possible from `node`."""
        s, L = self._scaled(words)
        j = len(words)
        if j < len(self.committed):          # this word was already spoken: only it is allowed
            u = self.committed[j]
            return math.log(s[u]) if node in self.on_path[u] else NEG
        return L[node]

    def _last(self, words, node):
        if node:
            return self.sym[self.lex.sound_in[node]]
        return self.brk if words else None

    # ---- one 80 ms step ----
    def step(self, logp):
        """logp: log probabilities of every model symbol at this step."""
        new = {}

        def add(key, b, nb):
            cur = new.get(key)
            if cur is None:
                new[key] = [b, nb]
            else:
                cur[0], cur[1] = logadd(cur[0], b), logadd(cur[1], nb)

        live = [i for i in np.flatnonzero(logp > self.prune) if i not in (self.blank, self.brk)]
        child, unit_at = self.lex.child, self.lex.unit_at
        for (words, node), (pb, pnb) in self.hyps.items():
            total = logadd(pb, pnb)
            last = self._last(words, node)
            add((words, node), total + logp[self.blank], NEG)             # blank: nothing new
            if last is not None:
                add((words, node), NEG, pnb + logp[last])                 # the same symbol held a little longer
            if node == 0:                                                  # extra word breaks change nothing
                add((words, node), NEG, (pb if last == self.brk else total) + logp[self.brk])
            here = self._lookahead(words, node) if node or live else 0.0
            for i in live:                                                 # one more sound in this word
                nxt = child[node].get(self.symbols[i])
                if nxt is None:
                    continue
                delta = self._lookahead(words, nxt) - here
                if delta <= NEG / 2:
                    continue
                add((words, nxt), NEG, (pb if i == last else total) + logp[i] + delta)
            u = unit_at[node]
            if u is not None and (len(words) >= len(self.committed) or self.committed[len(words)] == u):
                s, _ = self._scaled(words)                                 # the word is finished: a word break
                finish = math.log(s[u]) - here + self.beta
                add((words + (u,), 0), NEG, total + logp[self.brk] + finish)
                if self.continuous:                                        # ...or the next word starts right away
                    done = words + (u,)
                    root = self._lookahead(done, 0)
                    for i in live:
                        nxt = child[0].get(self.symbols[i])
                        if nxt is None:
                            continue
                        delta = self._lookahead(done, nxt) - root
                        if delta > NEG / 2:
                            add((done, nxt), NEG, (pb if i == last else total) + logp[i] + finish + delta)
        best = sorted(new.items(), key=lambda kv: -logadd(*kv[1]))[:self.beam]
        self.hyps = dict(best)

    # ---- what the beam believes ----
    def posterior(self, k=None):
        """Probability of each word being word number k of the sentence (default: the next one to speak).
        Returns (vector over units, fraction of the probability where word k is already finished)."""
        k = len(self.committed) if k is None else k
        post, done, total = np.zeros(len(self.lex)), 0.0, 0.0
        top = max(logadd(*v) for v in self.hyps.values())
        for (words, node), (pb, pnb) in self.hyps.items():
            m = math.exp(logadd(pb, pnb) - top)
            total += m
            if len(words) > k:
                post[words[k]] += m
                done += m
                continue
            if len(words) == k:              # word k in progress: share m among the words still possible
                s, L = self._scaled(words)
                share = s[:len(self.lex)] * self.lex.under[node]
            else:                            # not reached word k yet (it was spoken early): the LM's guess
                s, _ = self._scaled(words + tuple(self.committed[len(words):k]))
                share = s[:len(self.lex)].copy()
            z = share.sum()
            if z > 0:
                post += m * share / z
        return post / total, done / total

    def best(self, finish=True):
        """The single most likely word sequence. finish=True: the sentence is over, so add the LM's end-of-sentence
        probability and drop hypotheses stuck in the middle of a word."""
        scored = []
        for (words, node), v in self.hyps.items():
            score = logadd(*v)
            if finish:
                u = self.lex.unit_at[node]
                if node and u is None:
                    continue                                               # stuck in the middle of a word
                if node:                                                   # ends on a whole word: finish it
                    score += math.log(self._scaled(words)[0][u]) - self._lookahead(words, node) + self.beta
                    words = words + (u,)
                score += math.log(self._scaled(words)[0][len(self.lex)])   # P(</s> | context)
            scored.append((score, words))
        return list(max(scored)[1]) if scored else []

    def commit(self, u):
        """Speak unit u as the next word: from now on every hypothesis must agree with it."""
        k = len(self.committed)
        self.committed.append(u)
        keep = {}
        for (words, node), v in self.hyps.items():
            if len(words) > k:
                ok = words[k] == u
            elif len(words) == k:
                ok = node in self.on_path[u]
            else:
                ok = True                    # still finishing an earlier (already spoken) word
            if ok:
                keep[words, node] = v
        self.hyps = keep or {(tuple(self.committed), 0): [0.0, NEG]}   # nothing agrees: restart from the spoken words
