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
    unknown: log-probability penalty for a word that ISN'T in the lexicon (None = only lexicon words allowed).
             With it, any sound string can still come out -- spoken as the sounds themselves -- but a real word
             wins whenever the signal fits one about as well. Needed when the person may say things the word
             list doesn't have (names, calibration prompts of random sounds).

    prefix_guard: a word that is the start of a longer word ("a" -> "aim", "i" -> "ice") is only spoken once the
             beam has seen it END (its word break), not merely because it's likely. Without context, how often a
             word is used says nothing about whether the person is done with it.

    A word is either a lexicon unit (an int) or, for unknown words, the tuple of its sounds. The word in progress is
    a prefix-tree node (int) or, for an unknown word, the tuple of its sounds so far.
    """

    def __init__(self, lexicon, lm, symbols, alpha=0.8, beta=1.0, beam=48, prune=-12.0, continuous=False,
                 unknown=None, prefix_guard=False):
        self.lex, self.lm, self.alpha, self.beta, self.beam, self.prune = lexicon, lm, alpha, beta, beam, prune
        self.continuous, self.unknown, self.prefix_guard = continuous, unknown, prefix_guard
        # words that continue into longer words: the end node of their path has more than one word below it
        self.is_prefix = np.array([lexicon.under[path[-1]].sum() > 1 for path in lexicon.path])
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
            if isinstance(u, tuple):         # ...and it was an unknown word, so no lexicon path fits
                return NEG
            return math.log(s[u]) if node in self.on_path[u] else NEG
        return L[node]

    def _last(self, words, node):
        if isinstance(node, tuple):
            return self.sym[node[-1]]
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
            j = len(words)
            forced = self.committed[j] if j < len(self.committed) else None
            if isinstance(node, tuple):                                    # an unknown word: any sound may follow
                for i in live:
                    longer = node + (self.symbols[i],)
                    if forced is None or (isinstance(forced, tuple) and forced[:len(longer)] == longer):
                        add((words, longer), NEG, (pb if i == last else total) + logp[i])
                if (forced is None and node not in self.lex.unit_of) or forced == node:
                    add((words + (node,), 0), NEG, total + logp[self.brk] + self.beta)
                continue
            if node == 0 and self.unknown is not None:                    # start an unknown word
                for i in live:
                    first = (self.symbols[i],)
                    if forced is None or (isinstance(forced, tuple) and forced[:1] == first):
                        add((words, first), NEG, total + logp[i] + self.unknown)
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
        Returns (vector over units, fraction of the probability where word k is already finished,
        {sounds: probability} for finished unknown words). Unknown words still in progress count towards nothing:
        they could still become anything."""
        k = len(self.committed) if k is None else k
        post, done, total, unknown = np.zeros(len(self.lex)), 0.0, 0.0, {}
        ended = np.zeros(len(self.lex))     # the part of post where word k has been seen to END
        top = max(logadd(*v) for v in self.hyps.values())
        for (words, node), (pb, pnb) in self.hyps.items():
            m = math.exp(logadd(pb, pnb) - top)
            total += m
            if len(words) > k:
                if isinstance(words[k], tuple):
                    unknown[words[k]] = unknown.get(words[k], 0.0) + m
                else:
                    post[words[k]] += m
                    ended[words[k]] += m
                done += m
                continue
            if len(words) == k and isinstance(node, tuple):
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
        self.ended = ended / total
        return post / total, done / total, {w: m / total for w, m in unknown.items()}

    def decide(self, threshold, deadline=True):
        """The next word to speak: the most likely one if it is at least `threshold` likely. With `deadline`, once
        most of the beam has seen the word END (its word break), the best guess is spoken even if it's less sure
        than that -- so a word is never later than it would be without read-back. Otherwise None (wait)."""
        post, done, unknown = self.posterior()
        if self.prefix_guard:              # a word that could still grow counts only once it has ended
            post = np.where(self.is_prefix, self.ended, post)
        u = int(post.argmax())
        w, p = max(unknown.items(), key=lambda kv: kv[1], default=(None, 0.0))
        if max(post[u], p) >= threshold or (deadline and done >= 0.5):
            if post[u] >= p:
                return u
            return w
        return None

    def sounds(self, word):
        """A word (unit id or unknown-word tuple) -> its sounds."""
        return list(word) if isinstance(word, tuple) else list(self.lex.prons[word])

    def name(self, word):
        """How to show a word: "see/sea", or the sounds of an unknown word ("S EY M?")."""
        return " ".join(word) + "?" if isinstance(word, tuple) else self.lex.names[word]

    def best(self, finish=True):
        """The single most likely word sequence. finish=True: the sentence is over, so add the LM's end-of-sentence
        probability and drop hypotheses stuck in the middle of a word."""
        scored = []
        for (words, node), v in self.hyps.items():
            score = logadd(*v)
            if finish and isinstance(node, tuple):                         # ends on an unknown word: finish it
                if node in self.lex.unit_of:
                    continue
                score += self.beta
                words = words + (node,)
            elif finish:
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
        """Speak word u (unit id, or sounds tuple for an unknown word) next: every hypothesis must now agree."""
        k = len(self.committed)
        self.committed.append(u)
        keep = {}
        for (words, node), v in self.hyps.items():
            if len(words) > k:
                ok = words[k] == u
            elif len(words) == k:
                if isinstance(u, tuple):
                    ok = isinstance(node, tuple) and u[:len(node)] == node
                else:
                    ok = not isinstance(node, tuple) and node in self.on_path[u]
            else:
                ok = True                    # still finishing an earlier (already spoken) word
            if ok:
                keep[words, node] = v
        self.hyps = keep or {(tuple(self.committed), 0): [0.0, NEG]}   # nothing agrees: restart from the spoken words
