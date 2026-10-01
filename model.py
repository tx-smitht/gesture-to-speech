"""The decoder network, CTC decoding, and the error metric.

Pipeline for every step:
  last 14 bins of the 160-channel signal (280 ms "patch")  ->  input layer  ->  GRU (memory of what came before)
  ->  a score for every output symbol.  A new step happens every 4 bins (80 ms), like the BrainGate speech decoder.
Symbol 0 is the CTC "blank" ("nothing new happening right now").
"""

import torch
import torch.nn as nn

BLANK = "_"


class Decoder(nn.Module):
    def __init__(self, n_in, n_out, hidden=128, layers=2, dropout=0.2, patch_len=14, patch_stride=4):
        super().__init__()
        self.patch_len, self.patch_stride = patch_len, patch_stride
        self.inp = nn.Sequential(nn.Linear(n_in * patch_len, hidden), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, n_out)

    def patch(self, x):
        """[batch, bins, channels] -> [batch, steps, patch_len * channels]: overlapping 280 ms windows every 80 ms."""
        p = x.unfold(1, self.patch_len, self.patch_stride)       # [batch, steps, channels, patch_len]
        return p.transpose(2, 3).reshape(x.shape[0], p.shape[1], -1)

    def steps(self, n_bins):
        """How many output steps a signal of n_bins produces."""
        return (n_bins - self.patch_len) // self.patch_stride + 1

    def forward(self, patches, h=None):
        """patches: [batch, steps, patch_len * channels] -> scores [batch, steps, symbols], plus GRU memory."""
        y, h = self.rnn(self.inp(patches), h)
        return self.out(y), h


def greedy_decode(ids, vocab):
    """Best symbol per step -> merge repeats -> drop blanks.  [_ _ EY EY _ | _ OW] -> [EY | OW]"""
    out, prev = [], None
    for i in ids:
        if i != prev and vocab[i] != BLANK:
            out.append(vocab[i])
        prev = i
    return out


def edit_distance(a, b):
    """Fewest insertions/deletions/substitutions turning a into b (Levenshtein)."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def save_model(path, model, vocab, config):
    torch.save({"state": model.state_dict(), "vocab": vocab, "config": config}, path)


def load_model(path):
    ck = torch.load(path, weights_only=False)
    c = ck["config"]
    model = Decoder(c["n_in"], len(ck["vocab"]), c["hidden"], c["layers"],
                    patch_len=c["patch_len"], patch_stride=c["patch_stride"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["vocab"], c
