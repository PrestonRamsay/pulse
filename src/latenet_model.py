"""The latenet network: one embedding per categorical input, then a small feed-forward net."""

import torch
from torch import nn

from src.latenet_data import CATEGORICAL


class LateNet(nn.Module):
    def __init__(self, vocab_sizes, embedding_dims, n_numeric, hidden, dropout):
        super().__init__()
        # +1 row in each table for index 0, "never seen in training".
        self.embeddings = nn.ModuleList(
            nn.Embedding(vocab_sizes[c] + 1, embedding_dims[c]) for c in CATEGORICAL)
        width = sum(embedding_dims[c] for c in CATEGORICAL) + n_numeric
        layers = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, num):
        # cat: (batch, 5) whole numbers in CATEGORICAL order; num: (batch, 4) floats.
        vectors = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(vectors + [num], dim=1)
        return self.mlp(x).squeeze(1)  # one logit per row


def build_model(cfg, vocab, n_numeric):
    """The network described by the config, sized to the train vocabulary."""
    m = cfg["model"]
    sizes = {c: len(vocab[c]) for c in CATEGORICAL}
    return LateNet(sizes, m["embedding_dims"], n_numeric, m["hidden"], m["dropout"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def load_latenet(path):
    """Rebuild the trained model from models/latenet.pt. Returns (model, saved dict)."""
    saved = torch.load(path)
    model = build_model(saved["config"], saved["vocab"], saved["n_numeric"])
    model.load_state_dict(saved["state_dict"])
    model.eval()  # dropout off: this is for scoring, not training
    return model, saved