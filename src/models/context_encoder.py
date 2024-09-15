import torch
import torch.nn as nn
from transformers import AutoTokenizer

class ContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model.context.encoder == 'simple':
            self.encoder = SimpleContextEncoder(config)
        else:
            raise ValueError(f"Invalid encoder type: {config.model.context.encoder}")

    def forward(self, x):
        return self.encoder(x)

class SimpleContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.model.context.vocab_size, config.model.context.embed_dim)

    def forward(self, x):
        return self.embedding(x)

