import torch
import torch.nn as nn
from transformers import AutoTokenizer
class CaptionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model.caption.encoder == 'simple':
            self.encoder = SimpleCaptionEncoder(config)
        else:
            raise ValueError(f"Invalid encoder type: {config.model.caption.encoder}")

    def forward(self, x):
        x = self.encoder(x)
        return x

class SimpleCaptionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.model.caption.vocab_size, config.model.caption.embed_dim)

    def forward(self, x):
        return self.embedding(x)