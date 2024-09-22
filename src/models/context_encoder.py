# context_encoder.py

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class ContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model.context.encoder == 'roberta':
            self.encoder = RobertaContextEncoder(config)
        elif config.model.context.encoder == 'simple':
            self.encoder = SimpleContextEncoder(config)
        else:
            raise ValueError(f"Invalid encoder type: {config.model.context.encoder}")

    def forward(self, x):
        return self.encoder(x)

class RobertaContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.projection = nn.Linear(self.roberta.config.hidden_size, config.model.context.embed_dim)
        
        if not config.model.context.fine_tune:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.roberta(x)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        return self.projection(pooled_output)

class SimpleContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.model.context.vocab_size, config.model.context.embed_dim)

    def forward(self, x):
        return self.embedding(x)