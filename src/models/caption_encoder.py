# caption_encoder.py
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class CaptionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model.caption.encoder == 'roberta':
            self.encoder = RobertaCaptionEncoder(config)
        elif config.model.caption.encoder == 'simple':
            self.encoder = SimpleCaptionEncoder(config)
        else:
            raise ValueError(f"Invalid encoder type: {config.model.caption.encoder}")

    def forward(self, x):
        return self.encoder(x)

class RobertaCaptionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.projection = nn.Linear(self.roberta.config.hidden_size, config.model.caption.embed_dim)
        
        if not config.model.caption.fine_tune:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.roberta(x)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        return self.projection(pooled_output)

class SimpleCaptionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.model.caption.vocab_size, config.model.caption.embed_dim)

    def forward(self, x):
        return self.embedding(x)