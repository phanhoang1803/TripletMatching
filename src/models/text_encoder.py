import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class TextEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(TextEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.fc = nn.Linear(self.roberta.config.hidden_size, output_dim)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.roberta(**inputs)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)