import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(ImageEncoder, self).__init__()
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(self.config.hidden_size, output_dim)

    def forward(self, x):
        outputs = self.vit(x)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)