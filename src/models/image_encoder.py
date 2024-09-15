import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048, config.model.image.embed_dim)

    def forward(self, x):
        return self.encoder(x)