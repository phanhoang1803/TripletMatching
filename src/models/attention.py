import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x1, x2):
        # Ensure x1 and x2 have the same feature dimension
        if x1.size(-1) != x2.size(-1):
            raise ValueError(f"Feature dimensions of x1 ({x1.size(-1)}) and x2 ({x2.size(-1)}) must match")

        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        return attn @ v