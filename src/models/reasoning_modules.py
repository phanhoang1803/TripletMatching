import torch
import torch.nn as nn

class ReasoningModules(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512):
        super(ReasoningModules, self).__init__()
        self.visual_reasoning = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.textual_reasoning = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.contextual_reasoning = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.integration = nn.Linear(input_dim * 3, input_dim)

    def forward(self, features):
        visual_features = self.visual_reasoning(features[0])
        textual_features = self.textual_reasoning(features[1])
        contextual_features = self.contextual_reasoning(features[2])
        
        combined = torch.cat([visual_features, textual_features, contextual_features], dim=-1)
        integrated = self.integration(combined)
        return integrated