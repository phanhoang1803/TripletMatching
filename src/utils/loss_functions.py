import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletMatchingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMatchingLoss, self).__init__()
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, features):
        classification_loss = self.ce_loss(outputs, labels)
        
        # Triplet loss
        anchor, positive, negative = torch.split(features, features.size(0) // 3)
        triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=self.margin)
        
        # Combine losses
        total_loss = classification_loss + triplet_loss
        return total_loss