from .contrastive_model import ContrastiveLearningModel
import torch.nn as nn
import torch

class CheapFakeModel(nn.Module):
    def __init__(self, config):
        super(CheapFakeModel, self).__init__()
        self.contrastive_model = ContrastiveLearningModel(config)
        self.dropout = nn.Dropout(0.5)
        
        self.classifier = nn.Linear(config.train.batch_size * 6, 1)
    def forward(self, image, context, caption):
        sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx = self.contrastive_model(image, context, caption)
        features = torch.cat([sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx], dim=1)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        output = self.classifier(features)
        return output
    
    def compute_loss(self, output, target):
        # contrastive_loss = self.contrastive_model.compute_loss(sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx)
        output = output.squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(output, target.float())
        return loss
        

        