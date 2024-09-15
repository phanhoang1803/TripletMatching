import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .context_encoder import ContextEncoder
from .caption_encoder import CaptionEncoder
from .attention import CrossAttention

class ContrastiveLearningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(config)
        self.context_encoder = ContextEncoder(config)
        self.caption_encoder = CaptionEncoder(config)
        
        self.image_projector = nn.Linear(config.model.image.embed_dim, config.model.contrastive.attn_feature_dim)
        self.context_projector = nn.Linear(config.model.context.embed_dim, config.model.contrastive.attn_feature_dim)
        self.caption_projector = nn.Linear(config.model.caption.embed_dim, config.model.contrastive.attn_feature_dim)
        
        self.cross_attention1 = CrossAttention(config.model.contrastive.attn_feature_dim)
        self.cross_attention2 = CrossAttention(config.model.contrastive.attn_feature_dim)
        self.cross_attention3 = CrossAttention(config.model.contrastive.attn_feature_dim)
        
        self.temperature = nn.Parameter(torch.ones([]) * config.model.contrastive.temperature)
        self.sim_projector = nn.Linear(config.model.contrastive.attn_feature_dim, 1)

    def forward(self, image, context, caption):
        # Extract features from the image, context, and caption
        img_features = self.image_encoder(image.float())
        ctx_features = self.context_encoder(context)
        cap_features = self.caption_encoder(caption)
        
        # Project the features to the attention feature dimension
        img_features = self.image_projector(img_features)
        ctx_features = self.context_projector(ctx_features)
        cap_features = self.caption_projector(cap_features)

        # Compute cross-attention between pairs
        sr_img_ctx = self.cross_attention1(img_features, ctx_features)
        sr_img_cap = self.cross_attention2(img_features, cap_features)
        sr_ctx_cap = self.cross_attention3(ctx_features, cap_features)

        # Project to similarity scores
        sim_img_ctx = self.sim_projector(sr_img_ctx).squeeze(-1)
        sim_img_cap = self.sim_projector(sr_img_cap).squeeze(-1)
        sim_ctx_cap = self.sim_projector(sr_ctx_cap).squeeze(-1)

        return sim_img_ctx, sim_img_cap, sim_ctx_cap

    def compute_loss(self, sim_img_ctx, sim_img_cap, sim_ctx_cap):
        loss_img_ctx = self.InfoNCE_loss(sim_img_ctx)
        loss_img_cap = self.InfoNCE_loss(sim_img_cap)
        loss_ctx_cap = self.InfoNCE_loss(sim_ctx_cap)

        weight = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32).to(sim_img_ctx.device)
        
        total_loss = loss_img_ctx * weight[0] + loss_img_cap * weight[1] + loss_ctx_cap * weight[2]
        return total_loss
    
    def InfoNCE_loss(self, sim_matrix, temperature=0.07):
        sim_matrix = sim_matrix / temperature
        
        exp_sim_ii = torch.exp(torch.diag(sim_matrix))
        exp_sim_ij = torch.exp(sim_matrix).sum(dim=1)
        
        loss = -torch.log(exp_sim_ii / exp_sim_ij).mean()
        return loss
