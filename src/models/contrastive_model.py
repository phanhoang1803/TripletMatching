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
        self.attn_dim = config.model.contrastive.attn_feature_dim
        
        self.image_encoder = ImageEncoder(config)
        self.context_encoder = ContextEncoder(config)
        self.caption_encoder = CaptionEncoder(config)
        
        self.image_projector = nn.Linear(config.model.image.embed_dim, self.attn_dim)
        self.context_projector = nn.Linear(config.model.context.embed_dim, self.attn_dim)
        self.caption_projector = nn.Linear(config.model.caption.embed_dim, self.attn_dim)
        
        self.temperature = nn.Parameter(torch.ones([]) * config.model.contrastive.temperature)

    def forward(self, image, context, caption):
        # Extract features from the image, context, and caption
        img_features = self.image_encoder(image.float())
        ctx_features = self.context_encoder(context)
        cap_features = self.caption_encoder(caption)
        
        # Project the features to the attention feature dimension
        img_features = self.image_projector(img_features)   # (batch_size, attn_dim)
        ctx_features = self.context_projector(ctx_features) # (batch_size, attn_dim)
        cap_features = self.caption_projector(cap_features) # (batch_size, attn_dim)

        sim_img_ctx = self.compute_similarity(img_features, ctx_features) # (batch_size, batch_size)        
        sim_ctx_img = self.compute_similarity(ctx_features, img_features) # (batch_size, batch_size)
        sim_img_cap = self.compute_similarity(img_features, cap_features) # (batch_size, batch_size)
        sim_cap_img = self.compute_similarity(cap_features, img_features) # (batch_size, batch_size)
        sim_ctx_cap = self.compute_similarity(ctx_features, cap_features) # (batch_size, batch_size)
        sim_cap_ctx = self.compute_similarity(cap_features, ctx_features) # (batch_size, batch_size)

        return sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx
        

    def compute_similarity(self, sim1, sim2):
        sim = torch.matmul(sim1, sim2.T)
        return sim

    def compute_loss(self, sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx):
        loss_img_ctx = self.InfoNCE_loss(sim_img_ctx)
        loss_ctx_img = self.InfoNCE_loss(sim_ctx_img)
        loss_img_cap = self.InfoNCE_loss(sim_img_cap)
        loss_cap_img = self.InfoNCE_loss(sim_cap_img)
        loss_ctx_cap = self.InfoNCE_loss(sim_ctx_cap)
        loss_cap_ctx = self.InfoNCE_loss(sim_cap_ctx)

        weight = torch.tensor([0.25, 0.25, 0.5, 0.5, 0.25, 0.25], dtype=torch.float32, device=sim_img_ctx.device)

        total_loss = loss_img_ctx * weight[0] + loss_ctx_img * weight[1] + loss_img_cap * weight[2] + loss_cap_img * weight[3] + loss_ctx_cap * weight[4] + loss_cap_ctx * weight[5]
        
        return total_loss
    
    def InfoNCE_loss(self, sim_matrix):
        sim_matrix = sim_matrix / self.temperature
        
        # Create labels for the diagonal elements, 1 for positive pairs and 0 for negative pairs
        labels = torch.eye(sim_matrix.size(0), dtype=torch.float32, device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
