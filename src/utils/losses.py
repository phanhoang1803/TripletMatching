import torch
import torch.nn.functional as F

def contrastive_loss_fn(sim_img_ctx, sim_img_cap, sim_ctx_cap):
    batch_size = sim_img_ctx.size(0)
    labels = torch.arange(batch_size).to(sim_img_ctx.device)
    
    loss_img_ctx = F.cross_entropy(sim_img_ctx, labels) + F.cross_entropy(sim_img_ctx.t(), labels)
    loss_img_cap = F.cross_entropy(sim_img_cap, labels) + F.cross_entropy(sim_img_cap.t(), labels)
    loss_ctx_cap = F.cross_entropy(sim_ctx_cap, labels) + F.cross_entropy(sim_ctx_cap.t(), labels)

    weight = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32).to(sim_img_ctx.device)
    
    total_loss = loss_img_ctx * weight[0] + loss_img_cap * weight[1] + loss_ctx_cap * weight[2]
    return total_loss
