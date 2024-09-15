import torch
import torch.nn.functional as F

def contrastive_loss(outputs, labels, temperature=0.07):
    sim_img_ctx, sim_img_cap, sim_ctx_cap = outputs
    
    loss_img_ctx = F.cross_entropy(sim_img_ctx, labels)
    loss_img_cap = F.cross_entropy(sim_img_cap, labels)
    loss_ctx_cap = F.cross_entropy(sim_ctx_cap, labels)

    loss = loss_img_ctx + loss_img_cap + loss_ctx_cap
    return loss

def InfoNCE(outputs, labels, temperature=0.07):
    sim_img_ctx, sim_img_cap, sim_ctx_cap = outputs
    sim_matrix = torch.matmul(sim_img_ctx, sim_img_ctx.t()) / temperature
    exp_sim_matrix = torch.exp(sim_matrix)
    
    positive_pairs = torch.eye(sim_img_ctx.size(0), device=sim_img_ctx.device)
    negative_pairs = 1 - positive_pairs
    
    pos_loss = -torch.log(exp_sim_matrix[positive_pairs.bool()] / exp_sim_matrix.sum(dim=1))
    neg_loss = -torch.log(1 - exp_sim_matrix[negative_pairs.bool()] / exp_sim_matrix.sum(dim=1))
    
    loss = (pos_loss.sum() + neg_loss.sum()) / (2 * sim_img_ctx.size(0))
    return loss

def accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    return correct / total