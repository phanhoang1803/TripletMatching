import numpy as np
from omegaconf import OmegaConf
import torch
import yaml
from src.models.contrastive_model import ContrastiveLearningModel
from src.data.data_loader import get_data_loaders, get_bbc_data_loaders
from src.utils.metrics import contrastive_loss, accuracy
import argparse
from tqdm import tqdm
import warnings
from src.utils.losses import contrastive_loss_fn

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--config', default='configs/config.yaml', help='config file path')
    
    args = parser.parse_args()
    
    return args

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images, contexts, captions, labels = [item.to(device) for item in batch]
        
        # Check for NaN values in the input data
        if np.isnan(images.cpu().numpy()).any() or np.isnan(contexts.cpu().numpy()).any() or np.isnan(captions.cpu().numpy()).any():
            print("NaN detected in input data!")
            continue  # Skip this batch if NaN is found
        
        optimizer.zero_grad()
        sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx = model(images, contexts, captions)
        loss = model.compute_loss(sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx)

        # Check for NaN in loss
        if np.isnan(loss.item()):
            print("NaN detected in loss!")
            continue  # Skip this batch if NaN is found

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss, 0

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images, contexts, captions, labels = [item.to(device) for item in batch]
            sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx = model(images, contexts, captions)
            loss = model.compute_loss(sim_img_ctx, sim_ctx_img, sim_img_cap, sim_cap_img, sim_ctx_cap, sim_cap_ctx)
            # acc = accuracy(outputs, labels)
            
            total_loss += loss.item()
            # total_acc += acc

    avg_loss = total_loss / len(dataloader)
    return avg_loss, 0

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the data loaders
    if config.data.dataset_name == 'RealTimeData/bbc_news_alltime':
        train_loader, val_loader = get_bbc_data_loaders(config)
    else:
        train_loader, val_loader = get_data_loaders(config)
    
    # Initialize the model, optimizer, and other training parameters
    model = ContrastiveLearningModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    for epoch in range(config.train.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{config.train.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Add checkpointing logic here
        

if __name__ == '__main__':
    main()