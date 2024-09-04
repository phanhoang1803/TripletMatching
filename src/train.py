import torch
from tqdm import tqdm
from models.triplet_matching_model import TripletMatchingModel
from utils.loss_functions import TripletMatchingLoss
from data.data_loader import get_data_loaders
from utils.metrics import compute_metrics
from transformers import get_linear_schedule_with_warmup

def train(config):
    device = torch.device(config.device)
    model = TripletMatchingModel(config.output_dim).to(device)
    criterion = TripletMatchingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    train_loader, val_loader, _ = get_data_loaders(config.data_dir, config.batch_size)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            images, captions, contexts, labels = [b.to(device) for b in batch.values()]
            
            optimizer.zero_grad()
            outputs = model(images, captions, contexts)
            loss = criterion(outputs, labels, model.get_features())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        print(f"Validation Metrics: {val_metrics}")

    torch.save(model.state_dict(), 'triplet_matching_model.pth')

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images, captions, contexts, labels = [b.to(device) for b in batch.values()]
            outputs = model(images, captions, contexts)
            loss = criterion(outputs, labels, model.get_features())
            total_loss += loss.item()
            
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_outputs, all_labels)
    return total_loss / len(data_loader), metrics

if __name__ == '__main__':
    from config import Config
    train(Config)