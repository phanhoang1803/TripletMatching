import torch
from models.contrastive_model import ContrastiveLearningModel
from data.data_loader import get_data_loaders
from utils.metrics import contrastive_loss, accuracy
from config import config

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in dataloader:
            images, contexts, captions, labels = [item.to(device) for item in batch]
            outputs = model(images, contexts, captions)
            loss = contrastive_loss(outputs, labels)
            acc = accuracy(outputs, labels)
            
            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ContrastiveLearningModel(config).to(device)
    # Load the trained model weights here
    model.load_state_dict(torch.load('path/to/trained_model.pth'))
    
    _, test_loader = get_data_loaders(config)
    
    test_loss, test_acc = evaluate(model, test_loader, device)
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main()