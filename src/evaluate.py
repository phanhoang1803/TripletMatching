import torch
from models.triplet_matching_model import TripletMatchingModel
from data.data_loader import get_data_loaders
from utils.metrics import compute_metrics
from tqdm import tqdm

def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TripletMatchingModel().to(device)
    model.load_state_dict(torch.load('triplet_matching_model.pth'))
    model.eval()

    _, _, test_loader = get_data_loaders(config.data_dir, config.batch_size)

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, captions, contexts, labels = [b.to(device) for b in batch.values()]
            outputs = model(images, captions, contexts)
            
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_outputs, all_labels)
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics

if __name__ == '__main__':
    from config import Config
    evaluate(Config)