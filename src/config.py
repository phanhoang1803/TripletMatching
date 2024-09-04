import torch

class Config:
    # Data
    data_dir = 'F:/Thesis/triplet_matching/data'
    
    # Model
    output_dim = 768
    
    # Training
    batch_size = 16  # Reduced due to larger model size
    num_epochs = 10
    learning_rate = 5e-5  # Adjusted for transformer fine-tuning
    warmup_steps = 1000
    
    # Loss
    triplet_margin = 1.0
    
    # Misc
    seed = 42
    num_workers = 4
    
    # Device
    device = 'cpu'