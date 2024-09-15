import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import TripletMatchingDataset, BBCNewsDataset
import datasets
from transformers import AutoTokenizer
from .dataset import collate_fn

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = TripletMatchingDataset(os.path.join(config.data.data_dir, 'train'), transform=transform)
    
    # Split the train dataset into train and validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    test_dataset = TripletMatchingDataset(os.path.join(config.data.data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)

    return train_loader, val_loader, test_loader

def get_data_loaders_bbc(config):
    """
    Get the data loaders for the BBC News dataset.
    
    Args:
        config: The configuration object.
    
    Returns:
        train_loader: The data loader for the training dataset.
        val_loader: The data loader for the validation dataset.
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    context_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    caption_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    data = datasets.load_dataset('RealTimeData/bbc_news_alltime', '2020-02')
    
    if config.debug == True:
        data = data['train'].select([i for i in range(200)])
    else:
        data = data['train']
    
    dataset = BBCNewsDataset(data, transform=transform, context_tokenizer=context_tokenizer, caption_tokenizer=caption_tokenizer, max_length=512)


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, collate_fn=collate_fn, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers, collate_fn=collate_fn, prefetch_factor=2)

    config.model.context.vocab_size = len(context_tokenizer)
    config.model.caption.vocab_size = len(caption_tokenizer)
    
    return train_loader, val_loader

