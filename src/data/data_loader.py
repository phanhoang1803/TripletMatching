import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import AugmentedContextDataset, TripletMatchingDataset, BBCNewsDataset
import datasets
from transformers import AutoTokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast
from ..utils.utils import collate_fn

def get_data_loaders(config):
    if config.data.dataset_name == 'RealTimeData/bbc_news_alltime':
        return get_bbc_data_loaders(config)
    elif config.data.dataset_name == 'augmented_context':
        return get_augmented_context_data_loaders(config)
    else:
        raise ValueError(f"Dataset {config.data.dataset_name} not found")
    
def get_bbc_data_loaders(config):
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

    context_tokenizer = get_context_tokenizer(config)
    caption_tokenizer = get_caption_tokenizer(config)

    data = datasets.load_dataset('RealTimeData/bbc_news_alltime', '2020-02')
    
    if config.debug == True:
        data = data['train'].select([i for i in range(config.data.num_debug_samples)])
    else:
        data = data['train']
    
    dataset = BBCNewsDataset(data, transform=transform, context_tokenizer=context_tokenizer, caption_tokenizer=caption_tokenizer, max_length=512)

    train_size = int(config.data.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, collate_fn=collate_fn, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers, collate_fn=collate_fn, prefetch_factor=2)

    config.model.context.vocab_size = len(context_tokenizer)
    config.model.caption.vocab_size = len(caption_tokenizer)
    
    return train_loader, val_loader

def get_augmented_context_data_loaders(config):
    with open(os.path.join(config.args.data_dir, config.data.json_file), 'r') as f:
        data = json.load(f)
    
    transform = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    context_tokenizer = get_context_tokenizer(config)
    caption_tokenizer = get_caption_tokenizer(config)

    if config.debug == True:
        data = data[:config.data.num_debug_samples]
    
    train_dataset = AugmentedContextDataset(data=data, 
                                            transform=transform, 
                                            context_tokenizer=context_tokenizer, 
                                            caption_tokenizer=caption_tokenizer,
                                            config=config)
    
    train_size = int(config.data.train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, collate_fn=collate_fn, prefetch_factor=config.train.prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers, collate_fn=collate_fn, prefetch_factor=config.train.prefetch_factor)
    
    return train_loader, val_loader
    
def get_caption_tokenizer(config):
    if config.model.caption.tokenizer == 'roberta_tokenizer':
        return RobertaTokenizer.from_pretrained('roberta-base')
    elif config.model.caption.tokenizer == 'roberta_tokenizer_fast':
        return RobertaTokenizerFast.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Tokenizer {config.model.caption.tokenizer} not found")

def get_context_tokenizer(config):
    if config.model.context.tokenizer == 'roberta_tokenizer':
        return RobertaTokenizer.from_pretrained('roberta-base')
    elif config.model.context.tokenizer == 'roberta_tokenizer_fast':
        return RobertaTokenizerFast.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Tokenizer {config.model.context.tokenizer} not found")

