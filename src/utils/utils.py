import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    images, captions, contexts, labels = zip(*batch)
    
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    
    return images, captions, contexts, labels