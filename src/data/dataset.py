import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class TripletMatchingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_dir, 'splitted_annotations.json'), 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item['img_local_path'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'caption': item['caption'],
            'context': item['context'],
            'label': item['label']  # 0 for fake, 1 for real
        }