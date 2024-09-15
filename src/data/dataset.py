import random
import requests
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os
import json
from urllib.parse import urlparse
from torch.nn.utils.rnn import pad_sequence

class TripletMatchingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()
        self.tokenizer = None  # Assuming tokenizer is initialized here

    def load_data(self):
        with open(os.path.join(self.data_dir, 'splitted_annotations.json'), 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item['img_local_path'])
        
        if not os.path.exists(image_path):
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return (
            image,
            torch.tensor(self.tokenizer.encode(item['context'], truncation=True, max_length=512)),
            torch.tensor(self.tokenizer.encode(item['caption'], truncation=True, max_length=128)),
            torch.tensor(item['label'], dtype=torch.long)
        )

class BBCNewsDataset(Dataset):
    """
    {'title': 'BBC to make Watchdog part of The One Show - BBC News',
 'published_date': '2020-02-21',
 'authors': ['https://www.facebook.com/bbcnews'],
 'description': 'The long-running consumer rights show will no longer be a standalone programme.',
 'section': 'Entertainment & Arts',
 'content': 'Matt Allwright, Nikki Fox and Steph McGovern have hosted Watchdog from Salford since 2017\n\nThe BBC\'s long-running consumer rights series Watchdog is to end as a standalone programme, instead becoming part of The One Show.\n\nWatchdog began in 1980 as a strand of Nationwide, but proved so popular it became a separate programme in 1985.\n\nCo-host Steph McGovern has moved to Channel 4, but Matt Allwright and Nikki Fox will stay to front the new strand.\n\nThe BBC said they would investigate viewer complaints all year round rather than for two series a year.\n\nAlice Beer, Jonathan Maitland, Anne Robinson and Chris Choi on Watchdog in 1995\n\nAlison Kirkham, controller of BBC Factual, said Allwright and Fox would "continue to be the viewers\' trusted guide", and would "achieve even greater prominence and success" in raising awareness of consumer rights.\n\nThe One Show editor Rob Unsworth said the change would mean that "more than ever the team can react on behalf of consumers whenever stories come up".\n\nHusband and wife John Stapleton and Lynn Faulds Wood are among the past presenters\n\nMore recently, it has been on air for 12 episodes per year. The 42nd and most recent series attracted an average of 3.1 million viewers per episode. Watchdog will be incorporated into The One Show this spring.\n\nThe news comes weeks after the corporation announced that Victoria Derbyshire\'s investigative BBC Two programme will end as a result of cuts to BBC News.\n\nFollow us on Facebook, or on Twitter @BBCNewsEnts. If you have a story suggestion email entertainment.news@bbc.co.uk.',
 'link': 'http://www.bbc.co.uk/news/entertainment-arts-51584666',
 'top_image': 'https://ichef.bbci.co.uk/news/976/cpsprodpb/104B2/production/_110983766_watchdogbbc_cut.jpg'}
    
    """
    def __init__(self, data, transform=None, context_tokenizer=None, caption_tokenizer=None, max_length=512):
        self.data = data
        self.transform = transform
        self.context_tokenizer = context_tokenizer
        self.caption_tokenizer = caption_tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data[idx]
        image_url = item['top_image']
        caption = item['title']
        context = item['description']

        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)

        # Ensure filename has an extension
        if not os.path.splitext(filename)[1]:
            filename += '.jpg'  # Default to .jpg if no extension

        save_dir = 'downloaded_images'
        os.makedirs(save_dir, exist_ok=True)

        local_image_path = os.path.join(save_dir, filename)

        # Download the image and save for later use
        if not os.path.exists(local_image_path):
            # print(f"Downloading image from {image_url}")
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(local_image_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download image from {image_url}")

        # Try to open the image, create a blank one if it fails
        try:
            image = Image.open(local_image_path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            print(f"Failed to open image: {local_image_path}. Creating a blank image.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # Create a blank image
            
        if self.transform:
            image = self.transform(image)
            
        caption_tensor = torch.tensor(self.caption_tokenizer.encode(caption, truncation=True, max_length=self.max_length))
        context_tensor = torch.tensor(self.context_tokenizer.encode(context, truncation=True, max_length=self.max_length))

        return (image, caption_tensor, context_tensor, torch.tensor(random.randint(0, 1), dtype=torch.long))

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    images, captions, contexts, labels = zip(*batch)
    
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    
    return images, captions, contexts, labels

