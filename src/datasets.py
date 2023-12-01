import torch
import torch.nn as nn
import torch.functional as F

from torch.utils.data import Dataset

from PIL import Image
import random

import logging

import os


logger = logging.getLogger('Datasets')

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'grey']

class TextImageDataset(Dataset):
    def __init__(
        self, path: str, metadata: dict, processor,
        permute_colors: bool, img_size: int = 224
    ):
        super().__init__()
        self.path = path
        self.img_path = os.path.join(path, 'images')
        self.metadata = metadata
        self.processor = processor
        self.img_size = img_size
        self.permute_colors = permute_colors
        
        logger.info(f'Found {len(self)} samples')
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        img_path = self.metadata[idx]['img_dir']
        text = self.metadata[idx]['answer']
        img = Image.open(
            os.path.join(self.img_path, img_path)
        ).convert("RGB").resize((self.img_size, self.img_size))
        
        if self.permute_colors:
            # Check color words in text
            is_contain_color = all([color in text.split(' ') for color in colors])
            
            # If text contains color words, Change color words randomly to other colors
            if is_contain_color:
                text = self.permute_colors_in_text(text)
        
        sample = self.processor(text=text, images=img, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {
            'input_ids': sample['input_ids'][0],
            'attention_mask': sample['attention_mask'][0],
            'pixel_values': sample['pixel_values'][0],
            'labels': torch.tensor([idx])
        }
        
    def permute_colors_in_text(self, text: str) -> str:
        # Change color words randomly to other colors
        random_color = random.choice(colors)
        
        text = text.split(' ')
        for c in colors:
            if c in text:
                text[text.index(c)] = random_color
        
        text = ' '.join(text)
        return text
