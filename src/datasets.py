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
pronouns = [
    'he', 'she', 'they', 'it',
    'He', 'She', 'They', 'It',
]


class TextImageDataset(Dataset):
    def __init__(
        self, path: str, metadata: dict, processor,
        permute_colors: bool, permute_pronouns: bool, img_size: int = 224
    ):
        super().__init__()
        self.path = path
        self.img_path = os.path.join(path, 'images')
        self.metadata = metadata
        self.processor = processor
        self.img_size = img_size
        self.permute_colors = permute_colors
        self.permute_pronouns = permute_pronouns
        
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
            is_contain_color = all([color in text for color in colors])
            
            # If text contains color words, Change color words randomly to other colors
            if is_contain_color:
                text = self.permute_colors_in_text(text)
            
        if self.permute_pronouns:
            # Check pronouns in text
            is_contain_pronoun = any([pronoun in text for pronoun in pronouns])
            
            # If text contains pronouns, Change pronouns randomly to other pronouns
            if is_contain_pronoun:
                text = self.permute_pronouns_in_text(text)
        
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
        
        for c in colors:
            text = text.replace(c, random_color)
            
        return text
    
    def permute_pronouns_in_text(self, text: str) -> str:
        # Change pronouns randomly to other pronouns
        random_pronoun = random.choice(pronouns)
        
        for p in pronouns:
            text = text.replace(p, random_pronoun)
            
        return text
