import torch
import torch.nn as nn
import torch.functional as F

from torch.utils.data import Dataset

from CLIP.clip import tokenize

from PIL import Image
import random

import logging

import os


logger = logging.getLogger('Datasets')

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'grey']

class TextImageDataset(Dataset):
    def __init__(
        self, path: str, metadata: dict, processor,
        permute_colors: bool, to_full_sentence: bool, img_size: int = 224
    ):
        super().__init__()
        self.path = path
        self.img_path = os.path.join(path, 'images')
        self.metadata = metadata
        self.processor = processor
        self.img_size = img_size
        self.permute_colors = permute_colors
        self.to_full_sentence = to_full_sentence
        
        logger.info(f'Found {len(self)} samples')
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        img_path = self.metadata[idx]['img_dir']
        text = self.metadata[idx]['answer']
        img = Image.open(
            os.path.join(self.img_path, img_path)
        ).convert("RGB").resize((self.img_size, self.img_size))
        
        img_token = self.processor(img)
        
        if self.permute_colors:
            # Check color words in text
            is_contain_color = all([color in text.split(' ') for color in colors])
            
            # If text contains color words, Change color words randomly to other colors
            if is_contain_color:
                text = self.permute_colors_in_text(text)
        
        if self.to_full_sentence:
            text = self.to_sentence(text)
            
        text = text.lower()  # CLIP trained on lower-cased text
        
        text_token = tokenize(text, truncate=True)[0]
        return {
            'input_ids': text_token,
            'pixel_values': img_token,
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

    def to_sentence(self, text: str) -> str:
        text_list = text.split(' ')
        if len(text_list) == 1:
            return "The action is " + text + "."
        else:
            return text
        