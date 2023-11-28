import torch
import torch.nn as nn
import torch.functional as F

from src.datasets import TextImageDataset
from test import test_model

from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel

from PIL import Image

import logging

import hydra
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger('Train')


def calculate_loss(logits):
    n = logits.shape[0]
    labels = torch.arange(n).to(logits.device)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_j) / 2


def _main(cfg):
    model = CLIPModel.from_pretrained(cfg.model.name)
    processor = CLIPProcessor.from_pretrained(cfg.model.name)
    
    metafile = OmegaConf.load(cfg.dataset.metafile_path)
    n_samples = len(metafile)
    train_size = int(n_samples * cfg.dataset.train_ratio)
    test_size = n_samples - train_size
    
    train_dataset = TextImageDataset(
        metafile[:train_size],
        processor,
        cfg.dataset.permute_colors,
        cfg.dataset.permute_pronouns,
        cfg.dataset.img_size
    )
    test_dataset = TextImageDataset(
        metafile[train_size:],
        processor,
        cfg.dataset.permute_colors,
        cfg.dataset.permute_pronouns,
        cfg.dataset.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = calculate_loss(outputs.logits)
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                loss = calculate_loss(outputs.logits)
        
        logger.info(f'Epoch {epoch}: Loss = {loss.item():.4f}')
        
    torch.save(model.state_dict(), cfg.train.save_path)
    