import torch
import torch.nn as nn
import torch.functional as F

from src.datasets import TextImageDataset
from test import test_model
from utils import get_optimizer, seed_everything, EarlyStopping

from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel

from PIL import Image

import logging
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import os


logger = logging.getLogger('Train')


def _main(cfg):
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./optimizers", exist_ok=True)
    
    model = CLIPModel.from_pretrained(cfg.model.name)
    processor = CLIPProcessor.from_pretrained(cfg.model.name)
    
    metafile = OmegaConf.load(cfg.dataset.metafile_path)
    n_samples = len(metafile)
    train_size = int(n_samples * cfg.dataset.train_ratio)
    
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
    
    optimizer = get_optimizer(cfg.optimizer, model)
    
    es = EarlyStopping(**cfg.early_stopping)
    
    for epoch in range(1, cfg.train.epochs + 1):
        logger.info(f'Epoch {epoch} / {cfg.train.epochs}')
        
        # Train
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f'Training ...')
        for batch in pbar:
            optimizer.zero_grad()
            
            del batch['labels']

            for k in batch:
                batch[k] = batch[k].to(cfg.train.device)
            
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        logger.info(f'Epoch {epoch} / {cfg.train.epochs} - Loss: {sum(losses) / len(losses)}')
            
        # Test
        model.eval()
        results = test_model(test_loader, model, cfg.train.device)
        
        # Early Stopping
        es(results, model)
        
        if es.early_stop:
            logger.info('Early Stopping at Epoch {}'.format(epoch))
            break
        
        # Save model
        torch.save(model.state_dict(), f'./models/{epoch}.pt')
        torch.save(optimizer.state_dict(), f'./optimizers/{epoch}.pt')
        
    logger.info('Training finished')
    
    
@hydra.main(config_path='config', config_name='main')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    _main(cfg)
    
    
if __name__ == '__main__':
    main()
