import torch
import torch.nn as nn
import torch.functional as F

from src.datasets import TextImageDataset
from test import test_model
from utils import (
    get_optimizer, get_scheduler, get_model, seed_everything, 
    EarlyStopping
)

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import CLIPProcessor, CLIPModel

from PIL import Image

import logging
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import os

import random


logger = logging.getLogger('Train')


def train(cfg):
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./optimizers", exist_ok=True)
    tb_logger = SummaryWriter("./tb_logs")
    step = 0
    
    model = get_model(cfg.model).to(cfg.train.device)
    processor = CLIPProcessor.from_pretrained(cfg.model.name)
    
    metafile = OmegaConf.load(
        os.path.join(cfg.dataset.dataset_path, 'metafile.yaml')
    )
    with open('./query.txt', 'w') as f:
        f.write(metafile['question'])
    
    train_dataset = TextImageDataset(
        cfg.dataset.dataset_path,
        metafile['train_samples'],
        processor,
        cfg.dataset.permute_colors,
        cfg.dataset.to_full_sentence,
        cfg.dataset.img_size
    )
    test_dataset = TextImageDataset(
        cfg.dataset.dataset_path,
        metafile['test_samples'],
        processor,
        False,
        True,
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
        shuffle=False,
        num_workers=cfg.train.num_workers
    )
    
    del metafile
    
    optimizer = get_optimizer(cfg.optimizer, model)
    scheduler = get_scheduler(cfg.scheduler, optimizer) if cfg.scheduler.name is not None else None
    es = EarlyStopping(**cfg.early_stopping) if cfg.early_stopping.patience is not None else None
    
    # Test model before training if not prompt learning
    if not cfg.model.prompt_learning:
        logger.info('Test model before training')
        results = test_model(test_loader, model, cfg.train.device)
    
    for epoch in range(1, cfg.train.epochs + 1):
        logger.info(f'Epoch {epoch} / {cfg.train.epochs}')
        
        # Train
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f'Training ...')
        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].to(cfg.train.device)
            
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            losses.append(loss.item())
            tb_logger.add_scalar('Train/loss', loss.item(), step)
            step += 1
            
        logger.info(f'Epoch {epoch} / {cfg.train.epochs} - Loss: {sum(losses) / len(losses)}')
        tb_logger.add_scalar('Train/epoch_loss', sum(losses) / len(losses), epoch)
        
        if scheduler is not None:
            scheduler.step()
            
        # Test
        model.eval()
        results = test_model(test_loader, model, cfg.train.device)
        for k, v in results.items():
            tb_logger.add_scalar(f'Test/{k}', v, epoch)
        
        # Early Stopping
        if es is not None:
            es(results, model)
            if es.is_stop:
                logger.info('Early Stopping at Epoch {}'.format(epoch))
                break
        
        # Save model
        torch.save(model.state_dict(), f'./models/{epoch}.pt')
        torch.save(optimizer.state_dict(), f'./optimizers/{epoch}.pt')
    
    # if es is not None:
    #     model.load_state_dict(es.best_state_dict_model)
    
    logger.info('Training finished')
    
    
@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.train.seed)
    train(cfg)
    
    
if __name__ == '__main__':
    main()
