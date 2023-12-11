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

from typing import Optional


logger = logging.getLogger('Train')


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return -nn.LogSoftmax(dim=-1)(logits).diag().mean()

def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    img_loss = contrastive_loss(logits_per_image)
    text_loss = contrastive_loss(logits_per_text)
    return (img_loss + text_loss) / 2


def geodesic_mix(lambda_mix, image_feat, text_feat):
    theta = torch.acos((image_feat * text_feat).sum(dim=[1])).view(image_feat.shape[0], 1)
    n1 = torch.sin(lambda_mix * theta) / torch.sin(theta) * image_feat
    n2 = torch.sin((1 - lambda_mix) * theta) / torch.sin(theta) * text_feat
    return n1 + n2


def train(cfg):
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./optimizers", exist_ok=True)
    tb_logger = SummaryWriter("./tb_logs")
    step = 0
    
    model, processor = get_model(cfg.model)
    model.to(cfg.train.device)
    
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
        num_workers=cfg.train.num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=False
    )
    
    del metafile
    
    optimizer = get_optimizer(cfg.optimizer, model)
    scheduler = get_scheduler(cfg.scheduler, optimizer) if cfg.scheduler.name is not None else None
    es = EarlyStopping(**cfg.early_stopping) if cfg.early_stopping.patience is not None else None
    
    # Test model before training if not prompt learning. 
    if not cfg.model.prompt_learning:
        logger.info('Test model before training')
        results = test_model(test_loader, model, cfg.train.device)
    
    for epoch in range(1, cfg.train.epochs + 1):
        logger.info(f'Epoch {epoch} / {cfg.train.epochs}')
        
        # Train
        model.train()
        if hasattr(model, 'q_former'):
            model.q_former.train()
        losses = []
        pbar = tqdm(train_loader, desc=f'Training ...')
        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].to(cfg.train.device)
            
            image_feat = model.encode_image(batch['pixel_values'])
            text_feat = model.encode_text(batch['input_ids'])
            
            image_feat = image_feat / image_feat.norm(p=2, dim=1, keepdim=True)
            text_feat = text_feat / text_feat.norm(p=2, dim=1, keepdim=True)
            
            t1 = model.logit_scale.exp()
            logits_per_image = t1 * image_feat @ text_feat.t()
            logits_per_text = logits_per_image.t()

            loss = clip_loss(logits_per_image, logits_per_text)
            
            if cfg.model.m2_loss_weight is not None:
                I = torch.eye(image_feat.shape[0], device=image_feat.device)
                t2 = model.logit_scale_2.exp()
                
                lambda_mix = random.betavariate(2., 2.,)
                
                mix = geodesic_mix(lambda_mix, image_feat, text_feat)
                
                logits2_i = mix @ text_feat.t()
                logits2_i = logits_per_image * I + logits2_i * (1 - I)
                
                logits2_t = mix @ image_feat.t()
                logits2_t = logits_per_text * I + logits2_t * (1 - I)
                
                loss += cfg.model.m2_loss_weight * clip_loss(logits2_i * t2, logits2_t * t2)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            losses.append(loss.item())
            pbar.set_postfix({'loss': sum(losses[:100]) / len(losses[:100])})
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
