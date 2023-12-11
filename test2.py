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

from test import calc_recall_at_k_fixed

import numpy as np

import matplotlib.pyplot as plt


if __name__ == '__main__':
    save_path = './train_logs/RN50/2023-12-06_14-36-08/'
    cfg = OmegaConf.load(os.path.join(save_path, '.hydra/config.yaml'))
    model, processor = get_model(cfg.model)
    model.load_state_dict(torch.load(os.path.join(save_path, 'models/3.pt')))
    model.eval()
    if hasattr(model, 'q_former'):
        model.q_former.eval()
    
    metafile = OmegaConf.load('./hpitp_dataset/metafile.yaml')
    test_dataset = TextImageDataset(
        './hpitp_dataset/',
        metafile['test_samples'],
        processor,
        False,
        True,
        224
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    
    del metafile
    
    device='cuda:0'
    model.to(device)
    model.q_former.to(device)
    
    with torch.inference_mode():
        img_embeddings = []
        text_embeddings = []

        # Fisrt get all image/text embeddings
        pbar = tqdm(test_loader, desc='Testing ...')
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            img_embedding = model.encode_image(batch['pixel_values'])
            text_embedding = model.encode_text(batch['input_ids'])

            img_embeddings.append(img_embedding.cpu())
            text_embeddings.append(text_embedding.cpu())
        
        img_embeddings = torch.cat(img_embeddings, dim=0).to(device)
        text_embeddings = torch.cat(text_embeddings, dim=0).to(device)
        
        t1 = model.logit_scale.exp()
        probs_text_to_img = (text_embeddings @ img_embeddings.t()) * t1
    
    recall_at_1 = calc_recall_at_k_fixed(probs_text_to_img, 1)
    recall_at_5 = calc_recall_at_k_fixed(probs_text_to_img, 5)
    recall_at_20 = calc_recall_at_k_fixed(probs_text_to_img, 20)
    recall_at_100 = calc_recall_at_k_fixed(probs_text_to_img, 100)
    
    results = {
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'recall_at_20': recall_at_20,
        'recall_at_100': recall_at_100
    }
    
    print(results)
    
    probs_text_to_img_np = probs_text_to_img.cpu().numpy()
    probs_text_to_img_softmax_np = probs_text_to_img.softmax(dim=-1).cpu().numpy()
    
    split_size = 500
    n_splits = len(probs_text_to_img_np) // split_size
    
    os.makedirs('./similarity_matrix/', exist_ok=True)
    os.makedirs('./similarity_matrix_softmax/', exist_ok=True)
    
    for i in np.arange(n_splits):
        probs_sim = probs_text_to_img_np[i*split_size:(i+1)*split_size, i*split_size:(i+1)*split_size]
        probs_sim_softmax = probs_text_to_img_softmax_np[i*split_size:(i+1)*split_size, i*split_size:(i+1)*split_size]
        
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(probs_sim)
        ax.set_title('Similarity matrix', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./similarity_matrix/{i}.png')
        
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(probs_sim_softmax)
        ax.set_title('Similarity matrix (softmax)', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./similarity_matrix_softmax/{i}.png')
        
    
    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # ax.imshow(probs_text_to_img_np)
    # ax.set_title('Similarity matrix', fontsize=20)
    # plt.tight_layout()
    # plt.savefig('./similarity_matrix.png')
    
    # plt.clf()
    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # ax.imshow(probs_text_to_img_softmax_np)
    # ax.set_title('Similarity matrix (softmax)', fontsize=20)
    # plt.tight_layout()
    # plt.savefig('./similarity_matrix_softmax.png')