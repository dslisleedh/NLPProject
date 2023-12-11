import torch
import torch.nn as nn
import torch.nn.functional as F

from src.visualization import GradCAM
from src.model import get_model

from CLIP.clip import tokenize

from PIL import Image
import numpy as np

import logging
import os
from omegaconf import DictConfig, OmegaConf

import cv2


logger = logging.getLogger('Inference')


def inference(cfg):
    model, processor = get_model(cfg.model)
    model.load_state_dict(torch.load(cfg.inference.model_path))
    
    model.eval()
    if hasattr(model, 'q_former'):
        model.q_former.eval()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    
    model.to(device)
    if hasattr(model, 'q_former'):
        model.q_former.to(device)
    
    image_dir = cfg.inference.image_dir
    image_paths = [os.path.join(image_dir, path) for path in cfg.inference.image_paths]
    imgs = [Image.open(path).convert('RGB').resize((224, 224)) for path in image_paths]
    
    imgs_tensor = torch.stack([processor(img) for img in imgs])
    
    text = cfg.inference.text
    text = text.lower()
    text_tensor = tokenize([text]).to(device)

    with GradCAM(model) as hook:
        if len(imgs_tensor.shape) == 3:
            imgs_tensor = imgs_tensor.unsqueeze(0)
            
        imgs_tensor = imgs_tensor.to(device)
        text_tensor = text_tensor.to(device)
        
        imgs_tensor = imgs_tensor.requires_grad_(True)
        
        embedding_imgs = model.encode_image(imgs_tensor)
        embedding_text = model.encode_text(text_tensor)
    
        embedding_imgs = embedding_imgs / embedding_imgs.norm(p=2, dim=-1, keepdim=True)
        embedding_text = embedding_text / embedding_text.norm(p=2, dim=-1, keepdim=True)
        
        similarity = embedding_text @ embedding_imgs.T
        similarity = similarity.diagonal()
        
        print(similarity)
        # GradCAM
        max_idx = similarity.argmax()
        similarity[max_idx].backward()
        
        act = hook.get_activation_maps[max_idx.squeeze(), ...]
        grad = hook.get_gradients[max_idx.squeeze(), ...]

    alpha = grad.mean(dim=(1, 2), keepdim=True)
    gradcam = torch.sum(act * alpha, dim=0, keepdim=False)
    gradcam = F.relu(gradcam) / (gradcam.max() + 1e-8)
    gradcam = gradcam.detach().cpu().numpy() * 255
    gradcam = gradcam.astype(np.uint8)
    
    # Upsample to original image size
    best_img = imgs[max_idx.squeeze()]
    w, h = best_img.size
    
    grad_img = cv2.applyColorMap(255 - gradcam, cv2.COLORMAP_JET)
    grad_img = cv2.resize(grad_img, (w, h))
    grad_img = Image.fromarray(grad_img)
    
    # Blend
    alpha = cfg.inference.blend_alpha
    blended = Image.blend(best_img, grad_img, alpha)
    
    # Save
    os.makedirs(cfg.inference.save_path, exist_ok=True)
    with open(os.path.join(cfg.inference.save_path, 'query.txt'), 'w') as f:
        f.write(text + '\n')
        f.write('best image index: ' + str(max_idx.squeeze().item()) + '\n')
        
    blended.save(os.path.join(cfg.inference.save_path, 'blended.png'))
    best_img.save(os.path.join(cfg.inference.save_path, 'best_img.png'))
    grad_img.save(os.path.join(cfg.inference.save_path, 'grad_img.png'))
    
    # same other images
    for i, img in enumerate(imgs):
        if i == max_idx.squeeze().item():
            continue
        img.save(os.path.join(cfg.inference.save_path, f'other_{i}.png'))
        
    
if __name__ == '__main__':
    cfg = OmegaConf.load('./config/inference.yaml')
    inference(cfg)
    
