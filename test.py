import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel
from src.datasets import TextImageDataset

import torchmetrics as tm

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm


logger = logging.getLogger('Test')


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    
    from https://github.com/tjddus9597/HIER-CVPR23/tree/mast
    HIER(CVPR2023)
    """
    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def calc_recall_at_k_fixed(Y: torch.Tensor, k: int):
    """
    Y : [nb_samples x k] (k predicted labels/neighbours)
    
    Fixed version of calc_recall_at_k
    """
    Y = Y.topk(k, dim=-1).indices
    labels = torch.arange(Y.shape[0]).unsqueeze(-1).to(Y.device)
    denom = Y.shape[0]
    num = Y.eq(labels).any(dim=-1).sum().item()
    return num / denom


def test_model(test_loader: torch.utils.data.DataLoader, model: nn.Module, device: str):
    model.eval()
    if hasattr(model, 'q_former'):
        model.q_former.eval()
    
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
        
        # There's no need to multiply logit_scale since we're only interested in the order of similarity
        probs_text_to_img = (text_embeddings @ img_embeddings.t()).softmax(dim=-1)
    
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
    
    for k, v in results.items():
        logger.info(f'{k}: {v}')
        
    return results


# TODO:
# 1. Load model from config
#  ** If not specify load_from, load clip pre-trained model from huggingface
# 2. Test and log results
if __name__ == '__main__':
    @hydra.main(config_path='config', config_name='test', version_base=None)
    def main(cfg: DictConfig):
        metadata = OmegaConf.load(cfg.dataset.dataset_path)
        processor = CLIPProcessor.from_pretrained(cfg.model.pretrained_model_name)
        test_dataset = TextImageDataset(
            metadata=metadata['test_samples'],
            processor=processor,
            permute_colors=False,
            img_size=cfg.dataset.img_size
        )
        
        # 1. If there's no load_from
        # Codes that load model from huggingface and test
        ...
        
        # 2. Load model from load_from
        # Codes that load model from load_from and test
        config = OmegaConf.load(cfg.model.load_from + './hydra/config.yaml')
        ...
        
        # 3. Test!
        ...
        
        # 4. Log results
        ...
        
    main()
    