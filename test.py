import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

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
    """
    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def test_model(test_loader: torch.utils.data.DataLoader, model: nn.Module, device: str):
    model.eval()
    
    with torch.inference_mode():
        img_embeddings = []
        text_embeddings = []
        labels = []
        
        # Fisrt get all image/text embeddings
        pbar = tqdm(test_loader, desc='Testing ...')
        for batch in pbar:
            labels.append(batch['labels'].cpu())
            
            del batch['labels']
            
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            outputs = model(**batch, return_loss=False)
             
            img_embeddings.append(outputs.image_embeds.cpu())
            text_embeddings.append(outputs.text_embeds.cpu())
            
        img_embeddings = torch.cat(img_embeddings, dim=0).to(device)
        text_embeddings = torch.cat(text_embeddings, dim=0).to(device)
        labels = torch.cat(labels, dim=0).to(device)
        
        probs_img_to_text = (img_embeddings @ text_embeddings.T).softmax(dim=-1)
    
    pred_top_1 = probs_img_to_text.topk(1, dim=-1).indices
    pred_top_5 = probs_img_to_text.topk(5, dim=-1).indices
    
    recall_at_1 = calc_recall_at_k(labels, pred_top_1, 1)
    recall_at_5 = calc_recall_at_k(labels, pred_top_5, 5)
    recall_at_20 = calc_recall_at_k(labels, pred_top_5, 20)
    
    results = {
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'recall_at_20': recall_at_20
    }
    
    for k, v in results.items():
        logger.info(f'{k}: {v}')
        
    return results


# # TODO:
# # 1. Load model from config
# #  ** If not specify load_from, load clip pre-trained model from huggingface
# # 2. Test and log results
# @hydra.main(config_path='config', config_name='test')
# def main(cfg: DictConfig):
#     ...


# if __name__ == '__main__':
#     main()
