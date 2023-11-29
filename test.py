import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

import torchmetrics as tm

import logging

import hydra
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger('Test')


def test_model(test_loader: torch.utils.data.DataLoader, model: nn.Module, device: str):
    model.eval()
    
    # Metrics
    recall_1 = tm.Recall(num_classes=1, average='micro').to(device)
    recall_5 = tm.Recall(num_classes=5, average='micro').to(device)
    precision_1 = tm.Precision(num_classes=1, average='micro').to(device)
    precision_5 = tm.Precision(num_classes=5, average='micro').to(device)
    
    with torch.no_grad():
        img_embeddings = []
        text_embeddings = []
        labels = []
        
        # Fisrt get all image/text embeddings
        for batch in test_loader:
            labels.append(batch['label'].cpu())
            
            del batch['label']
            
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            outputs = model(**batch, return_loss=False)
             
            img_embeddings.append(outputs.image_embeds.cpu())
            text_embeddings.append(outputs.text_embeds.cpu())
            
        img_embeddings = torch.cat(img_embeddings, dim=0).to(device)
        text_embeddings = torch.cat(text_embeddings, dim=0).to(device)
        labels = torch.cat(labels, dim=0).to(device)
        
        probs_img_to_text = (img_embeddings @ text_embeddings.T).softmax(dim=-1)
        
        recall_1.update(probs_img_to_text, labels)
        recall_5.update(probs_img_to_text, labels)
        precision_1.update(probs_img_to_text, labels)
        precision_5.update(probs_img_to_text, labels)
        
    results = {
        "recall_1": recall_1.compute().item(),
        "recall_5": recall_5.compute().item(),
        "precision_1": precision_1.compute().item(),
        "precision_5": precision_5.compute().item(),
    }
    for k, v in results.items():
        logger.info(f'{k}: {v}')
        
    return results


# TODO:
# 1. Load model from config
#  ** If not specify load_from, load clip pre-trained model from huggingface
# 2. Test and log results
@hydra.main(config_path='config', config_name='test')
def main(cfg: DictConfig):
    ...


if __name__ == '__main__':
    main()
