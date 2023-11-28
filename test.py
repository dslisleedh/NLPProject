import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

import torchmetrics as tm

import logging

logger = logging.getLogger('Test')


"""
TODO:
    1. Get Cosine Similarity between image and text using CLIP
    2. Check Recall@1, Recall@5, Precision@1, Precision@5
"""


def test_model(test_loader: torch.utils.data.DataLoader, model: nn.Module, device: str):
    model.eval()
    
    # Metrics
    recall_1 = tm.Recall(num_classes=1, average='micro')
    recall_5 = tm.Recall(num_classes=5, average='micro')
    precision_1 = tm.Precision(num_classes=1, average='micro')
    precision_5 = tm.Precision(num_classes=5, average='micro')
    
    with torch.no_grad():
        for batch in test_loader:
            images, texts, labels = batch
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            
            # Get cosine similarity
            outputs = model(images, texts)
            similarity = outputs.logits
            
            # Get top 5 indices
            _, indices = torch.topk(similarity, 5)
            
            # Get top 1 indices
            _, index = torch.topk(similarity, 1)
            
            # Get top 5 predictions
            predictions_5 = labels[indices]
            
            # Get top 1 predictions
            predictions_1 = labels[index]
            
            # Update metrics
            recall_1(predictions_1, labels)
            recall_5(predictions_5, labels)
            precision_1(predictions_1, labels)
            precision_5(predictions_5, labels)
    
    results = {
        "recall_1": recall_1.compute(),
        "recall_5": recall_5.compute(),
        "precision_1": precision_1.compute(),
        "precision_5": precision_5.compute()
    }
    for k, v in results.items():
        logger.info(f'{k}: {v}')
        
    return results


if __name__ == '__main__':
    ...
