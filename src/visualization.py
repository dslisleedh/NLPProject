import torch
import torch.nn as nn
import torch.functional as F

import einops

from typing import Optional


class GradCAM:
    def __init__(self, device):
        self.device = device
        
    def hook_grad(self, module, grad_in, grad_out):
        self.data = grad_out
        grad_out.requires_grad_(True)
        grad_out.retain_grad()
        
    def clear_hook(self):
        self.data = None
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad
    
    def __call__(
        self, model: nn.Module, x: torch.Tensor, class_idx: Optional[int] = None
    ) -> torch.Tensor:
        # model and inputs to device
        cur_device = model.visual.layer4.weight.device
        model.to(self.device)
        x = x.to(self.device)
        
        # remove gradient of input
        x = x.detach()
        x = x.requires_grad_(True)
        
        # Change model to eval mode and turn off gradients
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
            
        # Register hook
        model.visual.layer4.register_backward_hook(self.hook_grad)
        
        # Forward pass
        logits_per_image, _ = model(x)
        
        # Get gradient
        if class_idx is None:
            class_idx = logits_per_image.topk(1, dim=-1).indices.squeeze(-1)
        logits_per_image[:, class_idx].backward()
        
        # Get Grad from last Conv layer
        act = self.activation
        grad = self.gradient
        
        # Get GradCAM
        alpha = grad.mean(dim=[2, 3], keepdim=True)
        cam = (alpha * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam) / torch.max(cam) + 1e-8  # Normalize
        
        # Upsample GradCAM
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Clear hook and turn on gradients
        self.clear_hook()
        model.to(cur_device)
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        
        return cam
    