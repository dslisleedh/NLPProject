import torch
import torch.nn as nn
import torch.functional as F

from typing import Optional


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hook = None
        self.data = None
    
    def __enter__(self):
        self.hook = self.model.visual.layer4.register_forward_hook(self.hook_fn_forward)
        return self
        
    def __exit__(self, *args):
        self.hook.remove()
        
    def hook_fn_forward(self, module, input, output):
        self.data = output
        output.retain_grad()

    @property
    def get_activation_maps(self):
        return self.data
    
    @property
    def get_gradients(self):
        return self.data.grad
    