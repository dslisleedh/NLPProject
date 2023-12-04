import torch
import torch.nn as nn
import torch.functional as F

import einops


class GradCAM:
    def __init__(self, device):
        self.device = device
        
    def hook_grad(self, grad_in, grad_out):
        self.grad = grad_in[0]
    
    def visualize(
        self, input: torch.Tensor, grad_feat: torch.Tensor
    ) -> torch.Tensor:
        grad = grad_feat.to(self.device)
        hw = int(grad.shape[1] ** 0.5)
        grad = einops.rearrange(grad, "b (h w) c -> b c h w", h=hw)
        
    
    def __call__(
        self, model: nn.Module, input: torch.Tensor
    ) -> torch.Tensor:
        model.eval()
        model.to(self.device)
        
        # register hook
        # model.layer4.register_backward_hook(self.hook_grad)
        model.clip.vision_model.encoder.register_backward_hook(self.hook_grad)
        # forward
        output = model(input)
        output = F.softmax(output, dim=1)
        pred = output.argmax(dim=1)
        
        # backward
        output[:, pred].backward()
        
        # visualize
        grad_feat = self.feature.grad
        grad_feat = grad_feat.mean(dim=[2, 3], keepdim=True)
        grad_feat = F.relu(grad_feat)
        grad_feat = grad_feat / (grad_feat.max() - grad_feat.min())
        grad_feat = grad_feat.repeat(1, 3, 1, 1)
        grad_feat = grad_feat.detach().cpu()
        
        return grad_feat