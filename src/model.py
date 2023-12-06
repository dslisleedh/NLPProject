"""
TODO:
    Change Code to use CLIP(Github) instead of CLIP(Huggingface)
    To use More model and GradCAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

# from transformers import CLIPModel as CLIPModel_hf
# from transformers.modeling_attn_mask_utils import AttentionMaskConverter
# from transformers.modeling_outputs import BaseModelOutputWithPooling
# from transformers.models.clip.modeling_clip import CLIPOutput
from CLIP.clip import load
from CLIP.clip.model import CLIP, build_model, convert_weights

from omegaconf import DictConfig, OmegaConf

from copy import deepcopy

from typing import Optional, Union, Tuple, List
from functools import partial
import logging

from random import random


model_logger = logging.getLogger('model')


"""
Implement Q-former

Which fuse CLS Token and given visual token
"""
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
        else:
            output = x
        return output


class MHSA(nn.Module):
    def __init__(
        self, dim: int, embedding_dim: int, cross_attn: bool,
        drop_rate: float = 0.1, heads: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.cross_attn = cross_attn
        
        self.pre_norm = nn.LayerNorm(dim)
        qk_inp_dim = embedding_dim if cross_attn else dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(qk_inp_dim, dim, bias=False)
        self.to_v = nn.Linear(qk_inp_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

        self.path_drop = DropPath(drop_rate)
        
        nn.init.normal_(self.to_q.weight, std=0.02)
        nn.init.normal_(self.to_k.weight, std=0.02)
        nn.init.normal_(self.to_v.weight, std=0.02)
        nn.init.normal_(self.to_out.weight, std=0.02)
        
    def forward(self, x: torch.Tensor, img_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        # IMG_Context:
        # B, 1, C
        
        x = self.pre_norm(x)
        
        if self.cross_attn:
            q = self.to_q(x)
            k = self.to_k(img_context)
            v = self.to_v(img_context)
            k = k.expand(B, N, -1)
            v = v.expand(B, N, -1)
        else:
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = einops.rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = einops.rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = einops.rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        score = q @ k.transpose(-2, -1) / (self.dim ** 0.5)
        attn = score.softmax(dim=-1)
        
        out = einops.rearrange(attn @ v, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.path_drop(out)
        return out        
    

class QFormerBlock(nn.Module):
    def __init__(
        self, dim: int, embedding_dim: int, ffn_exp_ratio: int,
        drop_rate: float = 0.1, heads: int = 8
    ):
        super().__init__()
        self.self_attn = MHSA(dim, embedding_dim, cross_attn=False, drop_rate=drop_rate, heads=heads)
        self.cross_attn = MHSA(dim, embedding_dim, cross_attn=True, drop_rate=drop_rate, heads=heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ffn_exp_ratio),
            nn.GELU(),
            nn.Linear(dim * ffn_exp_ratio, dim),
            DropPath(drop_rate)
        )
        
    def forward(self, x: torch.Tensor, img_context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        x = x + self.cross_attn(x, img_context)
        x = x + self.ffn(x)
        return x
    

class QFormer(nn.Module):
    def __init__(
        self, n_cls: int, dim: int, depth: int,
        embedding_dim: int, ffn_exp_ratio: int, drop_rate: float = 0.1, heads: int = 8,
        out_dim: Optional[int] = None
    ):
        super().__init__()
        self.register_buffer('cls_token', torch.randn(1, n_cls, dim) * 0.01)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_cls, dim) * 0.01, requires_grad=True)
        self.layers = nn.ModuleList([
            QFormerBlock(dim, embedding_dim, ffn_exp_ratio, drop_rate, heads) for _ in range(depth)
        ])
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
        
    def forward(self, img_context: torch.Tensor) -> torch.Tensor:
        B = img_context.shape[0]
        img_context = img_context.type(self.cls_token.dtype)
        cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embedding
        for layer in self.layers:
            cls_token = layer(cls_token, img_context)
        return self.to_out(cls_token)

    
def encode_text_prompt(self, text):
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    
    x = x + self.positional_embedding.type(self.dtype)
    
    # Add prompt
    prompt = self.prompts.expand(x.shape[0], -1, -1).to(x.device).type(self.dtype)  
    # Prompt shape = [batch_size, n_prompt, d_model]
    x = torch.cat([prompt, x], dim=1)
    
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    return x
    
def encode_text_visual_prompt(self, text):
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    
    x = x + self.positional_embedding.type(self.dtype)
    
    # Add prompt
    prompt = self.q_former(self.encoded_image.unsqueeze(1)).type(self.dtype)  # Which is attention pooled image embedding
    x = torch.cat([prompt, x], dim=1)
    
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    return x
    

def encode_image(self, image):
    image_embedding = self.visual(image.type(self.dtype))
    self.encoded_image = image_embedding.detach().clone()
    return image_embedding


def prepare_for_prompt_learning(model: nn.Module, n_prompt: int):
    for param in model.parameters():
        param.requires_grad = False
    
    model.context_length = model.context_length + n_prompt
    for layer in model.transformer.resblocks:
        layer.attn_mask = model.build_attention_mask()
        
    return model

        
def get_model(model_config: DictConfig) -> nn.Module:
    model, processor = load(model_config.name, jit=False)
    
    del model.logit_scale  # to make tau = 1.0
    model.logit_scale = nn.Parameter(torch.tensor(model_config.logit_scale), requires_grad=True)
    # since it is exponetialed and multiplied to logits
    # Many researchs found that when temperature goes to 0, it is same as triplet loss with 0 margin
    # So we change temperature to 1.0 to consider negative samples more.
    
    if model_config.m2_loss_weight is not None:
        model.logit_scale_2 = nn.Parameter(torch.tensor(model_config.logit_scale), requires_grad=True)
    
    # if model_config.prompt_learning, override encode_text and encode_image
    if model_config.prompt_learning and model_config.prompt_from_visual_tokens:
        out_dim = model.transformer.width
        embedding_dim = model.visual.output_dim
        n_cls = model_config.n_prompt
        qformer_kwargs = OmegaConf.to_container(model_config.qformer_config)
        qformer_kwargs['out_dim'] = out_dim
        qformer_kwargs['embedding_dim'] = embedding_dim
        qformer_kwargs['n_cls'] = n_cls
        model = prepare_for_prompt_learning(model, model_config.n_prompt)
        model.q_former = QFormer(**qformer_kwargs)
        model.encode_text = partial(encode_text_visual_prompt, model)
        model.encode_image = partial(encode_image, model)
        
    elif model_config.prompt_learning:
        model = prepare_for_prompt_learning(model, model_config.n_prompt)
        model.prompts = nn.Parameter(
            torch.randn(1, model_config.n_prompt, model.transformer.width), requires_grad=True)
        model.encode_text = partial(encode_text_prompt, model)
        
    return model, processor
