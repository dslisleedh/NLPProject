"""
TODO:
    Change Code to use CLIP(Github) instead of CLIP(Huggingface)
    To use More model and GradCAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from transformers import CLIPModel as CLIPModel_hf
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPOutput

from omegaconf import DictConfig, OmegaConf

from copy import deepcopy

from typing import Optional, Union, Tuple, List
import logging


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
        self.to_q = nn.Linear(qk_inp_dim, dim, bias=False)
        self.to_k = nn.Linear(qk_inp_dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
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
            q = self.to_q(img_context)
            k = self.to_k(img_context)
            v = self.to_v(x)
            q = q.expand(-1, N, -1)
            k = k.expand(-1, N, -1)
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
        self.cls_token = nn.Parameter(torch.randn(1, n_cls, dim) * 0.02, requires_grad=False)
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
        cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embedding
        for layer in self.layers:
            cls_token = layer(cls_token, img_context)
        return self.to_out(cls_token)
    

def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _create_4d_causal_attention_mask(
    input_shape: Union[torch.Size, Tuple, List],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`

    Args:
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        device (`int`):
            The torch device the created mask shall have.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
    )

    return attention_mask


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class CLIPModel(nn.Module):
    def __init__(
        self, name: str, qformer_config: DictConfig,
        prompt_learning: bool = False, n_prompt: int = 2, prompt_from_visual_tokens: bool = False,
        auxiliary_model: bool = False, m2_loss: bool = False
    ):
        super().__init__()
        # Dictconfig to dict
        # model_config_dict = OmegaConf.to_container(model_config)
        self.clip = CLIPModel_hf.from_pretrained(name)
        self.prompt_learning = prompt_learning
        self.n_prompt = n_prompt
        self.prompt_from_visual_tokens = prompt_from_visual_tokens
        self.auxiliary_model = auxiliary_model
        self.m2_loss = m2_loss  # Multi-model geodesic loss
        
        if self.prompt_learning or self.auxiliary_model:
            for param in self.clip.parameters():
                param.requires_grad = False
        """
        self, n_cls: int, dim: int, depth: int,
        embedding_dim: int, ffn_exp_ratio: int, drop_rate: float = 0.1, heads: int = 8,
        out_dim: Optional[int] = None
        """
        if self.prompt_learning:
            if self.prompt_from_visual_tokens:
                vision_embedding_dim = self.clip.vision_model.config.hidden_size
                out_dim = self.clip.config.projection_dim
                qformer_kwargs = OmegaConf.to_container(qformer_config)
                qformer_kwargs['embedding_dim'] = vision_embedding_dim
                qformer_kwargs['out_dim'] = out_dim
                qformer_kwargs['n_cls'] = n_prompt
                self.qformer = QFormer(**qformer_kwargs)
            
            else:
                embedding_dim = self.clip.config.projection_dim
                self.prompt = nn.Parameter(torch.randn(n_prompt, embedding_dim), requires_grad=True)
    
    def forward(self, **kwargs):
        if not self.prompt_learning and not self.auxiliary_model and not self.m2_loss:
            return self.clip(**kwargs)
        
        if self.auxiliary_model:
            raise NotImplementedError
        
        else:
            image_embeds, raw_image_embeds = self.get_visual_normalized_feat(kwargs["pixel_values"])
            # Raw image embeds for Q-Former if needed
            
            if self.prompt_from_visual_tokens:
                prompt = self.qformer(raw_image_embeds.unsqueeze(1))
            else:
                prompt = self.prompt.expand(image_embeds.shape[0], -1)
                
            text_embeds = self.get_text_normalized_feat(
                kwargs["input_ids"], kwargs["attention_mask"], prompt
            )
            
        logit_scale = self.clip.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        if kwargs['return_loss']:
            loss = clip_loss(logits_per_text)
        else:
            loss = None

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )
            
    def get_visual_normalized_feat(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_attentions = self.clip.config.output_attentions
        output_hidden_states = (self.clip.config.output_hidden_states)
        return_dict = self.clip.config.use_return_dict
        
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[1]
        image_embeds = self.clip.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds, vision_outputs[1]
    
    def get_text_normalized_feat(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt: torch.Tensor
    ) -> torch.Tensor:
        output_attentions = self.clip.config.output_attentions
        output_hidden_states = (self.clip.config.output_hidden_states)
        return_dict = self.clip.config.use_return_dict
        input_shape = input_ids.size()
        input_shape = (
            input_shape[0], input_shape[1] + self.n_prompt
        )
        
        hidden_states = self.clip.text_model.embeddings(input_ids=input_ids, position_ids=None)
        # Concat prompt to hidden states
        hidden_states = torch.cat([prompt, hidden_states], dim=1)
        # Add ones to attentionmask for prompt
        attention_mask = torch.cat([
            torch.ones(attention_mask.shape[0], self.n_prompt).to(hidden_states.device), attention_mask
        ], dim=1)
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.clip.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.clip.text_model.final_layer_norm(last_hidden_state[:, self.n_prompt:])  # Drop the prompt

        if self.clip.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.clip.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        text_features = self.clip.text_projection(pooled_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features
        
        
def get_model(model_config: DictConfig) -> nn.Module:
    model = CLIPModel(**model_config)
    logging.info(f'Loaded model from {model_config.name}')
    logging.info(model)
    return model