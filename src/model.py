"""
TODO:
1. Classic Huggingface Model (Fine tuning)
2. Prompt Learning Model
3. Prompt Learning From Visual tokens
4. Auxiliary Model
5. Focal loss instead of CE(becuase our dataset is so noisy)


HOW TO USE MULTIMODAL-MIXUP?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel as CLIPModel_hf
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPOutput

from omegaconf import DictConfig, OmegaConf

from copy import deepcopy

from typing import Optional, Union, Tuple, List
import logging


model_logger = logging.getLogger('model')


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
        self, name: str, model_config: DictConfig, prompt_learning: bool = False,
        prompt_from_visual_tokens: bool = False, auxiliary_model: bool = False, m2_loss: bool = False
    ):
        super().__init__()
        # Dictconfig to dict
        # model_config_dict = OmegaConf.to_container(model_config)
        self.clip = CLIPModel_hf.from_pretrained(name)
        self.prompt_learning = prompt_learning
        self.prompt_from_visual_tokens = prompt_from_visual_tokens
        self.auxiliary_model = auxiliary_model
        self.m2_loss = m2_loss  # Multi-model geodesic loss
        
        if self.prompt_learning or self.auxiliary_model:
            for param in self.clip.parameters():
                param.requires_grad = False

        if self.prompt_learning:
            embedding_dim = self.clip.config.projection_dim
            self.prompt = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
            self.prompt.requires_grad = True
            
            if self.prompt_from_visual_tokens:
                self.visual_prompt_conditioner = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, embedding_dim)
                )
    
    def forward(self, **kwargs):
        if not self.prompt_learning and not self.auxiliary_model and not self.m2_loss:
            return self.clip(**kwargs)
        
        if self.auxiliary_model:
            raise NotImplementedError
        
        else:
            image_embeds = self.get_visual_normalized_feat(kwargs["pixel_values"])
            
            prompt = self.prompt.expand(image_embeds.shape[0], -1)
            if self.prompt_from_visual_tokens:
                prompt = prompt + self.visual_prompt_conditioner(kwargs["pixel_values"])
                
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
        return image_embeds
    
    def get_text_normalized_feat(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt: torch.Tensor
    ) -> torch.Tensor:
        output_attentions = self.clip.config.output_attentions
        output_hidden_states = (self.clip.config.output_hidden_states)
        return_dict = self.clip.config.use_return_dict
        input_shape = input_ids.size()
        input_shape = (
            input_shape[0], input_shape[1] + 1
        )
        
        hidden_states = self.clip.text_model.embeddings(input_ids=input_ids, position_ids=None)
        # Concat prompt to hidden states
        hidden_states = torch.cat([prompt.unsqueeze(1), hidden_states], dim=1)
        # Add ones to attentionmask for prompt
        attention_mask = torch.cat([torch.ones(attention_mask.shape[0], 1).to(hidden_states.device), attention_mask], dim=1)
        
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
        last_hidden_state = self.clip.text_model.final_layer_norm(last_hidden_state[:, 1:])  # Drop the prompt

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