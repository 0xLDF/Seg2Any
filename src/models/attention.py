from typing import Callable, List, Optional, Tuple, Union, Any, Dict
import torch.nn.functional as F
import torch

from utils.lora_controller import select_lora


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:

    cos, sin = freqs_cis  
    if cos.ndim == 2:
        # [S, D] -> [B, H, S, D]
        cos = cos[None, None]
        sin = sin[None, None]
    elif cos.ndim == 3:
        # [B, S, D] -> [B, H, S, D]
        cos = cos.unsqueeze(dim=1)
        sin = sin.unsqueeze(dim=1)
        
    cos, sin = cos.to(x.device), sin.to(x.device)
    
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out

class FluxRegionalAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        cond_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cond_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        with select_lora((attn.to_q, attn.to_k, attn.to_v),'default'):
            # load default lora for noisy image token
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            
        if cond_hidden_states is not None:
            with select_lora((attn.to_q, attn.to_k, attn.to_v),'cond'):
                # load default lora for condition token
                cond_query = attn.to_q(cond_hidden_states)
                cond_key = attn.to_k(cond_hidden_states)
                cond_value = attn.to_v(cond_hidden_states)
            
            cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            if attn.norm_q is not None:
                cond_query = attn.norm_q(cond_query)
            if attn.norm_k is not None:
                cond_key = attn.norm_k(cond_key)
            if cond_rotary_emb is not None:
                cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
                cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)
            query = torch.cat([query, cond_query], dim=2)
            key = torch.cat([key, cond_key], dim=2)
            value = torch.cat([value, cond_value], dim=2)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            if cond_hidden_states is not None:
                encoder_hidden_states, hidden_states, cond_hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] : -cond_hidden_states.shape[1]],
                    hidden_states[:, -cond_hidden_states.shape[1] :],
                )
                with select_lora((attn.to_out[0],),'cond'):
                    # linear proj
                    cond_hidden_states = attn.to_out[0](cond_hidden_states)
                    # dropout
                    cond_hidden_states = attn.to_out[1](cond_hidden_states)
            else:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] :],
                )
            
            with select_lora((attn.to_out[0],),'default'):
                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
 
            return (
                (hidden_states, encoder_hidden_states, cond_hidden_states)
                if cond_hidden_states is not None else (hidden_states, encoder_hidden_states)
            )
        else:
            if cond_hidden_states is not None:
                hidden_states, cond_hidden_states = (
                    hidden_states[:, : -cond_hidden_states.shape[1]],
                    hidden_states[:, -cond_hidden_states.shape[1] :],
                )
                return (hidden_states, cond_hidden_states)
            else:
                return (hidden_states,)