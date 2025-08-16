# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import warnings
import numpy as np
import torch
import cv2
import PIL

from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, 
    T5EncoderModel, 
    T5TokenizerFast,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)

from ..models.transformer_flux import FluxTransformer2DModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class FluxRegionalPipeline(FluxPipeline):   
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):        
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor
        )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor
        )
    

    def check_inputs(
        self,
        global_prompt,
        regional_prompts,
        regional_labels,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if global_prompt is not None:
            if len(global_prompt) != len(regional_prompts) or len(regional_prompts) != len(regional_labels):
                raise ValueError("global_prompt, regional_prompts, regional_labels batch sizes must match.")  
        
        for regional_prompt, region_label in zip(regional_prompts,regional_labels):
            if len(regional_prompt) != len(region_label):
                raise ValueError("Each regional_prompt list length must match its region_label length.")
        
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )
        

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def _get_t5_prompt_embeds_nopad(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # get text token without padding
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        
        new_prompt_embeds = []
        attention_mask = text_inputs.attention_mask.to(device=device)
        for i in range(len(prompt_embeds)):
            new_prompt_embeds.append(prompt_embeds[i][attention_mask[i]==1]) # drop padding
        prompt_embeds = new_prompt_embeds
        
        return prompt_embeds

          
    def encode_all_prompt(
        self,
        global_prompt: Optional[Union[str, List[str]]],
        regional_prompts:List[List[str]],
        global_max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device
        dtype = self.text_encoder_2.dtype
        
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)
            
        # process global_prompt
        if global_prompt is None:
            # when no global text, use concatenated regional_prompts for pooled_prompt_embeds.
            global_prompt = ['.'.join(r) for r in regional_prompts] 
            
            global_prompt_embeds = None
            global_pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=global_prompt,
                device=device,
            )
        else:
            global_prompt = [global_prompt] if isinstance(global_prompt, str) else global_prompt
            
            global_prompt_embeds = self._get_t5_prompt_embeds_nopad(
                prompt=global_prompt,
                max_sequence_length=global_max_sequence_length,
                device=device,
            )
            global_pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=global_prompt,
                device=device,
            )
        
        # process regional_prompts
        if isinstance(regional_prompts,list) and isinstance(regional_prompts[0],str):
            regional_prompts = [regional_prompts]
        
        batch_size =  len(regional_prompts)

        arr = np.array([0] + [len(p) for p in regional_prompts])        
        regional_prompts_offset = np.cumsum(arr)

        # flatten for batch inference
        flatten_regional_prompts = [item for sublist in regional_prompts for item in sublist]
        flatten_regional_prompt_embeds = self._get_t5_prompt_embeds_nopad(
            prompt=flatten_regional_prompts,
            max_sequence_length=regional_max_sequence_length,
            device=device,
        ) # list[tensor]
        
        # Extract the regional_prompt_embeds of each image.
        result_prompt_embeds = []
        max_len = 0
        txt_seq_lens = []

        for i in range(len(regional_prompts_offset)-1):
            offset_begin, offset_end = regional_prompts_offset[i], regional_prompts_offset[i+1]
            tmp = []
            txt_seq_len = []
            if global_prompt_embeds is not None:
                # first element is length of global text token
                tmp.append(global_prompt_embeds[i])
                txt_seq_len.append(global_prompt_embeds[i].shape[0])
            else:
                txt_seq_len.append(0)
                
            for i in range(offset_begin,offset_end):
                tmp.append( flatten_regional_prompt_embeds[i] )
                txt_seq_len.append(flatten_regional_prompt_embeds[i].shape[0])

            txt_seq_lens.append(txt_seq_len)
            tmp = torch.concat(tmp,dim=0)
            result_prompt_embeds.append(tmp)
            max_len = max(max_len, tmp.shape[0])
        
        # Pad with zero embeddings to align to max_len in the batch.       
        for i in range(len(result_prompt_embeds)):
            cur_len = len(result_prompt_embeds[i])
            pad_len = max_len-cur_len
            if pad_len>0:
                txt_seq_lens[i].append(pad_len)
            
                empty_prompt_embeds = torch.zeros([pad_len, result_prompt_embeds[i].shape[1]],dtype=dtype,device=device)
                result_prompt_embeds[i] = torch.concat([result_prompt_embeds[i],empty_prompt_embeds],dim=0)

        result_prompt_embeds = torch.stack(result_prompt_embeds,dim=0)
        
        # repeat num_images_per_prompt
        new_txt_seq_lens = []
        for lens in txt_seq_lens:
            for _ in range(num_images_per_prompt):
                new_txt_seq_lens.append(lens)
        txt_seq_lens = new_txt_seq_lens
        global_pooled_prompt_embeds = global_pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        global_pooled_prompt_embeds = global_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        
        result_prompt_embeds = result_prompt_embeds.repeat(1, num_images_per_prompt,1)
        result_prompt_embeds = result_prompt_embeds.view(batch_size * num_images_per_prompt, -1,result_prompt_embeds.shape[-1])
        
        text_ids = torch.zeros(result_prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        
        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return result_prompt_embeds, global_pooled_prompt_embeds, txt_seq_lens, text_ids
    
    @staticmethod
    def prepare_attention_mask(
        attention_mask_method: str,
        regional_labels: list[torch.FloatTensor],
        txt_seq_lens: list[list[int]],
        cond_seq_lens: list[int],
        pad_seq_lens: Optional[list[int]],
        height: int,
        width: int,
        num_attention_heads,
        dtype,
        device,
        cond2image_attention_weight=1,
    ):
        """
        prepare masks for regional control
        regional_labels: list[torch.FloatTensor], where len(regional_labels)==batch_size, each tensor shape (n, h, w) where n is entity count.
        txt_seq_lens: list[list[int]]
        cond_seq_lens: list[int]
        pad_seq_lens: Optional[list[int]]
        """
        total_txt_seq_len = sum(txt_seq_lens[0])
        for lens in txt_seq_lens:
            assert total_txt_seq_len == sum(lens)
        
        temp = cond_seq_lens[0] + pad_seq_lens[0]
        for cond_seq_len,pad_seq_len in zip(cond_seq_lens,pad_seq_lens):
            assert cond_seq_len+pad_seq_len==temp

        
        img_seq_len = height * width
        
        attention_masks = []
        hard_attention_masks = []        
        
        for bs_i,label in enumerate(regional_labels):
            # initialize attention mask
            # sequence: global_txt_token, regional_txt_token, padding_txt_token, noisy_image_token, condition_token, padding_token
            attention_mask = torch.zeros(
                (total_txt_seq_len+img_seq_len+cond_seq_lens[bs_i]+pad_seq_lens[bs_i], 
                 total_txt_seq_len+img_seq_len+cond_seq_lens[bs_i]+pad_seq_lens[bs_i]),
                dtype=torch.float32,
                device=device
            )
            
            if attention_mask_method == "hard":
                # initialize self-attended mask / union mask / fg_masks
                self_attend_masks = torch.zeros((img_seq_len, img_seq_len), dtype=torch.bool,device=device)
                union_masks = torch.zeros((img_seq_len, img_seq_len), dtype=torch.bool,device=device)
                fg_masks = torch.zeros((img_seq_len,), dtype=torch.bool,device=device)
            
            img_start = total_txt_seq_len
            img_end = img_start + img_seq_len
            gloabl_seq_len = txt_seq_lens[bs_i][0]
            
            # img+cond attends to itself
            attention_mask[img_start:img_end+cond_seq_lens[bs_i], img_start:img_end+cond_seq_lens[bs_i]] = True
            
            # cond to image
            attention_mask[img_start:img_end, img_end:img_end+cond_seq_lens[bs_i]] = cond2image_attention_weight
            attention_mask[img_end:img_end+cond_seq_lens[bs_i], img_start:img_end] = cond2image_attention_weight

            # global txt attends to itself
            attention_mask[:gloabl_seq_len,:gloabl_seq_len] = True
            
            # global txt attends to img
            attention_mask[:gloabl_seq_len,img_start:img_end] = True
            
            # img attends to global txt
            attention_mask[img_start:img_end,:gloabl_seq_len] = True

            if pad_seq_lens[bs_i]!=0:
                # pad attends to pad
                attention_mask[:, -pad_seq_lens[bs_i]:] = False
                attention_mask[-pad_seq_lens[bs_i]:, :] = False
                attention_mask[-pad_seq_lens[bs_i]:, -pad_seq_lens[bs_i]:] = True

            arr = np.array([0] + [l for l in txt_seq_lens[bs_i][1:]]) # get regional text offset
            region_offset = np.cumsum(arr)
      
            for region_i in range(len(region_offset)-1):             
                # regional txt attends to itself
                region_txt_start = gloabl_seq_len + region_offset[region_i]
                region_txt_end = gloabl_seq_len + region_offset[region_i+1]
                region_seq_len = region_txt_end - region_txt_start
                attention_mask[region_txt_start:region_txt_end, region_txt_start:region_txt_end] = True
                
                if region_i>=len(label): 
                    # skip padding_txt_token
                    continue
                
                mask = label[region_i]
                mask = mask.float() # shaep: h,w

                if attention_mask_method in ["base","hard"]:
                    mask = torch.nn.functional.interpolate(mask[None, None, :, :], (height, width), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, region_seq_len)
                    # hw,region_seq_len
                elif attention_mask_method == "place": 
                    mask = torch.nn.functional.pixel_unshuffle(mask[None, :, :],mask.shape[0]//height) # (down_factor**2,h,w)
                    chs_num = mask.shape[0]
                    mask = torch.sum(mask,dim=0,keepdim=True)/chs_num # (1,h,w)
                    mask = mask.flatten().unsqueeze(1).repeat(1, region_seq_len)
                else:
                    raise ValueError(f"attention_mask_method must be one of ['base','place','hard'], got {attention_mask_method}")

                # regional txt attends to corresponding regional img
                attention_mask[region_txt_start:region_txt_end, img_start:img_end] = mask.transpose(-1, -2)

                # regional img attends to corresponding txt
                attention_mask[img_start:img_end, region_txt_start:region_txt_end] = mask

                if attention_mask_method == "hard":
                    # update fg_masks / self_attend_masks / union_masks
                    fg_masks = torch.logical_or(fg_masks, mask[:, 0])
                    
                    img_size_masks = mask[:, :1].repeat(1, img_seq_len)
                    img_size_masks_transpose = img_size_masks.transpose(-1, -2)
                    self_attend_masks = torch.logical_or(self_attend_masks, torch.logical_and(img_size_masks, img_size_masks_transpose))

                    union_masks = torch.logical_or(union_masks, torch.logical_or(img_size_masks, img_size_masks_transpose))
            
            if attention_mask_method == "hard":
                hard_attention_mask = attention_mask.clone()
                
                # img to img
                background_masks = torch.logical_not(union_masks)
                background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)
                hard_attention_mask[img_start:img_end, img_start:img_end] = background_and_self_attend_masks
                
                # mask fg img to global txt
                fg_patch_idxs = fg_masks.nonzero(as_tuple=True)[0].to(device)
                hard_attention_mask[img_start+fg_patch_idxs, :gloabl_seq_len] = 0
                
                # keep bg img to all img
                bg_patch_idxs = (fg_masks == 0).nonzero(as_tuple=True)[0].to(device)
                hard_attention_mask[img_start+bg_patch_idxs, img_start:img_end] = 1

                hard_attention_mask = torch.log(hard_attention_mask).to(dtype)
                hard_attention_masks.append(hard_attention_mask)
       
       
            attention_mask = torch.log(attention_mask).to(dtype)
            attention_masks.append(attention_mask) 
                
        attention_masks = torch.stack(attention_masks,dim=0)
        attention_masks = attention_masks.unsqueeze(1)  
        attention_masks = attention_masks.expand(-1, num_attention_heads, -1, -1)  # shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        
        if attention_mask_method == "hard":
            hard_attention_masks = torch.stack(hard_attention_masks,dim=0)
            hard_attention_masks = hard_attention_masks.unsqueeze(1)
            hard_attention_masks = hard_attention_masks.expand(-1, num_attention_heads, -1, -1)
        else:
            hard_attention_masks = None
                
        return attention_masks, hard_attention_masks
    
    def prepare_image(
        self,
        image,
        width,
        height,
        device,
        dtype
    ):
   
        image = self.image_processor.preprocess(image, height=height, width=width)

        image = image.to(device=device, dtype=dtype)

        return image
    
    @staticmethod
    def get_valid_cond_token_num(
        cond,
        vae_scale_factor=16
    ): 
        invalid_value = -1
        if cond.min() >= 0:
            warnings.warn(
                "Passing `cond` with value range in [0,1] is deprecated. The expected cond range for image tensor is [-1,1] "
                f"You passed `cond` with value range [{cond.min()},{cond.max()}]",
            )
            invalid_value = 0
            
        mask = cond.clone()
        mask[mask == invalid_value] = 0
        mask = torch.nn.functional.pixel_unshuffle(mask, vae_scale_factor)
        mask = torch.sum(mask, dim=-3, keepdim=True)
        valid_cond_token_count = torch.sum(mask != 0).item() # get non-zero value condition token num
        return valid_cond_token_count
    
    @staticmethod
    def filter_cond_token(
        cond,
        cond_hidden_states,
        cond_ids,
        vae_scale_factor=16
    ):
        invalid_value = -1
        if cond.min() >= 0:
            warnings.warn(
                "Passing `cond` with value range in [0,1] is deprecated. The expected cond range for image tensor is [-1,1] "
                f"You passed `cond` with value range [{cond.min()},{cond.max()}]",
            )
            invalid_value = 0
            
        valid_cond_hidden_states = []
        valid_cond_ids = []
        pad_seq_lens = []   
        cond_seq_lens = []
        
        mask = cond.clone() # B,C,H,W
        mask[mask==invalid_value] = 0 # after normalize, 0 -> -1  
        mask = torch.nn.functional.pixel_unshuffle(mask,vae_scale_factor)
        mask = torch.sum(mask,dim=-3,keepdim=True) # B,1,H,W
        mask = mask!=0
        mask = mask.flatten(start_dim=1) # B,H*W   

        for i in range(len(cond)):            
            valid_cond_hidden_states.append(cond_hidden_states[i][mask[i]==1]) # L,C
            valid_cond_ids.append(cond_ids[mask[i]==1]) # L,C
            
        max_valid_cond_num = max([len(valid_cond) for valid_cond in valid_cond_hidden_states])
  
        for i in range(len(valid_cond_hidden_states)):
            valid_cond = valid_cond_hidden_states[i]
            pad_seq_len = max_valid_cond_num - len(valid_cond)
            cond_seq_lens.append(len(valid_cond))
            pad_seq_lens.append(pad_seq_len)
            
            pad_cond = torch.zeros((pad_seq_len, valid_cond.shape[-1]), dtype=valid_cond.dtype, device=valid_cond.device)
            valid_cond_hidden_states[i] = torch.concat([valid_cond, pad_cond],dim=0)
            
            valid_id = valid_cond_ids[i]
            pad_id = torch.zeros((pad_seq_len, valid_id.shape[-1]), dtype=valid_id.dtype, device=valid_id.device)
            valid_cond_ids[i] = torch.concat([valid_id, pad_id],dim=0)

        cond_hidden_states = torch.stack(valid_cond_hidden_states, dim=0)
        cond_ids = torch.stack(valid_cond_ids, dim=0)
        return  cond_hidden_states, cond_ids, cond_seq_lens, pad_seq_lens

    @torch.inference_mode()
    def __call__(
        self,
        global_prompt: Optional[Union[str, List[str]]],
        regional_prompts:Union[List[str], List[List[str]]],
        regional_labels:list[torch.FloatTensor],
        negative_prompt: Union[str, List[str]] = None,
        attention_mask_method="hard",
        hard_attn_block_range=[19,37],
        cond: Union[torch.Tensor,List[torch.Tensor]] = None,
        cond_scale_factor: int = 1,
        is_filter_cond_token=True,        
        cond2image_attention_weight=1,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
    ):        
        r"""
        Args:
            attention_mask_method: 
                Determines which attention-mask pattern is injected into the MM-Attention blocks. choices: ["hard", "base", "place"].
                - "hard": the full scheme proposed in Seg2Any. Semantic Alignment Attention (SAA) and Attribute Isolation Attention (AIA) are active.
                - "base": only Semantic Alignment Attention (SAA) is used.
                - "place": Uses the PLACE (https://arxiv.org/abs/2403.01852) attention mask, re-implemented for MM-Attention.
            hard_attn_block_range: 
                Specifies the range of blocks in which Attribute Isolation Attention (AIA) is applied. Valid only when attention_mask_method == "hard.
            cond_scale_factor:
                Downsampling ratio of the condition image relative to the generated image.
            is_filter_cond_token: 
                If True, zero-value condition-image tokens are dropped before the sequence is fed to MM-Attention, reducing computation.  
            cond2image_attention_weight: 
                attention weight between condition token and image token. Lower values (<1.0) relax the spatial constraint, allowing sketch-like masks to guide generation.
        """
        device = self._execution_device
        
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        
        if isinstance(global_prompt, str):
            global_prompt = [global_prompt]
            
        if isinstance(regional_prompts, list) and isinstance(regional_prompts[0], str):
            regional_prompts = [regional_prompts]
        
        if isinstance(regional_labels,torch.Tensor):
            regional_labels = [regional_labels]
        
        batch_size = len(regional_prompts)
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            global_prompt,
            regional_prompts,
            regional_labels,
            height,
            width,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        )
        # Make sure height / width is divisible
        scale = self.vae_scale_factor * 2 * cond_scale_factor
        height = int(height) // scale * scale
        width = int(width) // scale * scale
        
        cond_height = height // cond_scale_factor
        cond_width = width // cond_scale_factor

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Prepare text embeddings
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None

            
        (
            prompt_embeds, 
            pooled_prompt_embeds,
            txt_seq_lens,
            text_ids,
        ) = self.encode_all_prompt(
            global_prompt=global_prompt,
            regional_prompts=regional_prompts,
            global_max_sequence_length=max_sequence_length,
            regional_max_sequence_length=regional_max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            lora_scale=lora_scale
        )
      
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_txt_seq_lens,
                negative_text_ids,
            ) = self.encode_all_prompt(
                global_prompt=[negative_prompt for _ in range(batch_size)],
                regional_prompts=[[negative_prompt]*len(regional_prompts[i]) for i in range(batch_size)],
                global_max_sequence_length=max_sequence_length,
                regional_max_sequence_length=regional_max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                lora_scale=lora_scale
            )

        
        # 3. Prepare condition image
        cond_hidden_states = None
        cond_ids= None
        if cond is not None:
            cond = self.prepare_image(
                image=cond,
                width=cond_width,
                height=cond_height,
                device=device,
                dtype=self.vae.dtype,
            )
            assert len(cond) == batch_size
  
            # vae encode condition
            cond_hidden_states = self.vae.encode(cond.to(self.vae.dtype)).latent_dist.sample()
            cond_hidden_states = (cond_hidden_states - self.vae.config.shift_factor) *  self.vae.config.scaling_factor
            cond_hidden_states = self._pack_latents(
                cond_hidden_states,
                batch_size=cond_hidden_states.shape[0],
                num_channels_latents=cond_hidden_states.shape[1],
                height=cond_hidden_states.shape[2],
                width=cond_hidden_states.shape[3],
            )


            cond_ids = self._prepare_latent_image_ids(
                cond.shape[0], 
                cond.shape[-2]//self.vae_scale_factor//2, 
                cond.shape[-1]//self.vae_scale_factor//2,
                device,
                prompt_embeds.dtype
            )
            cond_ids[...,1:] *= cond_scale_factor
        
        # discard cond tokens that are entirely composed of zero values
        cond_seq_lens = [0 for _ in range(batch_size)]
        pad_seq_lens = [0 for _ in range(batch_size)]
        if cond is not None:
            if is_filter_cond_token:
                cond_hidden_states, cond_ids, cond_seq_lens, pad_seq_lens = self.filter_cond_token(
                    cond,
                    cond_hidden_states,
                    cond_ids,
                    vae_scale_factor=self.vae_scale_factor * 2
                )
            else:
                cond_seq_lens = [hs.shape[0] for hs in cond_hidden_states]
        
        if cond is not None:
            cond_hidden_states = cond_hidden_states.repeat(num_images_per_prompt, 1, 1)
            cond_ids = cond_ids.repeat(num_images_per_prompt, 1, 1)
            
        cond_seq_lens = [item for item in cond_seq_lens for _ in range(num_images_per_prompt)]
        pad_seq_lens = [item for item in pad_seq_lens for _ in range(num_images_per_prompt)]
        
            
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1] 
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
                    
        # 6. Prepare attention masks            
        regional_labels = [label.clone().to(device) for label in regional_labels for _ in range(num_images_per_prompt)] # repeat labels
        attention_mask, hard_attention_mask = self.prepare_attention_mask(
            attention_mask_method=attention_mask_method,
            regional_labels=regional_labels,
            txt_seq_lens=txt_seq_lens,
            cond_seq_lens=cond_seq_lens,
            pad_seq_lens = pad_seq_lens,
            height=height//self.vae_scale_factor//2,
            width=width//self.vae_scale_factor//2,
            num_attention_heads=self.transformer.config.num_attention_heads,
            dtype=prompt_embeds.dtype,
            device=device,
            cond2image_attention_weight = cond2image_attention_weight
        )
     
        self._joint_attention_kwargs["attention_mask"] = attention_mask 
        self._joint_attention_kwargs["hard_attention_mask"] = hard_attention_mask
        
        if do_true_cfg:
            neg_joint_attention_kwargs = {}
            attention_mask, hard_attention_mask = self.prepare_attention_mask(
                attention_mask_method=attention_mask_method,
                regional_labels=regional_labels,
                txt_seq_lens=negative_txt_seq_lens,
                cond_seq_lens=cond_seq_lens,
                height=height//self.vae_scale_factor//2,
                width=width//self.vae_scale_factor//2,
                num_attention_heads=self.transformer.config.num_attention_heads,
                dtype=prompt_embeds.dtype,
                device=device,
                cond2image_attention_weight = cond2image_attention_weight,
                pad_seq_lens = pad_seq_lens,
            )
            neg_joint_attention_kwargs["attention_mask"] = attention_mask
            neg_joint_attention_kwargs["hard_attention_mask"] = hard_attention_mask

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                latent_model_input = latents
 
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    cond_hidden_states=cond_hidden_states,
                    hard_attn_block_range = hard_attn_block_range,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timestep / 1000,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    cond_ids=cond_ids,
                    guidance=guidance,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,   
                )[0]
                
                if do_true_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        cond_hidden_states=cond_hidden_states,
                        hard_attn_block_range = hard_attn_block_range,
                        encoder_hidden_states = negative_prompt_embeds,
                        pooled_projections = negative_pooled_prompt_embeds,
                        timestep = timestep / 1000,
                        img_ids = latent_image_ids,
                        txt_ids = negative_text_ids,
                        cond_ids = cond_ids,
                        guidance=guidance,
                        joint_attention_kwargs=neg_joint_attention_kwargs,
                        return_dict=False,   
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            output = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            output = self.vae.decode(latents, return_dict=False)[0]
            output = self.image_processor.postprocess(output, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output,)

        return FluxPipelineOutput(images=output)
