import os
import argparse
import copy
import torch
import numpy as np
from pathlib import Path
import time
from transformers import AutoProcessor, AutoModel
import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from collections import defaultdict
from PIL import Image
from functools import partial
from torch.nn import functional as F
from peft.utils import get_peft_model_state_dict
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

from omegaconf import OmegaConf,DictConfig
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from cleanfid import fid as clean_fid
import ImageReward as RM

from src.models import FluxTransformer2DModel
from src.pipelines import FluxRegionalPipeline
from dataset.collate_fn import collate_fn
from dataset.no_pad_sampler import NonPadDistributedSampler
from utils.utils import instantiate_from_config

def load_img_and_convert_tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = torch.from_numpy(img).permute(2,0,1) # c,h,w
    return img

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    
    return parser

def main(args):    
    
    # logging_dir = Path(args.project.output_dir, args.project.logging_dir)
    # accelerator_project_config = ProjectConfiguration(project_dir=args.project.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.trainer.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    gen_image_dir = Path(args.project.gen_image_dir)

    
    if accelerator.is_main_process:
        # os.makedirs(args.project.output_dir, exist_ok=True)
        os.makedirs(gen_image_dir, exist_ok=True)
            
        for i in range(args.eval.num_images_per_prompt):
            os.makedirs(os.path.join(gen_image_dir,f'group_{i}'), exist_ok=True)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    
    pipeline = FluxRegionalPipeline.from_pretrained(
        args.model.pretrained_model_name_or_path,
        transformer=flux_transformer,
        torch_dtype=weight_dtype,
    )
    
    pipeline.load_lora_weights(os.path.join(args.resume_from_checkpoint,'default'), adapter_name="default")
    pipeline.set_adapters("default")
    if os.path.exists(os.path.join(args.resume_from_checkpoint,'cond')):
        pipeline.load_lora_weights(os.path.join(args.resume_from_checkpoint,'cond'), adapter_name="cond")
        pipeline.set_adapters(['cond','default'])

    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device)
        
    val_dataset = instantiate_from_config(args.data.val)
    
    data_sampler = NonPadDistributedSampler(
        val_dataset,
        num_replicas=accelerator.num_processes,
        rank = accelerator.process_index
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        sampler=data_sampler
    )
    
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    print("###### generate eval image ######")
    for batch in tqdm(val_dataloader):
        image_name = batch["image_name"][0]
        save_image_path = os.path.join(gen_image_dir,f'group_{args.eval.num_images_per_prompt-1}', image_name)
        if os.path.exists(save_image_path):
            continue
                 
        gen_images = pipeline(
            global_prompt = batch["global_caption"],
            regional_prompts = batch["regional_captions"],
            regional_labels = batch["label"],
            cond = (batch["cond_pixel_values"]+1)/2.0 if args.model.is_use_cond_token else None, # denormalize
            attention_mask_method = args.model.attention_mask_method,
            is_filter_cond_token=args.model.is_filter_cond_token,
            hard_attn_block_range = args.model.hard_attn_block_range,
            height = batch["pixel_values"].shape[-2],
            width = batch["pixel_values"].shape[-1],
            cond_scale_factor = args.cond_scale_factor,
            num_images_per_prompt = args.eval.num_images_per_prompt,
            guidance_scale = args.eval.guidance_scale,
            num_inference_steps = args.model.num_inference_steps,
            generator = generator,
            max_sequence_length = args.model.max_sequence_length,
            regional_max_sequence_length = args.model.regional_max_sequence_length
        ).images
        for i,image in enumerate(gen_images):
            save_image_path = os.path.join(gen_image_dir, f'group_{i}', image_name)
            image.save(save_image_path)
                
    accelerator.wait_for_everyone()    
    del pipeline
    del flux_transformer

    print("###### compute FID ######") 
    if accelerator.is_main_process:
        fid_score = 0.0
        for i in range(args.eval.num_images_per_prompt):
            fid_score += clean_fid.compute_fid(
                args.data.val.params.image_root,
                os.path.join(gen_image_dir, f'group_{i}'),
                dataset_res=args.resolution,
                batch_size=128
            )
        fid_score /= args.eval.num_images_per_prompt
        
        print("# --------------------------------- #")
        print(f"FID: {fid_score}")
        print("# --------------------------------- #")
        with open(os.path.join(os.path.dirname(gen_image_dir),"global_quality.txt"), "a") as f:
            f.write(f"FID: {round(fid_score, 4)}\n")
    
    if batch["global_caption"] is None:
        return
    
    accelerator.wait_for_everyone()
    
    print("###### compute clip score ######")
    clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    clip_score_metric = clip_score_metric.to(accelerator.device)
    for batch in tqdm(val_dataloader):
        tensor_images = []
        for image_name in batch["image_name"]:
            tensor_image = load_img_and_convert_tensor(os.path.join(gen_image_dir,'group_0', image_name))
            tensor_images.append(tensor_image)
        tensor_images = torch.stack(tensor_images,dim=0)
        tensor_images = tensor_images.to(accelerator.device)
        clip_score_metric.update(tensor_images, batch["global_caption"])

    clip_score = clip_score_metric.compute().item()
    if accelerator.is_main_process:
        print("# --------------------------------- #")
        print(f"CLIP score (avg): {clip_score}")
        print("# --------------------------------- #")
        with open(os.path.join(os.path.dirname(gen_image_dir),"global_quality.txt"), "a") as f:
            f.write(f"CLIP score (avg): {round(clip_score, 4)}\n")

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    unknown = [s.lstrip('-') for s in unknown]
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    print('###### cli input evaluation setup:  ######\n',cli)
    config = OmegaConf.merge(*configs, cli)
    
    if config.resolution % (16 * config.cond_scale_factor) != 0:
        raise ValueError(
            f"Image resolution {config.resolution} must be divisible by {16 * config.cond_scale_factor} "
            f"(16 * cond_scale_factor) to ensure proper feature map alignment in the model. "
            f"Please adjust either the resolution or cond_scale_factor."
        )
    
    main(config)