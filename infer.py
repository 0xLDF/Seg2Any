import torch
import os
import argparse
from tqdm import tqdm
import cv2 
import numpy as np
import time
import json
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F

from src.models import FluxTransformer2DModel
from src.pipelines import FluxRegionalPipeline

from utils.visualizer import Visualizer

visualizer = Visualizer()

def load_seg_map_and_prompt(seg_map_path, seg_anno_path, cond_scale_factor):
    
    seg_map = Image.open(seg_map_path).convert('RGB')
    img_w, img_h = seg_map.size
    seg_map = np.array(seg_map)
    
    with open(seg_anno_path, 'r') as f:
        seg_anno = json.load(f)

    s = cond_scale_factor * 16
    cond_resolution = [img_h // s * 16, img_w // s * 16]
            
    cond_transforms = transforms.Compose(
        [
            transforms.Resize(cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    global_caption = seg_anno['caption'] 
    seed = seg_anno.get("seed",None)
    
    new_segm_info = {}
    for region in seg_anno['segments_info']:
        new_segm_info[tuple(region['color'])] = region['text']
    seg_anno = new_segm_info
    
    label = []
    regional_captions = []
    
    color_list = np.unique(seg_map.reshape(-1,3), axis=0)
    color_list = [tuple(color.tolist()) for color in color_list]
    
    for color in color_list:
        if color not in seg_anno:
            continue
        
        regional_caption = seg_anno[color]
        mask = (seg_map[...,0] == color[0]) & (seg_map[...,1] == color[1]) & (seg_map[...,2] == color[2])
        label.append(mask)
        regional_captions.append(regional_caption)
    
    label = np.stack(label, axis=0) # n,h,w
    label = torch.from_numpy(label).long()

    cond_pixel_values = np.zeros([label.shape[-2],label.shape[-1],3],dtype=np.uint8)
    cond_pixel_values = visualizer.draw_contours(
        cond_pixel_values,
        label.cpu().numpy(),
        thickness=1,
        colors=[(255,255,255),]*len(regional_captions)
    )
    cond_pixel_values = Image.fromarray(cond_pixel_values)
    cond_pixel_values = cond_transforms(cond_pixel_values)
    
    return {
        "label":label,
        "regional_captions": regional_captions,
        "global_caption": global_caption,
        "cond_pixel_values": cond_pixel_values,
        "image_name": os.path.basename(seg_map_path),
        "image_width": cond_resolution[1]*cond_scale_factor,
        "image_height": cond_resolution[0]*cond_scale_factor,
        "seed": seed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev"
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--seg_mask_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=r'./result'
    )
    
    parser.add_argument(
        "--weight_dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16"
    )
    parser.add_argument(
        "--cond_scale_factor",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cond2image_attention_weight",
        type=float,
        default=1,
        help="attention weight between condition token and image token. Lower values (<1.0) relax the spatial constraint, allowing sketch-like masks to guide generation."
    )
    parser.add_argument(
        "--regional_max_sequence_length",
        type=int,
        default=50
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=32
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5
    )

    args = parser.parse_args()
                
    os.makedirs(args.output_path, exist_ok=True)
    
    visualizer = Visualizer()
    
    weight_dtype = torch.float32
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",torch_dtype=weight_dtype
    )
            
    pipeline = FluxRegionalPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )

    pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path,'default'), adapter_name="default")
    pipeline.set_adapters("default")
    if os.path.exists(os.path.join(args.lora_ckpt_path,'cond')):
        pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path,'cond'), adapter_name="cond")
        pipeline.set_adapters(['cond','default'])
    

    pipeline = pipeline.to('cuda')
    
    anno_files = sorted([os.path.join(args.seg_mask_path,f) for f in os.listdir(args.seg_mask_path) if f.endswith('.json')])
                
    for anno_file in anno_files:
        batch = load_seg_map_and_prompt(
            seg_map_path=anno_file.replace(".json",".png"), 
            seg_anno_path=anno_file, 
            cond_scale_factor=args.cond_scale_factor
        )
        
        image_name = batch["image_name"]

        if batch["seed"] is not None:
            generator = [torch.Generator("cuda").manual_seed(batch["seed"]+i) for i in range(args.num_images_per_prompt)]
        else:
            generator = None
        
        images = pipeline(
            global_prompt = batch["global_caption"],
            regional_prompts = batch["regional_captions"],
            regional_labels = batch["label"],
            cond = (batch["cond_pixel_values"]+1)/2.0, # denormalize
            attention_mask_method = "hard",
            is_filter_cond_token=True,
            cond2image_attention_weight=args.cond2image_attention_weight,
            hard_attn_block_range = [19,37],
            height = batch["image_height"],
            width = batch["image_width"], 
            cond_scale_factor = args.cond_scale_factor,
            num_images_per_prompt = args.num_images_per_prompt,
            guidance_scale = args.guidance_scale,
            num_inference_steps = args.num_inference_steps,
            generator = generator,
            max_sequence_length = args.max_sequence_length,
            regional_max_sequence_length = args.regional_max_sequence_length,
        ).images
        
        for ids, image in enumerate(images):
            save_image_path = os.path.join(args.output_path, image_name.split('.')[0]+f'_{ids}.jpg')
            image = np.array(image)
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_image_path,image)
