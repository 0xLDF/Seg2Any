import os
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import json
import numpy as np
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import pycocotools.mask as mask_util
from joblib import Parallel, delayed

from torch.utils.data import Dataset
import torch

from transformers import T5Tokenizer,T5TokenizerFast

from src.pipelines import FluxRegionalPipeline
from utils.utils import mask2box
from utils.visualizer import Visualizer


class SACap_1M_Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        seg_caption_path,
        image_root,
        is_group_bucket = False,
        cache_root=None,
        resolution:Union[List,int]=1024,
        cond_scale_factor:int =2,
    ):        
        super(SACap_1M_Dataset, self).__init__()
        self.image_root = image_root
        self.seg_caption_path = seg_caption_path
        self.cache_root = cache_root
        
        self.resolution = [resolution,resolution] if isinstance(resolution, int) else resolution
        self.cond_resolution = [self.resolution[0]//cond_scale_factor,self.resolution[1]//cond_scale_factor]
        
        if is_group_bucket:
            if self.cache_root is None:
                raise ValueError("Bucket grouping is enabled. Please specify a cache_root directory to store the bucket information.")
            os.makedirs(self.cache_root,exist_ok=True)

        self.images_info = pd.read_parquet(seg_caption_path)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transforms = transforms.Compose(
            [
                transforms.Resize(self.cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.visualizer = Visualizer()
        
        if is_group_bucket:
            self.tokenizer = T5TokenizerFast.from_pretrained('google/t5-v1_1-xxl')
            self._set_group_flag()
        else:
            self.flag = np.full([len(self),], fill_value=0, dtype=np.int64)
                   
    def __len__(self):
        return len(self.images_info)
    
    def _set_group_flag(self):
        # Related to `GroupSampler` in `group_sampler.py`.
        # group data by cond_seq_len and txt_seq_len values.
        
        print("=====SACap_1M_Dataset set group flag=====")  
        cache_path = os.path.join(self.cache_root, f'{self.cond_resolution[0]}H_{self.cond_resolution[1]}W-group_bucket.parquet')
        if os.path.exists(cache_path): # use cache
            result_df = pd.read_parquet(cache_path)
        else:
            # Parallel run _compute_token_num function
            num_workers = 24
            chunk_size = max(1, len(self) // (num_workers * 100)) 
            
            ids = list(range(len(self)))
            chunks = [ids[i:i+chunk_size] for i in range(0, len(self), chunk_size)]
            
            results = Parallel(n_jobs=num_workers, verbose=10)(
                delayed(self._compute_token_num)(chunk) 
                for chunk in chunks
            )
            save_data = {'cond_seq_len':[],'txt_seq_len':[]}
            for chunk in results:
                for k in chunk:
                    save_data[k].extend(chunk[k])
            
            # save
            result_df = pd.DataFrame(save_data)
            result_df.to_parquet(cache_path, index=False)
                
        assert len(result_df) == len(self)
        
        flags = []
        for index, row in result_df.iterrows():
            cond_seq_len = row['cond_seq_len']
            txt_seq_len = row['txt_seq_len']
            flag = cond_seq_len // 50 + txt_seq_len // 50 * 1e6 
            # Bins seq_len into 50-unit buckets (//50), to create approximately equal-sized groups while allowing ±50 variation.
            # with 1e6 shift `txt_seq_len` into the higher bits, ensuring the two terms remain non-overlapping.  

            flags.append(flag)
            
        self.flag = np.array(flags, dtype=np.int64)       

    def _compute_token_num(self, idxs):
        # get valid condition token num after filtering out zero-value condition tokens, and obtain text token num via tokenizer.
        
        chunk_results = {'cond_seq_len':[],'txt_seq_len':[]}
        for idx in idxs:
            img_info = self.images_info.iloc[idx]
            image_name = img_info['imagename']
            image_group = img_info.get('image_group', '') 
            image_path = os.path.join(self.image_root,image_group,image_name)
            anno_path = image_path[: image_path.rfind(".")] + ".json"
            
            with open(anno_path, "r", encoding="utf-8") as file:
                anns = json.load(file)
            
            img_w = anns["image"]["width"]
            img_h = anns["image"]["height"]
            
            segments_info = {seg_info['anno_id']:seg_info for seg_info in img_info['segments_info']}
            
            label = []
            
            txt_seq_len = 0
            
            global_caption = img_info['caption']
            txt_seq_len += self.get_text_token_len(global_caption)
        
            for seg in anns["annotations"]:
                if seg['id'] in segments_info:
                    mask = mask_util.decode(seg["segmentation"])==1
                    label.append(mask)
                    
                    regional_caption = segments_info[seg['id']]["caption"]
                    txt_seq_len += self.get_text_token_len(regional_caption)
            
            if len(label) != 0:
                label = np.stack(label, axis=0) # n,h,w
                label = torch.from_numpy(label)
                label = label[None, ...]
                label = F.interpolate(label.float(), size=self.resolution, mode='nearest-exact')
                label = label[0, ...].long()  # n,h,w
                
                cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
                cond_pixel_values = self.visualizer.draw_contours(
                    cond_pixel_values,
                    label.cpu().numpy(),
                    thickness=1,
                    colors=[(255, 255, 255), ] * len(label)
                )
                cond_pixel_values = Image.fromarray(cond_pixel_values)
                cond_pixel_values = self.cond_transforms(cond_pixel_values)
                
                valid_cond_token_num =  FluxRegionalPipeline.get_valid_cond_token_num(cond_pixel_values)
            else:
                valid_cond_token_num = 0
            
            chunk_results['cond_seq_len'].append(valid_cond_token_num)
            chunk_results['txt_seq_len'].append(txt_seq_len)
        return chunk_results  
    
    def get_text_token_len(self,text):
        input_ids = self.tokenizer(
            text,
            padding="longest",
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        return input_ids.shape[-1]  

    def __getitem__(self, idx):
        img_info = self.images_info.iloc[idx]
        image_name = img_info['imagename']
        image_group = img_info.get('image_group', '') 
        image_path = os.path.join(self.image_root,image_group,image_name)
        anno_path = image_path[: image_path.rfind(".")] + ".json"
    
        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size

        with open(anno_path, "r", encoding="utf-8") as file:
            anns = json.load(file)

        global_caption = img_info['caption']
        segments_info = {seg_info['anno_id']:seg_info for seg_info in img_info['segments_info']}

        label = []
        boxes = []
        regional_captions = []
        short_regional_captions = []
        anno_ids = []
        
        for seg in anns["annotations"]:
            if seg['id'] in segments_info:
                mask = mask_util.decode(seg["segmentation"])==1
                x0, y0, x1, y1 = mask2box(mask)
                box = np.array([
                    x0 / img_w,
                    y0 / img_h,
                    x1 / img_w ,
                    y1 / img_h ,
                ])
                boxes.append(box)
                anno_ids.append(seg['id'])
                label.append(mask)
                
                regional_captions.append(segments_info[seg['id']]["caption"])
                if "short_caption" in segments_info[seg['id']]:
                    short_regional_captions.append(segments_info[seg['id']]["short_caption"])

        if len(regional_captions)==0: # try again
            return self.__getitem__(np.random.randint(len(self)))
        
        label = np.stack(label, axis=0) # n,h,w
        label = torch.from_numpy(label)
        label = label[None,...]
        label = F.interpolate(label.float(), size=self.resolution, mode='nearest-exact')
        label = label[0,...].long() # n,h,w
        
        pixel_values = self.image_transforms(image) # c,h,w

        cond_pixel_values = np.zeros([label.shape[-2],label.shape[-1],3],dtype=np.uint8)
        cond_pixel_values = self.visualizer.draw_contours(
            cond_pixel_values,
            label.cpu().numpy(),
            thickness=1,
            colors=[(255,255,255),]*len(regional_captions) # 黑底白边
        )
        cond_pixel_values = Image.fromarray(cond_pixel_values)
        cond_pixel_values = self.cond_transforms(cond_pixel_values)
            
        return_dict =  {
            "label":label,
            "regional_captions":regional_captions,
            "global_caption":global_caption,
            "pixel_values":pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "image_name":image_name,
            "image_path":image_path,
            "boxes":boxes,
            "anno_ids":anno_ids
        }
        
        if len(short_regional_captions)>0:
            # only used in test
            return_dict["short_regional_captions"] = short_regional_captions

        return return_dict