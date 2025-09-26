import os
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import numpy as np
from pathlib import Path
import torch
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

from transformers import T5Tokenizer,T5TokenizerFast

from src.pipelines import FluxRegionalPipeline
from utils.utils import mask2box
from utils.visualizer import Visualizer


CLASSNAMES={
    1 : "person",
    2 : "bicycle",
    3 : "car",
    4 : "motorcycle",
    5 : "airplane",
    6 : "bus",
    7 : "train",
    8 : "truck",
    9 : "boat",
    10 : "traffic light",
    11 : "fire hydrant",
    12 : "street sign",
    13 : "stop sign",
    14 : "parking meter",
    15 : "bench",
    16 : "bird",
    17 : "cat",
    18 : "dog",
    19 : "horse",
    20 : "sheep",
    21 : "cow",
    22 : "elephant",
    23 : "bear",
    24 : "zebra",
    25 : "giraffe",
    26 : "hat",
    27 : "backpack",
    28 : "umbrella",
    29 : "shoe",
    30 : "eye glasses",
    31 : "handbag",
    32 : "tie",
    33 : "suitcase",
    34 : "frisbee",
    35 : "skis",
    36 : "snowboard",
    37 : "sports ball",
    38 : "kite",
    39 : "baseball bat",
    40 : "baseball glove",
    41 : "skateboard",
    42 : "surfboard",
    43 : "tennis racket",
    44 : "bottle",
    45 : "plate",
    46 : "wine glass",
    47 : "cup",
    48 : "fork",
    49 : "knife",
    50 : "spoon",
    51 : "bowl",
    52 : "banana",
    53 : "apple",
    54 : "sandwich",
    55 : "orange",
    56 : "broccoli",
    57 : "carrot",
    58 : "hot dog",
    59 : "pizza",
    60 : "donut",
    61 : "cake",
    62 : "chair",
    63 : "couch",
    64 : "potted plant",
    65 : "bed",
    66 : "mirror",
    67 : "dining table",
    68 : "window",
    69 : "desk",
    70 : "toilet",
    71 : "door",
    72 : "tv",
    73 : "laptop",
    74 : "mouse",
    75 : "remote",
    76 : "keyboard",
    77 : "cell phone",
    78 : "microwave",
    79 : "oven",
    80 : "toaster",
    81 : "sink",
    82 : "refrigerator",
    83 : "blender",
    84 : "book",
    85 : "clock",
    86 : "vase",
    87 : "scissors",
    88 : "teddy bear",
    89 : "hair drier",
    90 : "toothbrush",
    91 : "hair brush",
    92 : "banner",
    93 : "blanket",
    94 : "branch",
    95 : "bridge",
    96 : "building",
    97 : "bush",
    98 : "cabinet",
    99 : "cage",
    100 : "cardboard",
    101 : "carpet",
    102 : "ceiling",
    103 : "tile ceiling",
    104 : "cloth",
    105 : "clothes",
    106 : "clouds",
    107 : "counter",
    108 : "cupboard",
    109 : "curtain",
    110 : "desk",
    111 : "dirt",
    112 : "door",
    113 : "fence",
    114 : "marble floor",
    115 : "floor",
    116 : "stone floor",
    117 : "tile floor",
    118 : "wood floor",
    119 : "flower",
    120 : "fog",
    121 : "food",
    122 : "fruit",
    123 : "furniture",
    124 : "grass",
    125 : "gravel",
    126 : "ground",
    127 : "hill",
    128 : "house",
    129 : "leaves",
    130 : "light",
    131 : "mat",
    132 : "metal",
    133 : "mirror",
    134 : "moss",
    135 : "mountain",
    136 : "mud",
    137 : "napkin",
    138 : "net",
    139 : "paper",
    140 : "pavement",
    141 : "pillow",
    142 : "plant",
    143 : "plastic",
    144 : "platform",
    145 : "playingfield",
    146 : "railing",
    147 : "railroad",
    148 : "river",
    149 : "road",
    150 : "rock",
    151 : "roof",
    152 : "rug",
    153 : "salad",
    154 : "sand",
    155 : "sea",
    156 : "shelf",
    157 : "sky",
    158 : "skyscraper",
    159 : "snow",
    160 : "solid",
    161 : "stairs",
    162 : "stone",
    163 : "straw",
    164 : "structural",
    165 : "table",
    166 : "tent",
    167 : "textile",
    168 : "towel",
    169 : "tree",
    170 : "vegetable",
    171 : "brick wall",
    172 : "concrete wall",
    173 : "wall",
    174 : "panel wall",
    175 : "stone wall",
    176 : "tile wall",
    177 : "wood wall",
    178 : "water",
    179 : "waterdrops",
    180 : "blind window",
    181 : "window",
    182 : "wood",
}


class COCOStuffDataset(Dataset):
    def __init__(
        self,
        image_root,
        segm_root,
        is_group_bucket = False,
        cache_root=None,
        resolution:Union[List,int]=512,
        cond_scale_factor:int = 1,
    ):      
        super(COCOStuffDataset, self).__init__()
        self.image_root = image_root
        self.segm_root = segm_root
        self.cache_root = cache_root
        
        self.resolution = [resolution,resolution] if isinstance(resolution, int) else resolution
        self.cond_resolution = [self.resolution[0]//cond_scale_factor,self.resolution[1]//cond_scale_factor]
        
        if is_group_bucket:
            if self.cache_root is None:
                raise ValueError("Bucket grouping is enabled. Please specify a cache_root directory to store the bucket information.")
            os.makedirs(self.cache_root,exist_ok=True)
        
        self.data = []
        files = sorted([f for f in os.listdir(image_root) if f.endswith('.jpg')])
        for file in files:
            self.data.append(
                {
                    "seg": file.replace(".jpg", ".png"),
                    "image": file,
                }
            )
        
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
        return len(self.data)
    
    def _set_group_flag(self):
        # Related to `GroupSampler` in `group_sampler.py`.
        # group data by cond_seq_len and txt_seq_len values.
        
        print("=====COCOStuffDataset set group flag=====")
        cache_path = os.path.join(self.cache_root, f'{self.cond_resolution[0]}H_{self.cond_resolution[1]}W-group_bucket.parquet')
        if os.path.exists(cache_path): # use cache
            result_df = pd.read_parquet(cache_path)
        else:
            # Parallel run _compute_token_num function
            num_workers = 24
            chunk_size = max(1, len(self) // (num_workers * 4)) 
            
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
            item = self.data[idx]
            segm_file = item['seg']
            segm_path = os.path.join(self.segm_root, segm_file)
            segm_map = np.array(Image.open(segm_path))
            
            label = []
            label_id_list = np.unique(segm_map).tolist()
            
            txt_seq_len = 0
            
            for label_id in label_id_list:
                if label_id == 255:  # 255 is unlabel
                    continue
                mask = segm_map == label_id
                label.append(mask)
                
                class_name = CLASSNAMES[label_id+1]
                txt_seq_len += self.get_text_token_len(class_name)

            if len(label) > 0:
                label = np.stack(label, axis=0)
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
                
                valid_cond_token_num = FluxRegionalPipeline.get_valid_cond_token_num(cond_pixel_values)
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
        item = self.data[idx]
        image_name = item['image']
        segm_file = item['seg']
        
        image_path = os.path.join(self.image_root, image_name)
        segm_path = os.path.join(self.segm_root, segm_file)
        
        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size
        
        segm_map = np.array(Image.open(segm_path))

        global_caption = None 

        boxes = []
        cat_names = []
        label = []
        regional_captions = []
               
        label_id_list = np.unique(segm_map).tolist()
        
        for label_id in label_id_list:
            if label_id==255: # 255 is unlabel
                continue
            class_name = CLASSNAMES[label_id+1]
            
            mask = segm_map == label_id
            
            x0, y0, x1, y1 = mask2box(mask)
            box = np.array([
                x0 / img_w,
                y0 / img_h,
                x1 / img_w ,
                y1 / img_h ,
            ])
            boxes.append(box)
            label.append(mask)
            
            cat_names.append(class_name)
            regional_captions.append(class_name)
        
        if len(regional_captions)==0: # try again
            return self.__getitem__(np.random.randint(len(self)))
        label = np.stack(label, axis=0)
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
            colors=[(255,255,255),]*len(regional_captions)
        )
        cond_pixel_values = Image.fromarray(cond_pixel_values)
        cond_pixel_values = self.cond_transforms(cond_pixel_values)
        
        return {
            "label":label,
            "regional_captions":regional_captions,
            "global_caption":global_caption,
            "pixel_values":pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "image_name":image_name,
            "image_path":image_path,
            "segm_path":segm_path,
            "boxes":boxes
        }