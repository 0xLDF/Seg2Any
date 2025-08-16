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
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from transformers import T5Tokenizer,T5TokenizerFast

from src.pipelines import FluxRegionalPipeline
from utils.utils import mask2box
from utils.visualizer import Visualizer


CLASSNAMES={
    1:'wall',
    2:'building edifice',
    3:'sky',
    4:'floor',
    5:'tree',
    6:'ceiling',
    7:'road',
    8:'bed',
    9:'windowpane',
    10:'grass',
    11:'cabinet',
    12:'sidewalk',
    13:'person',
    14:'earth ground',
    15:'door',
    16:'table',
    17:'mountain',
    18:'plant flora',
    19:'curtain,drapery,mantle',
    20:'chair',
    21:'car',
    22:'water',
    23:'painting,picture',
    24:'sofa,lounge',
    25:'shelf',
    26:'house',
    27:'sea',
    28:'mirror',
    29:'carpet',
    30:'field',
    31:'armchair',
    32:'seat',
    33:'fence',
    34:'desk',
    35:'rock,stone',
    36:'wardrobe,closet',
    37:'lamp',
    38:'bathtub',
    39:'railing',
    40:'cushion',
    41:'base,pedestal,stand',
    42:'box',
    43:'pillar',
    44:'signboard,sign',
    45:'chest,bureau,dresser',
    46:'counter',
    47:'sand',
    48:'sink',
    49:'skyscraper',
    50:'fireplace',
    51:'refrigerator',
    52:'grandstand',
    53:'path',
    54:'stairs',
    55:'runway',
    56:'showcase,vitrine',
    57:'pool table,billiard table',
    58:'pillow',
    59:'screen door',
    60:'stairway',
    61:'river',
    62:'bridge',
    63:'bookcase',
    64:'blind screen',
    65:'coffee table,cocktail table',
    66:'toilet,can,commode',
    67:'flower',
    68:'book',
    69:'hill',
    70:'bench',
    71:'countertop',
    72:'cooking stove',
    73:'palm tree',
    74:'kitchen island',
    75:'computer',
    76:'swivel chair',
    77:'boat',
    78:'bar',
    79:'arcade machine',
    80:'hovel,shack',
    81:'autobus,motorbus,omnibus',
    82:'towel',
    83:'light',
    84:'truck',
    85:'tower',
    86:'chandelier,pendant',
    87:'awning,sunblind',
    88:'streetlight',
    89:'booth,cubicle',
    90:'television,tv,telly',
    91:'airplane',
    92:'dirt track',
    93:'apparel',
    94:'pole',
    95:'land,ground,soil',
    96:'balustrade,handrail',
    97:'escalator',
    98:'pouf,hassock',
    99:'bottle',
    100:'buffet,counter,sideboard',
    101:'poster,placard,notice card',
    102:'stage',
    103:'van',
    104:'ship',
    105:'fountain',
    106:'conveyor,belt,transporter',
    107:'canopy',
    108:'washing machine',
    109:'toy',
    110:'swimming pool,natatorium',
    111:'stool',
    112:'barrel',
    113:'basket,handbasket',
    114:'waterfall',
    115:'tent',
    116:'bag',
    117:'motorbike',
    118:'cradle',
    119:'oven',
    120:'ball',
    121:'food',
    122:'stair',
    123:'storage tank',
    124:'brand marque',
    125:'microwave oven',
    126:'flowerpot',
    127:'animal fauna',
    128:'bicycle',
    129:'lake',
    130:'dishwasher',
    131:'screen,silver screen',
    132:'blanket',
    133:'sculpture',
    134:'exhaust hood',
    135:'sconce',
    136:'vase',
    137:'traffic light',
    138:'tray',
    139:'ashcan,trash can',
    140:'fan',
    141:'pier,wharfage',
    142:'crt screen',
    143:'plate',
    144:'monitoring device',
    145:'notice board',
    146:'shower',
    147:'radiator',
    148:'drinking glass',
    149:'clock',
    150:'flag'
}


class ADE20KDataset(Dataset):
    def __init__(
        self,
        image_root,
        segm_root,
        is_group_bucket = False,
        cache_root=None,
        resolution:Union[List,int]=512,
        cond_scale_factor:int = 1,
    ):                
        super(ADE20KDataset, self).__init__()
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
        segm_files = sorted([f for f in os.listdir(segm_root) if f.endswith('.png')])
        for segm_file in segm_files:
            self.data.append(
                {
                    "seg": segm_file,
                    "image": segm_file.replace(".png", ".jpg"),
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
        
        print("=====ADE20KDataset set group flag=====")
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
                if label_id == 0:  # 0 is unlabel
                    continue
                mask = segm_map == label_id
                label.append(mask)
                
                class_name = CLASSNAMES[label_id]
                txt_seq_len += self.get_text_token_len(class_name)     
            
            if len(label) > 0:
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
            if label_id==0: # 0 is unlabel
                continue
            class_name = CLASSNAMES[label_id]
            
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
            colors=[(255,255,255),]*len(regional_captions)
        )
        cond_pixel_values = Image.fromarray(cond_pixel_values)
        cond_pixel_values = self.cond_transforms(cond_pixel_values)
                
        return {
            "label": label,
            "regional_captions": regional_captions,
            "global_caption": global_caption,
            "pixel_values": pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "image_name": image_name,
            "image_path": image_path,
            "boxes": boxes
        }
