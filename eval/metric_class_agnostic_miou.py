import torch
import torch.distributed as dist
from torch.nn import functional as F
import os
from tqdm import tqdm
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
from multiprocessing import Manager
import copy
import random
import argparse
from accelerate.utils import set_seed

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dataset.sacap_1m_dataset import SACap_1M_Dataset
from dataset.collate_fn import collate_fn
from dataset.no_pad_sampler import NonPadDistributedSampler

from utils.visualizer import Visualizer

def worker(args,queue):
    torch.cuda.set_device(args.rank)
    set_seed(42)
    
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, args.sam2_checkpoint))
    
    visualizer = Visualizer()
    
    val_dataset = SACap_1M_Dataset(
        image_root=args.image_root,
        seg_caption_path=args.seg_caption_path,
        resolution = args.resolution,
        cond_scale_factor = args.cond_scale_factor,
        is_group_bucket=False
    )
    
    data_sampler = NonPadDistributedSampler(
        val_dataset,
        num_replicas=args.num_replicas,
        rank=args.rank
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=data_sampler,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=4
    )
    score = 0 
    num_batches = 0
    
    for data in tqdm(val_dataloader, desc=f"Rank {args.rank}"):
        image_names = data['image_name']
        gen_img_paths = [os.path.join(args.gen_img_dir,name) for name in image_names]
        gen_imgs  = [Image.open(path).convert('RGB').resize([args.resolution,args.resolution],resample=Image.BICUBIC) for path in gen_img_paths]
        gen_imgs = [np.array(img) for img in gen_imgs]
        
        mask_input_batch = [] 
        box_batch = []
        point_coords_batch = []
        gts = []
        for label,boxes in zip(data['label'],data['boxes']):
            gt_label = label.cpu().numpy()
            resized_label = F.interpolate(label[None,...].float(),size=[256,256],mode='nearest-exact')
            resized_label = resized_label[0,...].long().cpu().numpy() # n,256,256
            
            region_num = len(gt_label)
            masks = []
            temp_boxes = []
            points = []
            
            for i in range(region_num):
                mask = resized_label[i:i+1] # 1,256,256
                masks.append(mask)
                
                box = boxes[i]*args.resolution
                temp_boxes.append(box)
                
                point  = None
                points.append(point)
                
            box_batch.append(temp_boxes)
            mask_input_batch.append(masks)
            gts.append(gt_label)
            point_coords_batch.append(points)
        
        for i,(img,masks,boxes) in enumerate(zip(gen_imgs,mask_input_batch,box_batch)):
            points = point_coords_batch[i]
            with torch.inference_mode():
                image_batch = [img] * len(boxes)
                predictor.set_image_batch(image_batch)

                masks_batch, scores_batch, _ = predictor.predict_batch(
                    point_coords_batch = None,
                    point_labels_batch = None,
                    box_batch=boxes,
                    mask_input_batch=masks,
                    multimask_output=False
                )
                
            masks_batch = [m[0].astype(np.bool_) for m in masks_batch] # list of (h,w)
            
            # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            # pred = visualizer.draw_binary_mask_with_number(img,masks_batch,alpha=0.4)
            # gt = visualizer.draw_binary_mask_with_number(img,gts[i],alpha=0.4)
            # cv2.imwrite(os.path.join('./debug',image_names[0].split('.')[0]+'_pred.jpg'),pred)
            # cv2.imwrite(os.path.join('./debug',image_names[0].split('.')[0]+'_gt.jpg'),gt)
            
            target = torch.from_numpy(gts[i]).long()
            preds = torch.from_numpy(np.stack(masks_batch,axis=0)).long()

            intersection = torch.sum(preds & target, dim=[1,2])
            target_sum = torch.sum(target, dim=[1,2])
            pred_sum = torch.sum(preds, dim=[1,2])
            union = target_sum + pred_sum - intersection
            iou = torch.where(union != 0, intersection / union, 0)
            
            score += torch.sum(iou)
            num_batches += len(masks_batch)

    queue.put((score,num_batches))
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--image_root",
        type=str,
        default='./data/SACap-1M/test',
    )    
    parser.add_argument(
        "--seg_caption_path",
        type=str,
        default='./data/SACap-1M/annotations/anno_test.parquet',
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default='./ckpt/sam2/sam2.1_hiera_large.pt',
    )
    parser.add_argument(
        "--gen_img_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True
    )
    parser.add_argument(
        "--cond_scale_factor",
        type=int,
        required=True
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)  
    
    if args.num_replicas == 1:
        args.rank = 0
        score,num_batches = worker(args)
    else:
        args_list = []
        for i in range(args.num_replicas):
            args_copy = copy.deepcopy(args)
            args_copy.rank = i
            args_list.append(args_copy)

        ctx = multiprocessing.get_context("spawn")
        
        result_queue = Manager().Queue()    
        
        process_list = []
        for i in range(args.num_replicas):
            process_list.append(
                ctx.Process(target=worker, args=(args_list[i],result_queue), daemon=False)
            )

        for process in process_list:
            process.start()
        for process in process_list:
            process.join()
        
        score,num_batches=0,0
        while not result_queue.empty():
            score_,num_batches_ = result_queue.get()
            print(score_,num_batches_)
            score += score_
            num_batches += num_batches_

    average_iou = (score/num_batches).item()
    print(f"Average IoU across all GPUs: {average_iou}")
    with open(os.path.join(args.output_dir,"miou.txt"), "a") as f:
        f.write(f"miou: {round(average_iou, 4)}\n")