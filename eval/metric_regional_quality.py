import argparse
import json
import copy
import math
import cv2
import re  
import pandas as pd 
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager
import os
from collections import defaultdict
import numpy as np
import datetime
import random
from PIL import Image

def worker(args,queue,barrier):
    cuda_visible_index = [args.rank+args.num_replicas*i for i in range(args.tensor_parallel_size)]
    if os.environ.get("CUDA_VISIBLE_DEVICES",None) is not None:
        tmp = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        tmp = list(map(int,tmp))
        cuda_visible_devices = [tmp[idx] for idx in cuda_visible_index]
    else:
        cuda_visible_devices = cuda_visible_index    
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in cuda_visible_devices]) 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  
    # CUDA_VISIBLE_DEVICES must be set before importing torch. Otherwise, CUDA_VISIBLE_DEVICES will have no effect.
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from accelerate.utils import set_seed
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from dataset.no_pad_sampler import NonPadDistributedSampler
    from dataset.sacap_1m_dataset import SACap_1M_Dataset
    from dataset.collate_fn import collate_fn
    from utils.visualizer import save_image_with_caption,Visualizer
    from utils.utils import mask2box
    from .qwen_processor import QwenProcessor
    
    set_seed(42)

    processor = AutoProcessor.from_pretrained(args.model_id)
    qwen_processor = QwenProcessor(processor,
                               min_pixels=224*224,max_pixels=1280 * 28 * 28)
    
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1, "video": 1},
        gpu_memory_utilization = args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=int(1e8),  # Set to a large value so that the effective limit is solely determined by `max_model_len`.
        stop_token_ids=[],
    )

    dataset = SACap_1M_Dataset(
        image_root=args.image_root,
        seg_caption_path=args.seg_caption_path,
        resolution = args.resolution,
        cond_scale_factor = args.cond_scale_factor,
        is_group_bucket=False
    )

    sampler = NonPadDistributedSampler(
        dataset, num_replicas=args.num_replicas, rank=args.rank
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=True
    )
    
    visualizer = Visualizer()
    
    # skip the regional caption where `attribute` is not mentioned.
    skip_anno_ids = {
        k: list() for k in args.questions.keys()
    }
    p = os.path.join(args.cache_skip_anno_ids,f"skip_anno_ids.json")
    if hasattr(args,'pre_questions') and not os.path.exists(p):        
        for batch in tqdm(data_loader):
            llm_inputs = []
            for i in range(len(batch["image_name"])):
                for j,anno_id in enumerate(batch["anno_ids"][i]):
                    for key in args.pre_questions.keys():
                        short_caption = batch["short_regional_captions"][i][j]
                        caption = batch["regional_captions"][i][j]
                        prompt = args.pre_questions[key].format(short_caption=short_caption,caption=caption)
                        llm_input = qwen_processor.process([],prompt)
                        llm_input["question"] = prompt
                        llm_input["anno_id"] = anno_id
                        llm_inputs.append(llm_input)
            
            for i in range(0,len(llm_inputs),args.batch_size):
                outputs = llm.generate(llm_inputs[i:i+args.batch_size], sampling_params=sampling_params,use_tqdm=False)
                for j, output in enumerate(outputs):
                    output = output.outputs[0]
                    answer = output.text.strip()
                    llm_inputs[i+j]["answer"] = answer
                    
            assert len(llm_inputs) % len(args.pre_questions) == 0
            
            for i in range(0,len(llm_inputs),len(args.pre_questions)):
                for j,key in enumerate(args.pre_questions.keys()):
                    llm_input = llm_inputs[i+j]
                    
                    if "no" in llm_input["answer"].lower():
                        skip_anno_ids[key].append(llm_input["anno_id"])
                          
        p = os.path.join(args.cache_skip_anno_ids,f"skip_anno_ids_{args.rank}.json")
        with open(p,"w") as f:
            json.dump(skip_anno_ids,f)
        
        barrier.wait()
        if args.rank==0:
            gather_skip_anno_ids = {
                k: list() for k in args.questions.keys()
            }
            for i in range(args.num_replicas):
                p = os.path.join(args.cache_skip_anno_ids,f"skip_anno_ids_{i}.json")
                with open(p,"r") as f:
                    part_skip_anno_ids = json.load(f)
                    for k in part_skip_anno_ids:
                        gather_skip_anno_ids[k].extend(part_skip_anno_ids[k])
                        
            with open(os.path.join(args.cache_skip_anno_ids,"skip_anno_ids.json"),"w") as f:
                json.dump(gather_skip_anno_ids,f)
        barrier.wait()  
                
    if hasattr(args,'pre_questions') and os.path.exists(p):      
        with open(os.path.join(args.cache_skip_anno_ids,"skip_anno_ids.json"),"r") as f:
            skip_anno_ids = json.load(f)
        
    for k in skip_anno_ids:
        skip_anno_ids[k] = set(skip_anno_ids[k])
    # done
    
    result_dict = {
        "count":{k: 0 for k in args.questions.keys()},
        "score":{k: 0 for k in args.questions.keys()},
    }                
    vqa_records = {}
    for batch in tqdm(data_loader):
        llm_inputs = []
        for i in range(len(batch["image_name"])):
            image_name = batch["image_name"][i]
            gen_image_path = os.path.join(args.gen_img_dir,image_name)
            image = cv2.imread(gen_image_path, cv2.IMREAD_COLOR)
            anno_ids = batch["anno_ids"][i]
            for j,anno_id in enumerate(anno_ids):
                vqa_records[image_name] = [] 
                mask = (batch["label"][i][j]).cpu().numpy()
                                
                final_image = image.copy()
                
                # crop image
                x0, y0, x1, y1 = mask2box(mask)
                final_image = final_image[y0 : y1 + 1, x0 : x1 + 1]
                for key in args.questions.keys():
                    if anno_id in skip_anno_ids[key]:
                        continue
                    
                    short_caption = batch["short_regional_captions"][i][j]
                    caption = batch["regional_captions"][i][j]
                    prompt = args.questions[key].format(short_caption=short_caption,caption=caption)
                    
                    llm_input = qwen_processor.process([final_image],prompt)
                    llm_input["question"] = prompt
                    llm_input["mask"] = mask
                    llm_input["gen_image_path"] = gen_image_path
                    llm_input["attribute"] = key
                    llm_input["image_name"] = image_name  
                    llm_input["anno_id"] = anno_id
                    llm_inputs.append(llm_input)
        
        for i in range(0,len(llm_inputs),args.batch_size):
            outputs = llm.generate(llm_inputs[i:i+args.batch_size], sampling_params=sampling_params,use_tqdm=False)
            for j, output in enumerate(outputs):
                output = output.outputs[0]
                answer = output.text.strip()
                llm_inputs[i+j]["answer"] = answer
                
                image_name = llm_inputs[i+j]["image_name"]
                anno_id = llm_inputs[i+j]["anno_id"]
                attribute = llm_inputs[i+j]["attribute"]
                
                vqa_records[image_name].append({
                    "mask_id": anno_id,
                    "attribute": attribute,
                    "answer": answer
                })
        
        for i in range(0,len(llm_inputs)):
            llm_input = llm_inputs[i]
            attribute = llm_input["attribute"]
            score = 0.0
            if "yes" in llm_input["answer"].lower():
                score = 1.0
                
            result_dict["score"][attribute] += score
            result_dict["count"][attribute] += 1
            
            if args.save_image:
                # for debug
                gen_img = Image.open(llm_input["gen_image_path"]).convert('RGB')
                mask = torch.from_numpy(llm_input["mask"])
                mask = F.interpolate(mask[None,None,...].float(),size=(gen_img.size[1],gen_img.size[0]),mode='nearest-exact')
                mask = mask[0,0,...].long().numpy() # h,w
                
                gen_img = np.array(gen_img)
                gen_img = cv2.cvtColor(gen_img,cv2.COLOR_RGB2BGR)
                gen_img = visualizer.draw_binary_mask_with_number(gen_img,mask,alpha=0.4)
                gen_img = cv2.cvtColor(gen_img,cv2.COLOR_BGR2RGB)
    
                filename_without_extension, extension = os.path.splitext(os.path.basename(llm_input["gen_image_path"]))
                save_img_name = (
                    filename_without_extension +f"_{i}"+ f"_{attribute}_" + extension
                )
                save_image_with_caption(
                    llm_input["multi_modal_data"]["image"][0],
                    llm_input["question"]+"\n answer:"+llm_input["answer"],
                    os.path.join(args.output_dir, save_img_name),
                )
                
    queue.put(result_dict)
    return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_replicas", type=int,required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float,required=True)
    parser.add_argument("--batch_size", type=int,required=True)

    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2-VL-72B-Instruct-AWQ"
    )
    parser.add_argument("--seg_caption_path",type=str,required=True)
    parser.add_argument("--image_root",type=str,required=True)
    parser.add_argument("--cache_skip_anno_ids",type=str,required=True)
    
    parser.add_argument("--gen_img_dir",type=str,required=True)
    
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--resolution",type=int,required=True)
    parser.add_argument("--cond_scale_factor",type=int,required=True)
    
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--save_image", action="store_true", default=False)
    
    args = parser.parse_args()
    
    args.pre_questions = {
        'color': "Answer directly with only ['Yes','No']. For description: {caption}, analyze if there are explicit color attributes describing subject: {short_caption}. output 'Yes' if subject color is mentioned in the description; otherwise, output 'No'.",
        'texture': "Answer strictly with only ['Yes','No']. For description: {caption}, analyze if there are explicit texture attributes describing subject: {short_caption}. output 'Yes' if subject texture is mentioned in the description; otherwise, output 'No'.",
        'shape': "Answer strictly with only ['Yes','No']. For description: {caption}, analyze if there are explicit shape attributes describing subject: {short_caption}. output 'Yes' if subject shape is mentioned in the description; otherwise, output 'No'.",         
    }
    args.questions = {
        'spatial': "Answer directly with only ['Yes','No']. Is the subject {short_caption} present in image?",
        'color': "Answer directly with only ['Yes','No']. Is the color of subject {short_caption} consistent with the description: {caption}?",
        'texture': "Answer directly with only ['Yes','No']. Is the texture of subject {short_caption} consistent with the description: {caption}?",
        'shape': "Answer directly with only ['Yes','No']. Is the shape of subject {short_caption} consistent with the description: {caption}?",
    } 
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_skip_anno_ids, exist_ok=True)

    args_list = []
    for i in range(args.num_replicas):
        args_copy = copy.deepcopy(args)
        args_copy.rank = i
        args_list.append(args_copy)

    ctx = multiprocessing.get_context("spawn")
    manager = Manager()
    result_queue = manager.Queue()
    barrier = manager.Barrier(parties=args.num_replicas)
    
    process_list = []
    for i in range(args.num_replicas):
        process_list.append(
            ctx.Process(target=worker, args=(args_list[i],result_queue,barrier), daemon=False)
        )

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    # gather    
    total_result = {
        "count":{k: 0 for k in args.questions.keys()},
        "score":{k: 0 for k in args.questions.keys()},
    }
    
    results = []
    for _ in range(args.num_replicas):
        result = result_queue.get()
        for key in args.questions.keys():
            total_result["count"][key] +=result["count"][key]
            total_result["score"][key] +=result["score"][key]

    for key in args.questions.keys():
        total_result[f"{key}_accuracy"] = total_result["score"][key] / total_result["count"][key]

    with open(os.path.join(args.output_dir, "regional_quality.json"), "w") as f:
        json.dump(total_result, f)