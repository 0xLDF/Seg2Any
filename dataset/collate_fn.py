import torch
import numpy as np

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
    cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    global_caption = [example["global_caption"] for example in examples]
    global_caption = None if None in global_caption else global_caption
        
    return_dict = {
        "pixel_values": pixel_values,
        "cond_pixel_values": cond_pixel_values,
        "global_caption": global_caption,
        **{k: [e[k] for e in examples] for k in examples[0].keys() 
           if k not in ["pixel_values", "cond_pixel_values", "global_caption"]}
    }
    
    return return_dict