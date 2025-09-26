import argparse
import random
import importlib
import torch
import pickle
import numpy as np
from panopticapi.utils import rgb2id
from PIL import Image

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    
    cls = get_obj_from_str(config["target"])
    return cls(**config.get("params", dict()))
    
def load_segm(segm_path):
    segmentation = np.array(
        Image.open(segm_path),
        dtype=np.uint8
    )
    segm_map = rgb2id(segmentation)

    return segm_map

def mask2box(mask):
    if isinstance(mask,torch.Tensor):
        ys, xs = torch.where(mask==1)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    else:
        ys, xs = np.where(mask==1)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    return x0, y0, x1, y1