import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from textwrap import wrap
import warnings

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

CV2_FONT=cv2.FONT_HERSHEY_DUPLEX


class Visualizer:
    def __init__(self):
        
        css4_colors = mcolors.CSS4_COLORS
         
        self.bgr_colors = [[int(x * 255) for x in reversed(mcolors.to_rgb(color))] for color in css4_colors.values()] 
    
    def draw_contours(self,image,label,thickness,colors=None):
        # label np.ndarray [num_mask,h,w] or list[np.ndarray] 
        # image [h,w,3]
        if isinstance(label,np.ndarray):
            label = [label[i] for i in range(len(label))]
            
        image = image.copy()
        num_mask = len(label)
        
        if colors is None:
            colors = random.choices(self.bgr_colors, k=num_mask)
        
        for i in range(num_mask):
            mask = label[i].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, color=colors[i], thickness=thickness)
                        
        return image
    
    def draw_binary_mask(self,image,label,alpha,thickness,colors=None):
        # label np.ndarray [num_mask,h,w] or list[np.ndarray] 
        # image [h,w,3]
        if isinstance(label,np.ndarray):
            label = [label[i] for i in range(len(label))]
        
        image = image.copy()
        num_mask = len(label)
        
        if colors is None:
            colors = random.choices(self.bgr_colors, k=num_mask)
        
        overlay_image = np.zeros_like(image)
        overlay_mask = np.zeros(image.shape[:2],dtype=np.bool_)
        for i in range(num_mask):
            mask = label[i].astype(np.uint8)
            overlay_mask[mask==1] = 1
            
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, color=colors[i], thickness=thickness)  
            
            overlay_image[mask == 1] = colors[i]
            
        weight1 = np.ones(image.shape[:2],dtype=np.float32)
        weight2 = np.zeros(image.shape[:2],dtype=np.float32)
        weight2[overlay_mask==1] = alpha
        image = cv2.blendLinear(image,overlay_image, weight1,weight2)
        
        return image
    
    
    def draw_text(self, image, text, center_coord, color, color_bg, text_scale, text_thickness, text_padding):
        center_x, center_y = center_coord

        lines = text.split('\n')

        # get each line width and height
        line_heights = []
        line_widths = []
        for line in lines:
            (line_w, line_h), _ = cv2.getTextSize(
                text=line,
                fontFace=CV2_FONT,
                fontScale=text_scale,
                thickness=text_thickness,
            )
            line_widths.append(line_w)
            line_heights.append(line_h)

        total_width = max(line_widths)  
        total_height = sum(line_heights) 

        text_bg_xyxy = [
            center_x - (total_width // 2) - text_padding,
            center_y - (total_height // 2) - text_padding,
            center_x + (total_width // 2) + text_padding,
            center_y + (total_height // 2) + text_padding,
        ]

        if color_bg is not None:
            cv2.rectangle(
                img=image,
                pt1=(text_bg_xyxy[0], text_bg_xyxy[1]),
                pt2=(text_bg_xyxy[2], text_bg_xyxy[3]),
                color=color_bg,
                thickness=-1,
            )

        # draw each line text
        y_start = text_bg_xyxy[1] + text_padding
        for i, line in enumerate(lines):
            (line_w, line_h), _ = cv2.getTextSize(
                text=line,
                fontFace=CV2_FONT,
                fontScale=text_scale,
                thickness=text_thickness,
            )
            x_start = center_x - (line_w // 2)
            y_start += line_h  # y increase

            cv2.putText(
                img=image,
                text=line,
                org=(x_start, y_start),
                fontFace=CV2_FONT,
                fontScale=text_scale,
                color=color,
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )

        return image
        
    def find_mask_center_coord(self,binary_mask):
        binary_mask = binary_mask.astype(np.uint8)
        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]
        
        return  (coords_x[len(coords_x)//2], coords_y[len(coords_y)//2] )
        
    def draw_binary_mask_with_number(self,image,label,alpha,numbers=None,contour_thickness=1,text_scale=0.5,text_thickness=1,text_padding=2,colors=None):
        # label np.ndarray [num_mask,h,w] or list[np.ndarray] 
        # image [h,w,3]
        if isinstance(label,np.ndarray):
            if label.ndim == 3:
                label = [label[i] for i in range(len(label))]
            elif label.ndim == 2:
                label = [label]

        num_mask = len(label)
        
        if colors is None:
            colors = random.choices(self.bgr_colors, k=num_mask)
        if numbers is None:
            numbers = [str(i) for i in range(num_mask)]
            
        image = self.draw_binary_mask(image,label,alpha,contour_thickness,colors)        
        
        for i in range(num_mask):
            mask = label[i]
            center_coord = self.find_mask_center_coord(mask)
            
            self.draw_text(image,numbers[i], center_coord, color=(255,255,255),color_bg=(0,0,0),text_scale=text_scale,text_thickness=text_thickness,text_padding=text_padding)
            
        return image

    def draw_binary_mask_with_caption(self, image, label, captions, alpha, contour_thickness=1, text_scale=0.5, text_thickness=1, text_padding=2, colors=None, max_line_length=20):
        # captions: List of strings, each string is a caption for the corresponding mask
        # label np.ndarray [num_mask,h,w] or list[np.ndarray] 
        # image [h,w,3]
        if isinstance(label,np.ndarray):
            label = [label[i] for i in range(len(label))]

        num_mask = len(label)
        
        if colors is None:
            colors = random.choices(self.bgr_colors, k=num_mask)
            
        image = self.draw_binary_mask(image, label, alpha, contour_thickness, colors)
        for i in range(num_mask):
            mask = label[i]
            center_coord = self.find_mask_center_coord(mask)
            
            # auto line feed
            wrapped_caption = wrap(captions[i], max_line_length)
            caption_text = "\n".join(wrapped_caption)

            self.draw_text(image, caption_text, center_coord, color=(255, 255, 255), color_bg=None, text_scale=text_scale, text_thickness=text_thickness, text_padding=text_padding)
            
        return image
    
    def draw_points_on_image(self, image, points, colors=None,radius=5, thickness=-1):
        # image [h,w,3]
        # points: list[np.array], array shape (n, 2), format (x, y)
        
        if colors is None:
            colors = random.choices(self.bgr_colors, k=len(points))
            
        for i,p in enumerate(points):
            for x, y in p:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (int(x), int(y)), radius, colors[i], thickness)
        return image
    
def save_image_with_caption(image_array, caption, save_path, line_length=80):

    fig, ax = plt.subplots()
    ax.imshow(image_array)
    ax.axis('off')

    wrapped_caption = '\n'.join(wrap(caption, line_length))
    ax.text(0.5, -0.1, wrapped_caption, ha='center', va='top', fontsize=12, transform=ax.transAxes)

    fig.subplots_adjust(bottom=0.2)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)