from qwen_vl_utils import process_vision_info
import cv2
from PIL import Image

class QwenProcessor():
    def __init__(self,processor,min_pixels=224*224,max_pixels=1280 * 28 * 28):
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels =max_pixels
        
    def process(self,image_list,prompt):
        
        image_content = []
        for image in image_list:
            # convert BRG to RGB
            image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_content.append({
                "type": "image",
                "image": image,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels
            })
            
        messages = [
            {
                "role": "user",
                "content": [*image_content,{"type": "text", "text": prompt}],
            }
        ]
        
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(
            messages
        )
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs # list[PIL.Image]
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data
        }