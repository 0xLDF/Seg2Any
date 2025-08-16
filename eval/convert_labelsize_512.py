import json
import os
from PIL import Image
import argparse
from tqdm import tqdm
import multiprocessing

def process_file(args):
    filename, input_folder, output_folder = args
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    if os.path.exists(output_path):
        return
    
    label = Image.open(input_path)
    label_resized = label.resize((512, 512), resample=Image.NEAREST)
    label_resized.save(output_path)

def resize_panoptic_labels(input_folder, output_folder, num_processes):
    file_args = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg')):
            file_args.append((filename, input_folder, output_folder))
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_file, file_args), 
                total=len(file_args)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--num_processes', type=int, default=16)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)

    num_processes = min(max(1, args.num_processes), multiprocessing.cpu_count() * 2)
    
    resize_panoptic_labels(
        args.input_folder, 
        args.output_folder,
        num_processes
    )
    print("Done!")