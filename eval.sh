#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

conda activate seg2any

ngpu=4

###### Convert ground-truth images to the required format. ######
# only run once!
python eval/convert_coco_stuff164k.py --input_folder="./data/coco_stuff/stuffthingmaps_trainval2017/val2017" --output_folder="./data/coco_stuff/stuffthingmaps_trainval2017/val2017_temp"
python eval/convert_labelsize_512.py --input_folder="./data/coco_stuff/stuffthingmaps_trainval2017/val2017_temp" --output_folder="./data/coco_stuff/stuffthingmaps_trainval2017/val2017_size512"
python eval/convert_labelsize_512.py --input_folder="./data/ADEChallengeData2016/annotations/validation" --output_folder="./data/ADEChallengeData2016/annotations/validation_size512"

###### eval seg2any on ade20k ######
step=20000
ckpt_dir="ade20k/seg2any"
output_dir="./ckpt/${ckpt_dir}/${step}-result"
gen_image_dir="${output_dir}/gen_imgs"
resume_from_checkpoint="./ckpt/${ckpt_dir}/checkpoint-${step}"
seg_map_path="./data/ADEChallengeData2016/annotations/validation_size512"

## generate image
accelerate launch -m eval.generate_image \
--num_processes=${ngpu} \
--base config/seg2any_ade20k.yaml \
--model.attention_mask_method="base" \
--model.is_use_cond_token=True \
--model.is_filter_cond_token=True \
--eval.num_images_per_prompt=4 \
--project.gen_image_dir="${gen_image_dir}" \
--resume_from_checkpoint="${resume_from_checkpoint}"

## compute MIoU
mim test mmseg eval/config/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py \
    --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
    --gpus 1 \
    --launcher pytorch \
    --cfg-options test_dataloader.dataset.datasets.0.data_prefix.img_path="${gen_image_dir}/group_0" \
                    test_dataloader.dataset.datasets.0.data_prefix.seg_map_path=${seg_map_path} \
                    test_dataloader.dataset.datasets.1.data_prefix.img_path="${gen_image_dir}/group_1" \
                    test_dataloader.dataset.datasets.1.data_prefix.seg_map_path=${seg_map_path} \
                    test_dataloader.dataset.datasets.2.data_prefix.img_path="${gen_image_dir}/group_2" \
                    test_dataloader.dataset.datasets.2.data_prefix.seg_map_path=${seg_map_path} \
                    test_dataloader.dataset.datasets.3.data_prefix.img_path="${gen_image_dir}/group_3" \
                    test_dataloader.dataset.datasets.3.data_prefix.seg_map_path=${seg_map_path} \
                    work_dir=${output_dir}


###### eval seg2any on COCO Stuff ######
step=20000
ckpt_dir="coco_stuff/seg2any"
output_dir="./ckpt/${ckpt_dir}/${step}-result"
gen_image_dir="${output_dir}/gen_imgs"
resume_from_checkpoint="./ckpt/${ckpt_dir}/checkpoint-${step}"
seg_map_path="./data/coco_stuff/stuffthingmaps_trainval2017/val2017_size512"

## generate image
accelerate launch -m eval.generate_image \
--num_processes=${ngpu} \
--base config/seg2any_coco_stuff.yaml \
--model.attention_mask_method="base" \
--model.is_use_cond_token=True \
--model.is_filter_cond_token=True \
--eval.num_images_per_prompt=4 \
--project.gen_image_dir="${gen_image_dir}" \
--resume_from_checkpoint="${resume_from_checkpoint}"

## compute MIoU
mim test mmseg eval/config/deeplabv3_r101-d8_4xb4-320k_coco-stuff164k-512x512.py \
    --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth \
    --gpus 1 \
    --launcher pytorch \
    --cfg-options test_dataloader.dataset.datasets.0.data_prefix.img_path="${gen_image_dir}/group_0" \
                    test_dataloader.dataset.datasets.0.data_prefix.seg_map_path=${seg_map_path} \
                    test_dataloader.dataset.datasets.1.data_prefix.img_path="${gen_image_dir}/group_1" \
                    test_dataloader.dataset.datasets.1.data_prefix.seg_map_path=${seg_map_path} \
                    test_dataloader.dataset.datasets.2.data_prefix.img_path="${gen_image_dir}/group_2" \
                    test_dataloader.dataset.datasets.2.data_prefix.seg_map_path=${seg_map_path} \
                    test_dataloader.dataset.datasets.3.data_prefix.img_path="${gen_image_dir}/group_3" \
                    test_dataloader.dataset.datasets.3.data_prefix.seg_map_path=${seg_map_path} \
                    work_dir=${output_dir}


###### eval seg2any on SACap-1M ######
step=20000
ckpt_dir="sacap_1m/seg2any"
output_dir="./ckpt/${ckpt_dir}/${step}-result"
gen_image_dir="${output_dir}/gen_imgs"
resume_from_checkpoint="./ckpt/${ckpt_dir}/checkpoint-${step}"

real_image_root="./data/SACap-1M/test"
seg_caption_path="./data/SACap-1M/annotations/anno_test.parquet"
cache_skip_anno_ids="./data/SACap-1M/cache/eval"
resolution=1024
cond_scale_factor=2
sam2_checkpoint="./ckpt/sam2/sam2.1_hiera_large.pt"

## generate image
accelerate launch -m eval.generate_image \
--num_processes=${ngpu} \
--base config/seg2any_sacap_1m.yaml \
--model.attention_mask_method="hard" \
--model.is_use_cond_token=True \
--model.is_filter_cond_token=True \
--eval.num_images_per_prompt=1 \
--project.gen_image_dir="${gen_image_dir}" \
--resume_from_checkpoint="${resume_from_checkpoint}"

## compute class_agnostic_miou
python -m eval.metric_class_agnostic_miou \
--num_replicas=${ngpu} \
--image_root="${real_image_root}" \
--seg_caption_path="${seg_caption_path}" \
--sam2_checkpoint="${sam2_checkpoint}" \
--gen_img_dir="${gen_image_dir}/group_0" \
--output_dir="${output_dir}" \
--resolution=${resolution} \
--cond_scale_factor=${cond_scale_factor} \