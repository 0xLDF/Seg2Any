#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

conda activate seg2any

ngpu=4

# Pre-compute the bucket map once and cache it as `?H_?W-group_bucket.parquet` (we bucket samples by condition-image and text token counts).
# only run once!
python prepare_dataset_bucket_map.py config/seg2any_ade20k.yaml --data.train.params.is_group_bucket=True --data.train.params.resolution=512 --data.train.params.cond_scale_factor=1 
python prepare_dataset_bucket_map.py config/seg2any_coco_stuff.yaml --data.train.params.is_group_bucket=True --data.train.params.resolution=512 --data.train.params.cond_scale_factor=1
python prepare_dataset_bucket_map.py config/seg2any_sacap_1m.yaml --data.train.params.is_group_bucket=True --data.train.params.resolution=1024 --data.train.params.cond_scale_factor=2

###### seg2any train on ADE20k ######
accelerate launch --config_file config/accelerate_default_config.yaml \
--num_processes=${ngpu} \
train.py config/seg2any_ade20k.yaml \
--model.attention_mask_method="base" \
--model.is_use_cond_token=True \
--model.is_filter_cond_token=True \
--trainer.train_batch_size=4 \
--trainer.learning_rate=1e-4 \
--trainer.max_train_steps=20000 \
--trainer.validation_steps=2000 \
--trainer.checkpointing_steps=2000 \
--trainer.checkpoints_total_limit=10 \
--project.output_dir="./ckpt/ade20k/seg2any"

###### seg2any train on COCO Stuff ######
accelerate launch --config_file config/accelerate_default_config.yaml \
--num_processes=${ngpu} \
train.py config/seg2any_coco_stuff.yaml \
--model.attention_mask_method="base" \
--model.is_use_cond_token=True \
--model.is_filter_cond_token=True \
--trainer.train_batch_size=4 \
--trainer.learning_rate=1e-4 \
--trainer.max_train_steps=20000 \
--trainer.validation_steps=2000 \
--trainer.checkpointing_steps=2000 \
--trainer.checkpoints_total_limit=10 \
--project.output_dir="./ckpt/coco_stuff/seg2any"

# ###### seg2any train on SACap-1M ######
accelerate launch --config_file config/accelerate_default_config.yaml \
--num_processes=${ngpu} \
train.py config/seg2any_sacap_1m.yaml \
--model.attention_mask_method="hard" \
--model.is_use_cond_token=True \
--model.is_filter_cond_token=True \
--trainer.train_batch_size=4 \
--trainer.learning_rate=1e-4 \
--trainer.max_train_steps=20000 \
--trainer.validation_steps=2000 \
--trainer.checkpointing_steps=2000 \
--trainer.checkpoints_total_limit=10 \
--project.output_dir="./ckpt/sacap_1m/seg2any"