#!/bin/bash
# Step 1: Generate training data for the Kangaroo adapter
# This collects hidden states from the full model on MMDuet2 multimodal data.

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_PATH=/data/wangzhichao/projects/SSD_full_history/data/annotations/adapter/all.jsonl
OUTPUT_DIR=/data/wangzhichao/projects/SSD_full_history/datasets/all_0.1_no_reply_egoexo4d
EXIT_LAYERS=2,4,6  # Comma-separated list of exit layers to save hidden states for


CUDA_VISIBLE_DEVICES=6 python generate_training_data.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --exit_layers $EXIT_LAYERS \
    --no_reply_keep_ratio 0.1 \
    --gpu cuda:0
