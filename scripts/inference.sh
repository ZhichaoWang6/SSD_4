#!/bin/bash
# Standard inference (baseline, no speculative decoding)

python -u inference.py \
        --use_speculative_decoding \
        --compare_AR_SSD \
        --output_fname ./outputs/2fps/preds_two_10_no_reply_ML_egoexo4d_14.jsonl \
        --device cuda:4 \
        --exit_layer 2 \
        --adapter_path /data/wangzhichao/projects/SSD_full_history/adapter_checkpoints/TWO_online_layer2_ratio01_steps8/epoch_014 \
        --speculative_threshold 0.6 \
    > ./logs/two_10_no_reply_ML_egoexo4d_14.log 2>&1
