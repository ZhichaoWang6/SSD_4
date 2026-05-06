CUDA_VISIBLE_DEVICES=5 accelerate launch \
    --num_processes 1 --num_machines 1 --mixed_precision bf16 \
    train_adapter_rollout.py \
    --basepath /data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt \
    --datadir /data/wangzhichao/projects/SSD_full_history/datasets/ar \
    --outdir /data/wangzhichao/projects/SSD_full_history/adapter_checkpoints/MLP/rollout_layer2 \
    --exit_layer 2 \
    --lr 3e-5 \
    --bs 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 30 \
    --rollout_warmup_epochs 10 \
    --rollout_steps 4 \
    --scheduled_sampling_p_start 0.0 \
    --scheduled_sampling_p_end 1.0 \
    --rollout_min_seq_len 8 \
    --num_warmup_steps 200 \
    --grad_clip 0.5 \
    --save_freq 1