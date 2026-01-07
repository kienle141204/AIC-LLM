#!/bin/bash

# PEMS08 with Adaptive RL Token Selection
# This script enables RL-based adaptive token selection for improved efficiency

python main.py \
    --data_path '../../data/traffic/PEMS08/PEMS08.npz' \
    --adj_filename ../../data/traffic/PEMS08/PEMS08.csv \
    --dataset PEMS08FLOW \
    --desc PEMS08_adaptive_rl \
    --sample_len 12 \
    --predict_len 12 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --epoch 500 \
    --val_epoch 1 \
    --test_epoch 5 \
    --batch_size 64 \
    --lr 0.001 \
    --causal 0 \
    --model gpt2 \
    --patience 50 \
    --ln_grad \
    --lora \
    --t_dim 64 \
    --node_emb_dim 64 \
    --node_embedding \
    --llm_layers 3 \
    --time_token \
    --dropout 0.05 \
    --trunc_k 64 \
    --weight_decay 0 \
    --task prediction \
    --sandglassAttn 1 \
    --sag_dim 128 \
    --sag_tokens 64 \
    --input_dim 1 \
    --output_dim 1 \
    --use_anchor_diff_token 2 \
    --use_diff 0 \
    --use_sep_token \
    --use_task_token \
    --use_context_token \
    --use_quality_token \
    --use_adaptive_rl \
    --min_sag_tokens 8 \
    --rl_lr 0.0001 \
    --rl_efficiency_coef 0.1 \
    --rl_update_freq 10
