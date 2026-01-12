
# python main.py \
#     --data_path '../../data/traffic/PEMS08/PEMS08.npz' \
#     --adj_filename ../../data/traffic/PEMS08/PEMS08.csv \
#     --dataset PEMS08FLOW \
#     --desc EMS08_pre \
#     --sample_len 12 \
#     --predict_len 12 \
#     --train_ratio 0.6 \
#     --val_ratio 0.2 \
#     --epoch 500 \
#     --val_epoch 1 \
#     --test_epoch 5 \
#     --batch_size 64 \
#     --lr 0.001 \
#     --causal 0 \
#     --model qwen3 \
#     --patience 50 \
#     --t_dim 64 \
#     --node_emb_dim 64 \
#     --node_embedding \
#     --llm_layers 3 \
#     --time_token \
#     --dropout 0.05 \
#     --trunc_k 64 \
#     --weight_decay 0 \
#     --task prediction \
#     --sandglassAttn 2 \
#     --sag_dim 128 \
#     --sag_tokens 46 \
#     --input_dim 1\ \
#     --output_dim 1 \
#     --use_anchor_diff_token 2 \
#     --use_diff 0 \
#     --user_instruction \
#     # --use_sep_token \
#     # --use_task_token \
#     # --use_context_token \
#     # --use_quality_token

python main.py \
    --data_path '../../data/traffic/PEMS08/PEMS08.npz' \
    --adj_filename ../../data/traffic/PEMS08/PEMS08.csv \
    --dataset PEMS08FLOW \
    --desc ftllm_EMS08_pre \
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
    --model qwen3 \
    --lora \
    --ln_grad \
    --patience 50 \
    --t_dim 64 \
    --node_emb_dim 64 \
    --node_embedding \
    --llm_layers 3 \
    --time_token \
    --dropout 0.05 \
    --trunc_k 64 \
    --weight_decay 0 \
    --task prediction \
    --sandglassAttn 2 \
    --sag_dim 128 \
    --sag_tokens 46 \
    --input_dim 1\ \
    --output_dim 1 \
    --use_anchor_diff_token 2 \
    --use_diff 0 \
    --user_instruction


# python main.py \
#     --data_path '../../data/traffic/PEMS08/PEMS08.npz' \
#     --adj_filename ../../data/traffic/PEMS08/PEMS08.csv \
#     --dataset PEMS08FLOW \
#     --desc use_anchor_diff_PEMS08_pre\ \
#     --sample_len 12 \
#     --predict_len 12 \
#     --train_ratio 0.6 \
#     --val_ratio 0.2 \
#     --epoch 500 \
#     --val_epoch 1 \
#     --test_epoch 5 \
#     --batch_size 64\ \
#     --lr 0.001 \
#     --causal 0 \
#     --model qwen3 \
#     --patience 50 \
#     --ln_grad \
#     --lora \
#     --t_dim 64 \
#     --node_emb_dim 64 \
#     --node_embedding \
#     --llm_layers 3 \
#     --time_token \
#     --dropout 0.05 \
#     --trunc_k 64 \
#     --weight_decay 0 \
#     --task prediction \
#     --sandglassAttn \
#     --sag_dim 128 \
#     --sag_tokens 128 \
#     --input_dim 1\ \
#     --output_dim 1 \
#     --use_anchor_diff_token 1

# python main.py \
#     --data_path '../../data/traffic/PEMS08/PEMS08.npz' \
#     --adj_filename ../../data/traffic/PEMS08/PEMS08.csv \
#     --dataset PEMS08FLOW \
#     --desc PEMS08_pre\ \
#     --sample_len 12 \
#     --predict_len 12 \
#     --train_ratio 0.6 \
#     --val_ratio 0.2 \
#     --epoch 500 \
#     --val_epoch 1 \
#     --test_epoch 5 \
#     --batch_size 64\ \
#     --lr 0.001 \
#     --causal 0 \
#     --model qwen3 \
#     --patience 50 \
#     --ln_grad \
#     --lora \
#     --t_dim 64 \
#     --node_emb_dim 64 \
#     --node_embedding \
#     --llm_layers 3 \
#     --time_token \
#     --dropout 0.05 \
#     --trunc_k 64 \
#     --weight_decay 0 \
#     --task prediction \
#     --sandglassAttn \
#     --sag_dim 128 \
#     --sag_tokens 128 \
#     --input_dim 1\ \
#     --output_dim 1 \
#     --use_anchor_diff_token 0
