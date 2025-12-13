
python main.py \
    --data_path '../../data/traffic/PEMS03/PEMS03.npz' \
    --adj_filename ../../data/traffic/PEMS03/PEMS03.csv \
    --dataset PEMS03FLOW \
    --desc sep_PEMS03_pre\
    --sample_len 12 \
    --predict_len 12 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --epoch 500 \
    --val_epoch 1 \
    --test_epoch 5 \
    --batch_size 64\
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
    --sandglassAttn \
    --sag_dim 128 \
    --sag_tokens 128 \
    --input_dim 1\
    --output_dim 1 \
    --use_anchor_diff_token 2 \
    --use_diff 0 \
    --use_sep_token \
    --use_task_token \
    --use_context_token \
    --use_quality_token

# python main.py \
#     --data_path '../../data/traffic/PEMS03/PEMS03.npz' \
#     --adj_filename ../../data/traffic/PEMS03/PEMS03.csv \
#     --dataset PEMS03FLOW \
#     --desc sag_PEMS03_pre\
#     --sample_len 12 \
#     --predict_len 12 \
#     --train_ratio 0.6 \
#     --val_ratio 0.2 \
#     --epoch 500 \
#     --val_epoch 1 \
#     --test_epoch 5 \
#     --batch_size 64\
#     --lr 0.001 \
#     --causal 0 \
#     --model gpt2 \
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
#     --input_dim 1\
#     --output_dim 1 \
#     --use_anchor_diff_token 0

# python main.py \
#     --data_path '../../data/traffic/PEMS03/PEMS03.npz' \
#     --adj_filename ../../data/traffic/PEMS03/PEMS03.csv \
#     --dataset PEMS03FLOW \
#     --desc use_anchor_diff_token__sag_PEMS03_pre\
#     --sample_len 12 \
#     --predict_len 12 \
#     --train_ratio 0.6 \
#     --val_ratio 0.2 \
#     --epoch 500 \
#     --val_epoch 1 \
#     --test_epoch 5 \
#     --batch_size 64\
#     --lr 0.001 \
#     --causal 0 \
#     --model gpt2 \
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
#     --input_dim 1\
#     --output_dim 1 \
#     --use_anchor_diff_token 1
