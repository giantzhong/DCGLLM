#!/bin/bash

# 获取 GPU 卡号参数，默认为 4
GPU_ID=${1:-4}

echo "Using GPU: $GPU_ID"
########################################
# 1. Transformer 任务1(MIMIC3) - Train: 80%
########################################
nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
python main_dcgllm.py \
  --dataset MIMIC3 \
  --modeltype Transformer \
  --cuda_choice cuda:0 \
  --train_lr 0.0015 \
  --weight_decay 0.001 \
  --batch_size 128 \
  --epochs_train 100 \
  --target CKD \
  --train_dropout_rate 0.1 \
  --use_dynamic_graph \
  --use_cross_attention \
  --edge_weight_mode norm \
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.8,0.1,0.1]" \
  > run_Transformer_ab_e-.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
python main_dcgllm.py \
  --dataset MIMIC3 \
  --modeltype Transformer \
  --cuda_choice cuda:0 \
  --train_lr 0.0015 \
  --weight_decay 0.001 \
  --batch_size 128 \
  --epochs_train 100 \
  --target CVD \
  --train_dropout_rate 0.1 \
  --use_dynamic_graph \
  --use_cross_attention \
  --edge_weight_mode norm \
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.8,0.1,0.1]" \
  > run_Transformer_ab_e-.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
python main_dcgllm.py \
  --dataset MIMIC4 \
  --modeltype Transformer \
  --cuda_choice cuda:0 \
  --train_lr 0.0015 \
  --weight_decay 0.001 \
  --batch_size 128 \
  --epochs_train 100 \
  --target CKD \
  --train_dropout_rate 0.1 \
  --use_dynamic_graph \
  --use_cross_attention \
  --edge_weight_mode norm \
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.8,0.1,0.1]" \
  > run_Transformer_ab_e-.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
python main_dcgllm.py \
  --dataset MIMIC4 \
  --modeltype Transformer \
  --cuda_choice cuda:0 \
  --train_lr 0.0015 \
  --weight_decay 0.001 \
  --batch_size 128 \
  --epochs_train 100 \
  --target CVD \
  --train_dropout_rate 0.1 \
  --use_dynamic_graph \
  --use_cross_attention \
  --edge_weight_mode norm \
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.8,0.1,0.1]" \
  > run_Transformer_ab_e-.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
python main_dcgllm.py \
  --dataset HuaDong \
  --modeltype Transformer \
  --cuda_choice cuda:0 \
  --train_lr 0.0015 \
  --weight_decay 0.001 \
  --batch_size 128 \
  --epochs_train 100 \
  --target DM \
  --train_dropout_rate 0.1 \
  --use_dynamic_graph \
  --use_cross_attention \
  --edge_weight_mode norm \
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.8,0.1,0.1]" \
  > run_Transformer_ab_e-.log 2>&1 &