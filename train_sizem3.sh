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
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.8,0.1,0.1]" \
  > run_Transformer_1_m3.log 2>&1 &

########################################
# 2. Transformer 任务2(MIMIC3) - Train: 60%
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
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.6,0.1,0.1]" \
  > run_Transformer_2_m3.log 2>&1 &

########################################
# 3. Transformer 任务3(MIMIC3) - Train: 40%
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
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.4,0.1,0.1]" \
  > run_Transformer_3_m3.log 2>&1 &

########################################
# 4. Transformer 任务4(MIMIC3) - Train: 20%
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
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.2,0.1,0.1]" \
  > run_Transformer_4_m3.log 2>&1 &

########################################
# 5. Transformer 任务5(MIMIC3) - Train: 10%
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
  --use_lr_scheduler \
  --lr_scheduler_type warmup_cosine \
  --warmup_epochs 10 \
  --eval_threshold_search \
  --eval_threshold_metric pr_auc \
  --eval_threshold_min 0.1 \
  --eval_threshold_max 0.5 \
  --eval_threshold_step 0.1 \
  --train_val_test_split "[0.1,0.1,0.1]" \
  > run_Transformer_5_m3.log 2>&1 &