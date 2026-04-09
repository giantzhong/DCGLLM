#!/bin/bash

# 获取 GPU 卡号参数，默认为 4
GPU_ID=${1:-4}

echo "Using GPU: $GPU_ID"


########################################
# 1. GRU  (HuaDong)
########################################
nohup env CUDA_VISIBLE_DEVICES=1 \
python main_dcgllm.py \
  --dataset HuaDong \
  --modeltype GRU \
  --train_lr 0.001 \
  --weight_decay 0.001 \
  --batch_size 256 \
  --epochs_train 100 \
  --target DM \
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
  > run_GRU_HuaDong_1.log 2>&1 &


########################################
# 2. HiTANet 任务 1 (HuaDong)
########################################
nohup env CUDA_VISIBLE_DEVICES=1 \
python main_dcgllm.py \
  --dataset HuaDong \
  --modeltype HiTANet \
  --train_lr 0.001 \
  --weight_decay 0.001 \
  --batch_size 256 \
  --epochs_train 100 \
  --target DM \
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
  > run_HiTANet_HuaDong_1.log 2>&1 &


########################################
# 3. Transformer 任务1(HuaDong)
########################################
nohup env CUDA_VISIBLE_DEVICES=2 \
python main_dcgllm.py \
  --dataset HuaDong \
  --modeltype Transformer \
  --train_lr 0.0015 \
  --weight_decay 0.001 \
  --batch_size 256 \
  --epochs_train 100 \
  --target DM \
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
  > run_Transformer_HuaDong_1.log 2>&1 &

########################################
# 4. StageNet 任务1(HuaDong)
########################################
nohup env CUDA_VISIBLE_DEVICES=2 \
python main_dcgllm.py \
  --dataset HuaDong \
  --modeltype StageNet \
  --train_lr 0.001 \
  --weight_decay 0.001 \
  --batch_size 256 \
  --epochs_train 100 \
  --target DM \
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
  > run_StageNet_HuaDong_1.log 2>&1 &