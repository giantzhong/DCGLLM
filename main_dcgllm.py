import logging
import torch
import random
import numpy as np
import math

import json
import torch.optim as optim
import argparse
import datetime

from time import time
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from utility.parser_seed import parse_args
from model.DCGLLM import DCGLLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from loader import get_dataloader

from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn, multilabel_metrics_fn, process_seed
import os
import pickle
from torch.cuda.amp import autocast, GradScaler
from config import LLM_MODEL_NAME
from data_preprocess.data_config import DataConfig

# 原始版本
# def patient_train_val_test_split(
#         dataset,  # List of patient samples with graph data
#         ratios: Union[Tuple[float, float, float], List[float]],
#         seed: Optional[int] = None,
# ):
#     if seed is not None:
#         np.random.seed(seed)
#     assert sum(ratios) == 1.0, "ratios must sum to 1.0"
#     patient_indx = list(range(0, len(dataset), 1))
#     label_list = [sample["label"] for sample in dataset]
#     temp_index, test_index, y_temp, y_test = \
#         train_test_split(patient_indx, label_list, test_size=ratios[2], stratify=label_list, random_state=seed)
#     train_index, val_index, y_train, y_val = train_test_split(temp_index, y_temp,
#                                                               test_size=ratios[1] / (ratios[0] + ratios[1]), #
#                                                               stratify=y_temp,
#                                                               random_state=seed)

#     train_dataset = torch.utils.data.Subset(dataset, train_index)
#     val_dataset = torch.utils.data.Subset(dataset, val_index)
#     test_dataset = torch.utils.data.Subset(dataset, test_index)
#     return train_dataset, val_dataset, test_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from typing import Union, Tuple, List, Optional

def patient_train_val_test_split(
        dataset,  # List of patient samples with graph data
        ratios: Union[Tuple[float, float, float], List[float]],
        seed: Optional[int] = None,
):
    if seed is not None:
        np.random.seed(seed)
    
    # 1. 允许总和小于1 (用于 Data Scarcity 实验)，但不能大于1
    # 使用 1e-6 容差处理浮点数精度问题
    assert sum(ratios) <= 1.0 + 1e-6, "ratios sum must be <= 1.0"
    
    patient_indx = list(range(len(dataset)))
    label_list = [sample["label"] for sample in dataset]

    # =========================================================================
    # 第一步：切分测试集 (Test)
    # 保持 test_size = ratios[2] (相对于总数据)
    # =========================================================================
    temp_index, test_index, y_temp, y_test = train_test_split(
        patient_indx, label_list, 
        test_size=ratios[2], 
        stratify=label_list, 
        random_state=seed
    )

    # =========================================================================
    # 第二步：从剩余数据中切分 训练集(Train) 和 验证集(Val)
    # 此时 temp_index 的长度约为总数据的 (1 - ratios[2])
    # 我们需要计算 Train 和 Val 在 temp_index 中的【相对比例】
    # =========================================================================
    
    # 剩余数据的总比例 (相对于原始数据)
    remainder_ratio = 1.0 - ratios[2]
    
    # 计算相对比例
    # 例如：总train=0.5, 总rest=0.9 -> 相对train = 0.5/0.9
    relative_train_size = ratios[0] / remainder_ratio
    relative_val_size = ratios[1] / remainder_ratio
    
    # 使用 sklearn 的 train_test_split 同时切分
    # 如果 relative_train + relative_val < 1.0，剩下的数据会被自动丢弃
    train_index, val_index, y_train, y_val = train_test_split(
        temp_index, y_temp,
        train_size=relative_train_size,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=seed
    )

    # =========================================================================
    # 第三步：构建 Subset
    # =========================================================================
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    
    # 打印信息以确认数据量 (调试用)
    print(f"Split Result: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")

def inference(model, dataloader, additional_outputs=None) -> Dict[str, float]:
    loss_all = []
    y_true_all = []
    y_prob_all = []
    if additional_outputs is not None:
        additional_outputs = {k: [] for k in additional_outputs}
    for data, graph_batch, dynamic_graph_batch in dataloader:
        model.eval()
        with torch.no_grad():
            graph_batch = graph_batch.to(model.device, non_blocking=True)
            # 将动态图数据移到设备上
            if dynamic_graph_batch is not None:
                dynamic_graph_batch = [
                    [g.to(model.device, non_blocking=True) for g in patient_graphs] 
                    for patient_graphs in dynamic_graph_batch
                ]
            with autocast(enabled=torch.cuda.is_available()):
                loss_cls, cls_logits, patient_y_true, patient_y_prob, patient_embed = model(
                    graph_batch, dynamic_graph_batch, **data)
            y_true = patient_y_true.data.cpu().numpy()
            y_prob = patient_y_prob.data.cpu().numpy()
            loss_all.append(loss_cls.item())
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
    loss_mean = sum(loss_all) / len(loss_all)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    if additional_outputs is not None:
        additional_outputs = {key: np.concatenate(val)
                              for key, val in additional_outputs.items()}
        return y_true_all, y_prob_all, loss_mean, additional_outputs
    return y_true_all, y_prob_all, loss_mean

def evaluate(model, dataloader, metrics,threshold=0.5) -> Dict[str, float]:
    y_true_all, y_prob_all, loss_mean = inference(model, dataloader)
    mode = model.model.mode
    
    # 根据mode选择对应的metrics函数
    if mode == "binary":
        metrics_fn = binary_metrics_fn
    elif mode == "multiclass":
        metrics_fn = multiclass_metrics_fn
    elif mode == "multilabel":
        metrics_fn = multilabel_metrics_fn
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # pyhealth.metrics may not support custom metrics like MCC; compute extras here.
    metrics = list(metrics) if metrics is not None else []
    want_mcc = any(str(m).lower() == "mcc" for m in metrics)
    metrics_for_pyhealth = [m for m in metrics if str(m).lower() != "mcc"]

    scores = metrics_fn(y_true_all, y_prob_all, metrics=metrics_for_pyhealth, threshold=threshold)

    if want_mcc:
        if mode == "binary":
            y_true = np.asarray(y_true_all).reshape(-1).astype(int)
            y_prob = np.asarray(y_prob_all).reshape(-1)
            y_pred = (y_prob >= float(threshold)).astype(int)
            scores["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        elif mode == "multiclass":
            y_true = np.asarray(y_true_all).reshape(-1).astype(int)
            y_pred = np.argmax(np.asarray(y_prob_all), axis=1).astype(int)
            scores["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        elif mode == "multilabel":
            y_true = np.asarray(y_true_all).astype(int)
            y_prob = np.asarray(y_prob_all)
            y_pred = (y_prob >= float(threshold)).astype(int)
            mccs = []
            for j in range(y_true.shape[1]):
                mccs.append(float(matthews_corrcoef(y_true[:, j], y_pred[:, j])))
            scores["mcc"] = float(np.mean(mccs)) if len(mccs) else 0.0
    return scores

def main(args, singlerun_seed):
    random.seed(singlerun_seed)
    np.random.seed(singlerun_seed)
    torch.manual_seed(singlerun_seed)

    save_dir = args.save_dir + 'trlr{}_wdcay{}_model{}_hdim{}_edim{}_elayer{}_nodedim{}_gdim-{}_seed{}_{}/'.format(
        args.train_lr,
        args.weight_decay,
        args.modeltype,
        args.hidden_dim,
        args.embed_dim,
        args.encoder_layer,
        args.node_dim,
        args.gencoder_dim_list,
        singlerun_seed,
        args.cuda_choice)

    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个运行添加独立的文件日志handler
    file_handler = logging.FileHandler(os.path.join(save_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    
    logging.info(args)

    def load_pickle(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    # Task-aware data loading: read from data/<dataset>/HTN2<TARGET>/
    data_config = DataConfig(args.dataset, target=getattr(args, "target", "dm"))
    merged_path = os.path.join(data_config.output_path, f"samples-graph-merged_{LLM_MODEL_NAME}.pkl")
    datasample, datagraph_sample = load_pickle(merged_path)

    x_key = ["conditions"]
    with open(data_config.id_to_icd_map_path, "r") as f:
        id_to_icd_map = json.load(f)

    node_num = len(id_to_icd_map)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(singlerun_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    train_ds, val_ds, test_ds = patient_train_val_test_split(datagraph_sample, eval(args.train_val_test_split),
                                                             singlerun_seed)

    train_dataloader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 参数定义详情在此
    model_kwargs = {"dataset": datasample, "feature_keys": x_key, "label_key": "label", "mode": "binary", "node_num": node_num}
    
    model = DCGLLM(args, **model_kwargs)
    model.to(device)
    logging.info(model)
    logging.info(args.cuda_choice)
    with open(save_dir + "params.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    last_checkpoint_path_name = os.path.join(save_dir, "last.ckpt")
    best_checkpoint_path_name = os.path.join(save_dir, "best.ckpt")

    if args.use_last_checkpoint != -1:
        logging.info(f"Loading checkpoint from {last_checkpoint_path_name}")
        state_dict = torch.load(last_checkpoint_path_name, map_location=args.cuda_choice)
        model.load_state_dict(state_dict)
    logging.info("")

    logging.info("Training:")
    param = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param if (not any(nd in n for nd in no_decay)) and (not "gencoders" in n)],
            "lr": args.train_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param if (not any(nd in n for nd in no_decay)) and ("gencoders" in n)],
            "lr": args.gencoder_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param if (any(nd in n for nd in no_decay)) and (not "gencoders" in n)],
            "lr": args.train_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in param if (any(nd in n for nd in no_decay)) and ("gencoders" in n)],
            "lr": args.gencoder_lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer_train = optim.Adam(optimizer_grouped_parameters)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # 新增：学习率学习器    
    scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_train, 
                T_max=args.epochs_train, 
                eta_min=1e-6
            )
        elif args.lr_scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer_train,
                step_size=args.lr_decay_epochs,
                gamma=args.lr_decay_rate
            )
        elif args.lr_scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_train,
                mode='max' if args.monitor_criterion == 'max' else 'min',
                factor=args.lr_decay_rate,
                patience=10,
                verbose=True
            )
        elif args.lr_scheduler_type == 'warmup_cosine':
            # Custom warmup + cosine annealing
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    return (epoch + 1) / args.warmup_epochs
                else:
                    progress = (epoch - args.warmup_epochs) / (args.epochs_train - args.warmup_epochs)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer_train, lr_lambda)
        
        logging.info(f"Using learning rate scheduler: {args.lr_scheduler_type}")
    
    data_iterator = iter(train_dataloader)
    best_score = -1 * float("inf") if args.monitor_criterion == "max" else float("inf")
    steps_per_epoch = len(train_dataloader)
    global_step = 0
    best_dev_epoch = 0

    for epoch in range(args.epochs_train):
        time0 = time()
        training_loss_all = []
        model.train()

        for _ in range(steps_per_epoch):
            optimizer_train.zero_grad()
            try:
                data, graph_batch, dynamic_graph_batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_dataloader)
                data, graph_batch, dynamic_graph_batch = next(data_iterator)
            graph_batch = graph_batch.to(device, non_blocking=True)
            # 将动态图数据移到设备上
            if dynamic_graph_batch is not None:
                dynamic_graph_batch = [
                    [g.to(device, non_blocking=True) for g in patient_graphs] 
                    for patient_graphs in dynamic_graph_batch
                ]
            with autocast(enabled=torch.cuda.is_available()):
                loss_cls, cls_logits, patient_y_true, patient_y_prob, patient_embed = model(
                    graph_batch, dynamic_graph_batch, **data)
            scaler.scale(loss_cls).backward()

            if args.clip != -1:
                scaler.unscale_(optimizer_train)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer_train)
            scaler.update()

            training_loss_all.append(loss_cls.item())
            global_step += 1

        logging.info("--- Train epoch-{}, step-{}, Total Time {:.1f}s---".format(epoch, global_step, time() - time0))
        logging.info(f"loss: {sum(training_loss_all) / len(training_loss_all):.4f}")

        if last_checkpoint_path_name is not None:
            state_dict = model.state_dict()
            torch.save(state_dict, last_checkpoint_path_name)

        if val_dataloader is not None:
            scores = evaluate(model, val_dataloader, args.metrics)
            logging.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
            logging.info(f"--- Val Metrics ---")
            for key in scores.keys():
                logging.info("{}: {:.4f}".format(key, scores[key]))
            if args.monitor is not None:
                assert args.monitor in args.metrics, "monitor not in metrics!"
                score = scores[args.monitor]
                if is_best(best_score, score, args.monitor_criterion):
                    logging.info(
                        f"New best {args.monitor} score ({score:.4f}) "
                        f"at epoch-{epoch}, step-{global_step}"
                    )
                    best_dev_epoch = epoch
                    best_score = score
                    if best_checkpoint_path_name is not None:
                        state_dict = model.state_dict()
                        torch.save(state_dict, best_checkpoint_path_name)
            
            # Update learning rate scheduler
            if scheduler is not None:
                if args.lr_scheduler_type == 'plateau':
                    # ReduceLROnPlateau needs validation metric
                    scheduler.step(score)
                else:
                    scheduler.step()
                
                # Log current learning rate
                current_lr = optimizer_train.param_groups[0]['lr']
                logging.info(f"Current learning rate: {current_lr:.6f}")

        if epoch > args.unfreeze_epoch and epoch - best_dev_epoch >= args.max_epochs_before_stop:
            break

    logging.info('Best eval score: {:.4f} (at epoch {})'.format(best_score, best_dev_epoch))

    if os.path.isfile(best_checkpoint_path_name):
        logging.info("Loaded best model")
        state_dict = torch.load(best_checkpoint_path_name, map_location=args.cuda_choice)
        model.load_state_dict(state_dict)

    # ---- Recompute or set threshold for evaluation ----
    # Default to user fixed threshold
    best_threshold = getattr(args, "fixed_threshold", 0.5)
    if val_dataloader is not None and getattr(args, "eval_threshold_search", False):
        thr_min = float(getattr(args, "eval_threshold_min", 0.01))
        thr_max = float(getattr(args, "eval_threshold_max", 0.99))
        thr_step = float(getattr(args, "eval_threshold_step", 0.01))
        # Construct grid including bounds
        num_steps = int(np.floor((thr_max - thr_min) / thr_step)) + 1
        threshold_grid = thr_min + np.arange(num_steps, dtype=float) * thr_step
        threshold_grid = threshold_grid[(threshold_grid >= 0.0) & (threshold_grid <= 1.0)]
        best_metric_value = -1.0
        for thr in threshold_grid:
            val_scores_thr = evaluate(model, val_dataloader, args.metrics, threshold=float(thr))
            metric_value = val_scores_thr.get(args.eval_threshold_metric, None)
            if metric_value is not None and metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = float(thr)
        logging.info(
            f"Selected threshold on Val by optimizing {args.eval_threshold_metric}: "
            f"{best_threshold:.2f} (val {args.eval_threshold_metric}={best_metric_value:.4f})"
        )

    if test_dataloader is not None:
        # Evaluate on test set with selected/fixed threshold for thresholded metrics
        scores = evaluate(model, test_dataloader, args.metrics, threshold=best_threshold)
        logging.info(f"--- Test ---")
        logging.info(f"Applied threshold: {best_threshold:.2f} for thresholded metrics")
        for key in scores.keys():
            logging.info("{}: {:.4f}".format(key, scores[key]))

    # 移除文件handler避免内存泄漏
    logger.removeHandler(file_handler)
    file_handler.close()
    
    return scores


if __name__ == "__main__":
    # 配置基本日志（只配置一次）
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from utility.parser_seed import parse_args
    args = parse_args()
    all_scores = []
    for seed in args.seed:
        scores = main(args,seed)
        all_scores.append(scores)
    print(all_scores)
    for metric in args.metrics:
        print("{}:{:.4f}({:.4f})".format(metric,np.mean([score[metric] for score in all_scores]),
                                         np.std([score[metric] for score in all_scores])))
    
    # 保存结果到txt文件
    model_name = args.modeltype
    # result_file = f"./trained_model/{args.dataset}/{model_name}_trainsize.txt"
    t = (getattr(args, "target", "dm") or "").strip().upper()
    target_up = t if t in ("DM", "CVD", "CKD") else t.upper()
    result_file = f"./trained_model/{args.dataset}/HTN2{target_up}/{model_name}.txt"
    # result_file = f"./trained_model/{args.dataset}/HTN2{target_up}/{model_name}_trainsize.txt"

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    # 确定损失类型
    if args.use_focal_loss:
        loss_type = "use_focal_loss"
    elif args.use_weighted_loss:
        loss_type = "use_weighted_loss"
    else:
        loss_type = "BCE"
    
    # 使用交叉注意力机制
    use_cross_attn = "True" if args.use_cross_attention else "False"
    use_dynamic_graph = "True" if args.use_dynamic_graph else "False"
    
    # 追加模式写入文件
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Learning Rate: {args.train_lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Loss Type: {loss_type}\n")
        f.write(f"Use Cross Attention: {use_cross_attn}\n")
        f.write(f"Use Dynamic Graph: {use_dynamic_graph}\n")
        f.write(f"Edge Weight Mode: {args.edge_weight_mode}\n")
        f.write(f"LLM Model: {LLM_MODEL_NAME}\n")
        f.write(f"Train Size: {args.train_val_test_split}\n")
        f.write("-" * 80 + "\n")
        
        # 保存每组（每个seed）的结果
        f.write("Individual Results:\n")
        for seed, score in zip(args.seed, all_scores):
            f.write(f"  Seed {seed}:\n")
            for metric in args.metrics:
                f.write(f"    {metric}: {score[metric]:.4f}\n")
        
        f.write("-" * 80 + "\n")
        f.write("Summary (Mean ± Std):\n")
        for metric in args.metrics:
            mean_val = np.mean([score[metric] for score in all_scores])
            std_val = np.std([score[metric] for score in all_scores])
            f.write(f"  {metric}: {mean_val:.4f} ({std_val:.4f})\n")
        f.write("=" * 80 + "\n\n")