import copy
import torch
import pickle
import random
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from pyhealth.datasets.sample_dataset import SampleEHRDataset
import json

def dataset_merge(graph_dataset_path, patient_dataset_path, cfipf_dict_dataset_path, patient_dataset_path_merged,code_to_id_map_path, enable_undersampling=False, target_total=10000, random_seed=42):
    """
    合并图数据集和患者数据集，并可选地进行欠采样。
    
    Args:
        graph_dataset_path (str): 图数据文件路径
        patient_dataset_path (str): 患者数据文件路径
        cfipf_dict_dataset_path (str): CFIPF字典数据文件路径
        patient_dataset_path_merged (str): 合并后的数据保存路径
        enable_undersampling (bool, optional): 是否启用欠采样。默认为 False。
            - True: 启用欠采样，保留所有 label=1 的样本，从 label=0 中采样使总数达到 target_total
            - False: 禁用欠采样，使用所有可用样本
        target_total (int, optional): 欠采样目标总数。默认为 10000。
            仅在 enable_undersampling=True 时生效。
        random_seed (int, optional): 随机种子，用于保证可复现性。默认为 42。
    
    Returns:
        list: 处理后的数据样本列表（包含图数据）
    """
    with open(graph_dataset_path, "rb") as f:
        graph_dict = pickle.load(f)
    with open(patient_dataset_path, "rb") as f:
        datasample = pickle.load(f)
    with open(cfipf_dict_dataset_path, "rb") as f:
        cfipf_dict = pickle.load(f)
    with open(code_to_id_map_path, 'r') as f:
        code_to_id = json.load(f)

    datasample_ori = copy.deepcopy(datasample)
    
    datagraph_sample = []
    filtered_ids = []
    for sample in tqdm(datasample):
        patient_id: str = sample['patient_id']
        patient_id = patient_id.replace("+", "")
        patient_id = patient_id.replace("-", "")
        
        try:
            patient_graph:list = graph_dict[int(patient_id)] # 获取患者的疾病网络
            patient_node_mask_cfipf:dict = cfipf_dict[patient_id]
        except KeyError:
            filtered_ids.append(int(patient_id))
            continue
        
        if len(patient_node_mask_cfipf) == 0 :
            filtered_ids.append(int(patient_id))
            continue

        node_mask = list(patient_node_mask_cfipf.keys())
        # 建立节点id到图索引的映射
        node_mask_dict = {node_id: i for i, node_id in enumerate(node_mask)}
        # 第一列：节点的原始ID（假设为数值型，以便后续转换为浮点张量）
        # 第二列：该节点在CFIPF中的评分/权重。
        node_feature = [[node_id, patient_node_mask_cfipf[node_id]] for _, node_id in enumerate(node_mask)]

        edge_index = []
        edge_weight = []
        for edge in patient_graph:
            (start, end, weight) = edge
            
            start = node_mask_dict[start]
            end = node_mask_dict[end]
            edge_index.append([start, end])
            edge_weight.append(weight)

        edge_index = torch.tensor(edge_index, dtype=torch.int).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        x = torch.tensor(node_feature, dtype=torch.float)
        # 为每个患者包装它的图数据
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        sample["graph"] = data
        
        # 构造动态图列表
        node_id = x[:, 0]
        cfipf_weight = x[:, 1]  # shape [N]
        num_nodes = cfipf_weight.shape[0]
        dynamic_graph_list = []
        
        # 遍历患者的每次就诊记录 (conditions是二维列表，每个子列表是一次就诊的疾病代码)
        conditions = sample.get('conditions', [])
        for cond_list in conditions:
            # 为当前就诊时刻创建参与度向量
            participation_t = torch.zeros(num_nodes)
            
            # 遍历当前就诊的所有疾病代码 (ICD编码)
            for code in cond_list:
                # 将ICD编码转换为ID
                if code in code_to_id:
                    disease_id = code_to_id[code]
                    # 通过ID查询图节点索引
                    if disease_id in node_mask_dict:
                        node_idx = node_mask_dict[disease_id]
                        participation_t[node_idx] = 1.0
            
            # 构造节点特征: [cfipf_weight, participation_t]
            X_t = torch.stack([node_id, cfipf_weight, participation_t], dim=1)
            
            # 构造时间步 t 的图
            G_t = Data(
                x=X_t,
                edge_index=edge_index,
                edge_attr=edge_weight
            )
            
            dynamic_graph_list.append(G_t)
        
        sample["dynamic_graph_list"] = dynamic_graph_list
        datagraph_sample.append(sample)

    print("All patient num: ", len(datasample_ori))
    print("Used patient num: ", len(datagraph_sample))
    print("Filtered patients: ", filtered_ids)
    
    # 创建规范化 patient_id 的函数
    def normalize_patient_id(patient_id):
        """规范化 patient_id，移除 '+' 和 '-'"""
        if isinstance(patient_id, str):
            return patient_id.replace("+", "").replace("-", "")
        return str(patient_id)
    
    # 根据参数决定是否进行欠采样
    if enable_undersampling:
        print(f"Undersampling enabled: target_total={target_total}")
        # 欠采样：保留所有 label=1 的样本，从 label=0 中采样使得总数达到 target_total
        label_1_samples = [sample for sample in datagraph_sample if sample.get('label') == 1]
        label_0_samples = [sample for sample in datagraph_sample if sample.get('label') == 0]
        
        num_label_1 = len(label_1_samples)
        num_label_0 = len(label_0_samples)
        
        print(f"Before undersampling - label=1: {num_label_1}, label=0: {num_label_0}, total: {len(datagraph_sample)}")
        
        # 计算需要采样的 label=0 样本数量
        num_label_0_needed = max(0, target_total - num_label_1)
        
        if num_label_0_needed < num_label_0:
            # 随机采样 label=0 的样本
            random.seed(random_seed)  # 设置随机种子以保证可复现性
            label_0_sampled = random.sample(label_0_samples, num_label_0_needed)
            datagraph_sample_final = label_1_samples + label_0_sampled
        else:
            # 如果 label=1 的数量已经 >= target_total，保留所有 label=1，不采样 label=0
            # 或者如果需要所有 label=0，则全部保留
            if num_label_1 >= target_total:
                print(f"Warning: label=1 samples ({num_label_1}) >= target total ({target_total}). Keeping all label=1 samples.")
                datagraph_sample_final = label_1_samples
            else:
                datagraph_sample_final = label_1_samples + label_0_samples
        
        # 创建采样后样本的规范化 patient_id 集合，用于更新 datasample_ori
        sampled_patient_ids_normalized = {
            normalize_patient_id(sample['patient_id']) 
            for sample in datagraph_sample_final
        }
        
        # 更新 datasample_ori，只保留采样后的样本
        # 注意：datasample_ori 中包含所有原始样本（包括被过滤掉的），我们需要匹配那些在采样后的样本
        datasample_ori_final = [
            sample for sample in datasample_ori 
            if normalize_patient_id(sample['patient_id']) in sampled_patient_ids_normalized
        ]
        
        print(f"After undersampling - label=1: {len(label_1_samples)}, label=0: {len([s for s in datagraph_sample_final if s.get('label') == 0])}, total: {len(datagraph_sample_final)}")
    else:
        print("Undersampling disabled: using all samples")
        # 不进行欠采样，使用所有样本
        datagraph_sample_final = datagraph_sample
        
        # 创建规范化 patient_id 集合，用于更新 datasample_ori
        sampled_patient_ids_normalized = {
            normalize_patient_id(sample['patient_id']) 
            for sample in datagraph_sample_final
        }
        
        # 更新 datasample_ori，只保留成功处理的样本（排除被过滤掉的）
        datasample_ori_final = [
            sample for sample in datasample_ori 
            if normalize_patient_id(sample['patient_id']) in sampled_patient_ids_normalized
        ]
        
        # 统计 label 分布
        label_1_count = len([s for s in datagraph_sample_final if s.get('label') == 1])
        label_0_count = len([s for s in datagraph_sample_final if s.get('label') == 0])
        print(f"Final dataset - label=1: {label_1_count}, label=0: {label_0_count}, total: {len(datagraph_sample_final)}")
    
    # 将 datasample_ori_final 转换为 SampleEHRDataset 对象（如果还不是的话）
    if not isinstance(datasample_ori_final, SampleEHRDataset):
        datasample_ori_final = SampleEHRDataset(datasample_ori_final)
        print("✓ Converted datasample_ori_final to SampleEHRDataset")
    
    # 验证一致性：确保 datasample_ori_final 和 datagraph_sample_final 中的 patient_id 一致
    datasample_ori_patient_ids = {
        normalize_patient_id(sample['patient_id']) 
        for sample in datasample_ori_final
    }
    datagraph_sample_patient_ids = {
        normalize_patient_id(sample['patient_id']) 
        for sample in datagraph_sample_final
    }
    
    if datasample_ori_patient_ids != datagraph_sample_patient_ids:
        print(f"Warning: Mismatch between datasample_ori and datagraph_sample patient IDs")
        print(f"datasample_ori unique patient_ids: {len(datasample_ori_patient_ids)}")
        print(f"datagraph_sample unique patient_ids: {len(datagraph_sample_patient_ids)}")
        missing_in_ori = datagraph_sample_patient_ids - datasample_ori_patient_ids
        missing_in_graph = datasample_ori_patient_ids - datagraph_sample_patient_ids
        if missing_in_ori:
            print(f"Patient IDs in datagraph_sample but not in datasample_ori: {list(missing_in_ori)[:10]}")
        if missing_in_graph:
            print(f"Patient IDs in datasample_ori but not in datagraph_sample: {list(missing_in_graph)[:10]}")
    else:
        print("✓ Verified: datasample_ori and datagraph_sample are consistent")
    
    print(f"datasample_ori size: {len(datasample_ori_final)}")
    print(f"datagraph_sample size: {len(datagraph_sample_final)}")

    with open(patient_dataset_path_merged, "wb") as f:
        pickle.dump((datasample_ori_final, datagraph_sample_final), f)
    
    return datagraph_sample_final

if __name__ == "__main__":
    import os
    import sys
    import argparse

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import LLM_MODEL_NAME
    from data_preprocess.data_config import DataConfig

    parser = argparse.ArgumentParser(description="Merge graph data with patient dataset (task-aware)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MIMIC3",
        choices=["MIMIC3", "MIMIC4", "HuaDong"],
        help="Dataset name (default: MIMIC3)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="dm",
        choices=["dm", "cvd", "ckd", "DM", "CVD", "CKD"],
        help="Comorbidity task target: dm/cvd/ckd (default: dm)",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Override input graph path (default: data/<dataset>/HTN2<TARGET>/graph_raw_<model>.pkl)",
    )
    parser.add_argument(
        "--patient-pickle-path",
        type=str,
        default=None,
        help="Override input patient pickle path (default: DataConfig(dataset,target).dataset_path)",
    )
    parser.add_argument(
        "--cfipf-dict-path",
        type=str,
        default=None,
        help="Override input cfipf_dict path (default: data/<dataset>/HTN2<TARGET>/cfipf_dict.pkl)",
    )
    parser.add_argument(
        "--code-to-id-map-path",
        type=str,
        default=None,
        help="Override code_to_id.json path (default: data/<dataset>/HTN2<TARGET>/code_to_id.json)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override merged output path (default: data/<dataset>/HTN2<TARGET>/samples-graph-merged_<model>.pkl)",
    )
    parser.add_argument(
        "--enable-undersampling",
        action="store_true",
        help="Enable undersampling (keep all label=1; sample label=0 to reach target_total)",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=10000,
        help="Undersampling target total size (default: 10000). Only effective when undersampling is enabled.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for undersampling (default: 42).",
    )
    args = parser.parse_args()

    config = DataConfig(args.dataset, target=args.target)

    graph_temp_dataset_path = (
        args.graph_path
        if args.graph_path
        else os.path.join(config.output_path, f"graph_norm_{LLM_MODEL_NAME}.pkl")
    )
    patient_temp_dataset_path = (
        args.patient_pickle_path if args.patient_pickle_path else config.dataset_path
    )
    cfipf_dict_temp_dataset_path = (
        args.cfipf_dict_path if args.cfipf_dict_path else config.cfipf_dict_path
    )
    patient_dataset_path_merged = (
        args.output_path
        if args.output_path
        else os.path.join(config.output_path, f"samples-graph-merged_{LLM_MODEL_NAME}.pkl")
    )
    code_to_id_map_path = (
        args.code_to_id_map_path if args.code_to_id_map_path else config.code_to_id_map_path
    )

    os.makedirs(os.path.dirname(patient_dataset_path_merged), exist_ok=True)

    if not os.path.exists(patient_dataset_path_merged):
        datagraph_sample = dataset_merge(
            graph_temp_dataset_path, 
            patient_temp_dataset_path, 
            cfipf_dict_temp_dataset_path, 
            patient_dataset_path_merged,
            code_to_id_map_path,
            enable_undersampling=args.enable_undersampling,
            target_total=args.target_total,
            random_seed=args.random_seed,
        )

    