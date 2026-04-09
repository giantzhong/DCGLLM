from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

def collate_fn_dict(batch):
    # 排除 "graph" 和 "dynamic_graph_list" 的其他字段
    data = {key: [d[key] for d in batch] for key in batch[0] if key not in ["graph", "dynamic_graph_list"]}
    
    # 处理静态图
    graph = [d["graph"] for d in batch]
    graph_batch = Batch.from_data_list(graph)
    
    # 处理动态图列表（如果存在）
    dynamic_graph_batch = None
    if "dynamic_graph_list" in batch[0]:
        # dynamic_graph_batch 是一个列表，每个元素是一个患者的动态图序列
        dynamic_graph_batch = [d.get("dynamic_graph_list", []) for d in batch]
    
    return data, graph_batch, dynamic_graph_batch


def get_dataloader(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict,
        num_workers=8,  # 增加数据加载并行度
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # 预取更多batch
    )
    return dataloader