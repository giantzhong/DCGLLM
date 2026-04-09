import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

# class DynamicGraphEncoder(nn.Module):
#     """
#     输入:
#         in_dim = 2（cfipf, participation）
#     输出:
#         graph_embed: [batch, out_dim]（默认 256）
#     """

#     def __init__(self, in_dim=2, out_dim=256):
#         super().__init__()
#         self.egcn = EvolveGCNO(in_dim)   # 输出永远是 in_dim
#         self.relu = nn.ReLU()

#         # 将2维投影到256维
#         self.proj = nn.Linear(in_dim, out_dim)

#         # 避免跨病人污染：EvolveGCN-O 的初始权重
#         self.register_buffer("initial_weight", None)

#     def forward(self, graph_seq):
#         # graph_seq = List[Data] （每个visit一个动态图）
#         if len(graph_seq) == 0:
#             device = next(self.parameters()).device
#             return torch.zeros(1, 256).to(device)

#         # 第一次 forward 时保存 initial_weight
#         if self.initial_weight is None:
#             self.initial_weight = self.egcn.initial_weight.clone()

#         # 重置追踪权重（避免跨病人污染）
#         self.egcn.weight = None
#         self.egcn.initial_weight.data[:] = self.initial_weight.data

#         step_embeds = []

#         for data in graph_seq:
#             H = self.egcn(data.x, data.edge_index, data.edge_attr)  # [N, 2]
#             H = self.relu(H)
#             step_embed = H.mean(dim=0)  # [2]
#             step_embeds.append(step_embed)

#         # EvolveGCN-O 最后一个时刻的信息最代表最新状态
#         last_step = step_embeds[-1]  # [2]

#         # 投影到 256 维
#         graph_embed = self.proj(last_step)  # [256]

#         return graph_embed

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, ChebConv

class DySATDynamicGraphEncoder(nn.Module):
    """
    DySAT 风格动态图编码器（高效并行版 + GRU 时间建模）

    输入：
        dynamic_graph_batch: List[List[Data]]
          - 外层 list：batch 中每个病人
          - 内层 list：该病人的动态图序列（每次 visit 一张 Data）
          - 每个 Data 至少包含：
              x: [num_nodes, in_dim]  （你这里是 [cfipf, participation] => in_dim=2）
              edge_index: [2, num_edges]
              edge_attr: [num_edges] 或 None

    输出：
        graph_embed: [batch_size, out_dim]
          - 每个病人一个动态图表示，用于和 patient_embed 融合
    
    优化要点：
        1. 用 GRU 替换 Transformer（O(T) vs O(T²)，速度提升 3-5x）
        2. 默认 max_timesteps=5，只用最近 5 次访问
        3. 减少不必要的计算开销
    """

    def __init__(
        self,
        in_dim: int = 2,           # 节点特征维度（cfipf + participation）
        gnn_dim: int = 256,        # 结构编码后的维度（GCN 输出）
        out_dim: int = 256,        # 最终输出维度（一般设成和静态图 encoder 输出一致）
        num_heads: int = 8,        # 保留参数兼容性，但不使用
        num_temporal_layers: int = 1,  # GRU 层数
        dropout: float = 0.1,
        max_timesteps: int = 5,    # 默认限制为 5，加速训练（原来是 None）
        edge_weight_mode: str = 'full',  # 消融实验参数：'full'|'none'|'random'
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.out_dim = out_dim
        self.max_timesteps = max_timesteps
        self.edge_weight_mode = edge_weight_mode

        # 结构层：GCNConv，对（所有患者 × 所有时间步）的节点一起做卷积
        # self.gnn = GCNConv(in_dim, gnn_dim)

        self.gnn = GATConv(
            in_channels=in_dim,
            out_channels=gnn_dim // num_heads,
            heads=num_heads,
            concat=True,    # 输出维度 = out_channels * heads = gnn_dim
            dropout=dropout,
            edge_dim=1      # 你的 edge_attr 是一个 scalar (PCS)
        )

        # self.gnn = ChebConv(
        #     in_channels=in_dim,
        #     out_channels=gnn_dim,
        #     K=3,                 # 推荐：3 或 5
        #     normalization='sym'  # 医疗图一般用对称规范化
        # )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 时间层：用 GRU 替换 Transformer（更快，O(T) 复杂度）
        self.temporal_gru = nn.GRU(
            input_size=gnn_dim,
            hidden_size=gnn_dim,
            num_layers=num_temporal_layers,
            batch_first=True,  # 输入形状 [B, T, D]
            dropout=dropout if num_temporal_layers > 1 else 0.0
        )

        # 映射到最终 out_dim（可以等于 gnn_dim，也可以再降/升维）
        self.out_proj = nn.Linear(gnn_dim, out_dim)

    def forward(self, dynamic_graph_batch):
        """
        dynamic_graph_batch: List[List[Data]]
        """
        device = self.out_proj.weight.device

        # ====== 1. 处理空的情况 ======
        if dynamic_graph_batch is None or len(dynamic_graph_batch) == 0:
            return torch.zeros(1, self.out_dim, device=device)

        batch_size = len(dynamic_graph_batch)

        # ====== 2. 展平：合并所有 (patient, t) 的图到一个 list ======
        flat_graphs = []
        seq_lens = []  # 记录每个病人的有效时间步数（可能被 max_timesteps 截断）

        for seq in dynamic_graph_batch:
            # seq: List[Data]，当前病人的时间序列
            if len(seq) == 0:
                # 极端情况：没有动态图，填一个占位长度 1（后面再特殊处理）
                seq_lens.append(0)
                continue

            # 如果限制了最大时间步数，则只保留最近 max_timesteps 个
            if (self.max_timesteps is not None) and (len(seq) > self.max_timesteps):
                seq = seq[-self.max_timesteps:]

            seq_lens.append(len(seq))
            flat_graphs.extend(seq)

        # 注意：可能有病人 seq_len=0，这种情况我们稍后用全0向量填充

        if len(flat_graphs) == 0:
            # 所有病人都没有动态图
            return torch.zeros(batch_size, self.out_dim, device=device)

        # ====== 3. 用 PyG 的 Batch 一次性打包所有时间步的图 ======
        big_batch = Batch.from_data_list(flat_graphs).to(device)

        x = big_batch.x                      # [sum_nodes, in_dim]
        edge_index = big_batch.edge_index    # [2, sum_edges]
        # 确保 edge_index 是 int64 类型（GATConv 要求）
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        edge_attr = getattr(big_batch, "edge_attr", None)

        # ====== 4. 处理消融实验的边权重模式 ======
        edge_attr_for_gnn = self._process_edge_attr(edge_attr)
        
        # ====== 5. 一次性做 GCNConv（结构编码） ======
        h = self.gnn(x, edge_index, edge_attr_for_gnn)   # [sum_nodes, gnn_dim]
        h = self.relu(h)
        h = self.dropout(h)

        # ====== 6. 图级池化：每个时间步一条图 embedding ======
        # big_batch.batch: 每个节点的 graph_id
        # graph_emb: [num_graphs(=总时间步数), gnn_dim]
        # [sum_nodes, gnn_dim] -> [num_graphs, gnn_dim]
        graph_emb = global_mean_pool(h, big_batch.batch)

        # ====== 7. 还原回 [batch_size, T_i, gnn_dim] 的时间序列形式 ======
        max_T = max(seq_lens) if len(seq_lens) > 0 else 1
        if max_T == 0:
            # 所有都没图，这种极端情况直接返回 0
            return torch.zeros(batch_size, self.out_dim, device=device)

        # padded_seq: [batch_size, max_T, gnn_dim]
        padded_seq = torch.zeros(batch_size, max_T, self.gnn_dim, device=device)

        cursor = 0
        for pid in range(batch_size):
            L = seq_lens[pid]
            if L == 0:
                # 没有任何动态图，就保持全0
                continue
            padded_seq[pid, :L, :] = graph_emb[cursor:cursor + L]
            cursor += L

        # ====== 8. 时间建模：使用 GRU（比 Transformer 快 3-5 倍） ======
        # GRU 输入：[B, T, D]，输出：[B, T, D] 和 hidden state
        temporal_out, _ = self.temporal_gru(padded_seq)  # [B, T, D]

        # ====== 9. 向量化取最后一个有效时间步（避免循环，更快） ======
        # 构建索引矩阵，直接提取每个样本的最后有效时间步
        seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.long)
        # 将 0 长度的序列设为 1（用于索引，反正是全 0）
        seq_lens_clamped = seq_lens_tensor.clamp(min=1) - 1  # [B]，转换为索引（0-based）
        
        # 使用 gather 提取最后时间步：temporal_out[i, seq_lens_clamped[i], :]
        # gather 需要索引维度匹配：[B, 1, D]
        indices = seq_lens_clamped.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.gnn_dim)
        dyn_graph_embed = torch.gather(temporal_out, dim=1, index=indices).squeeze(1)  # [B, D]
        
        # 对于长度为 0 的样本，手动置零
        zero_mask = (seq_lens_tensor == 0).unsqueeze(1)  # [B, 1]
        dyn_graph_embed = dyn_graph_embed.masked_fill(zero_mask, 0.0)

        # ====== 10. 投影到 out_dim（一般 out_dim 设成和静态图 encoder 输出一致，例如 256） ======
        dyn_graph_embed = self.out_proj(dyn_graph_embed)  # [B, out_dim]

        return dyn_graph_embed

    def _process_edge_attr(self, edge_attr):
        """
        处理消融实验中的边权重模式。
        
        Args:
            edge_attr: 原始边权重 [num_edges] 或 None
        
        Returns:
            处理后的边权重，根据 edge_weight_mode：
            - 'full': 返回原始 edge_attr
            - 'none': 返回 None（不使用边权重）
            - 'random': 返回打乱后的边权重
        """
        if self.edge_weight_mode == 'full':
            return edge_attr
        elif self.edge_weight_mode == 'none':
            return None
        elif self.edge_weight_mode == 'random':
            if edge_attr is None:
                return None
            # 克隆并打乱边权重
            edge_attr_random = edge_attr.clone()
            perm = torch.randperm(edge_attr_random.numel(), device=edge_attr_random.device)
            edge_attr_random = edge_attr_random[perm]
            return edge_attr_random
        else:
            raise ValueError(f"Unknown edge_weight_mode: {self.edge_weight_mode}")

class DySATDynamicGraphEncoder_V1(nn.Module):
    """
    DySAT 风格动态图编码器（适配 DCGLLM 节点加权逻辑）
    
    输入：
        dynamic_graph_batch: List[List[Data]]
        node_embedding: nn.Embedding 对象，外部传入，用于查找节点向量
    """

    def __init__(
        self,
        gnn_dim: int = 256,        # 输入 Embedding 维度 & GNN 隐层维度
        out_dim: int = 256,        # 最终输出维度
        num_heads: int = 8,        
        num_temporal_layers: int = 1,  
        dropout: float = 0.1,
        max_timesteps: int = 5,    
        edge_weight_mode: str = 'full', 
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.out_dim = out_dim
        self.max_timesteps = max_timesteps
        self.edge_weight_mode = edge_weight_mode

        # ====== 1. 结构层 (GNN) ======
        # 输入维度直接对齐 Embedding 维度 (gnn_dim)
        self.gnn = GATConv(
            in_channels=gnn_dim,   
            # out_channels=gnn_dim // num_heads,
            out_channels=gnn_dim,
            # heads=num_heads,
            heads=1,
            # concat=True,    # 输出维度 = (gnn_dim // heads) * heads = gnn_dim
            concat=False,    # 输出维度 = (gnn_dim // heads) * heads = gnn_dim
            dropout=dropout,
            edge_dim=1      
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ====== 2. 时间层 (GRU) ======
        self.temporal_gru = nn.GRU(
            input_size=gnn_dim,
            hidden_size=gnn_dim,
            num_layers=num_temporal_layers,
            batch_first=True,  
            dropout=dropout if num_temporal_layers > 1 else 0.0
        )

        # ====== 3. 输出映射 ======
        self.out_proj = nn.Linear(gnn_dim, out_dim)

    def forward(self, dynamic_graph_batch, node_embedding):
        """
        Args:
            dynamic_graph_batch: List[List[Data]]
            node_embedding: nn.Embedding or callable, shape [num_all_nodes, gnn_dim]
        """
        device = self.out_proj.weight.device

        # ====== 1. 预处理：处理空 Batch ======
        if dynamic_graph_batch is None or len(dynamic_graph_batch) == 0:
            return torch.zeros(1, self.out_dim, device=device)

        batch_size = len(dynamic_graph_batch)

        # ====== 2. 展平：List[List[Data]] -> List[Data] ======
        flat_graphs = []
        seq_lens = [] 
        for seq in dynamic_graph_batch:
            if len(seq) == 0:
                seq_lens.append(0)
                continue
            # 截断长序列
            if (self.max_timesteps is not None) and (len(seq) > self.max_timesteps):
                seq = seq[-self.max_timesteps:]
            seq_lens.append(len(seq))
            flat_graphs.extend(seq)

        if len(flat_graphs) == 0:
            return torch.zeros(batch_size, self.out_dim, device=device)

        # ====== 3. 构建大图 Batch ======
        big_batch = Batch.from_data_list(flat_graphs).to(device)
        
        # 解析特征 x: [sum_nodes, 3] -> [node_id, cfipf, participation]
        x_raw = big_batch.x
        x_index = x_raw[:, 0].long()     # 节点 ID
        cfipf_weight = x_raw[:, 1]       # TF-IDF 权重
        participation = x_raw[:, 2]      # 参与度 (0 or 1)

        edge_index = big_batch.edge_index.long()
        edge_attr = getattr(big_batch, "edge_attr", None)
        edge_attr_for_gnn = self._process_edge_attr(edge_attr)

        # ====== 4. Embedding 查找 ======
        # 使用外部传入的 node_embedding
        x = node_embedding(x_index) # [sum_nodes, gnn_dim]

        # ====== 5. GNN 结构编码 ======
        x = self.gnn(x, edge_index, edge_attr_for_gnn)
        x = self.relu(x)
        x = self.dropout(x)

        # ====== 6. DCGLLM 核心逻辑: Diagnostic Masked Pooling ======
        # 目标: softmax(cfipf) * x，但必须 mask 掉 participation=0 的节点
        
        # 步骤 A: 构造 Logits 
        # 将未参与的节点权重设为负无穷，Softmax 后变为 0
        mask_value = -1e9
        masked_logits = cfipf_weight.masked_fill(participation == 0, mask_value)
        
        # 步骤 B: 计算归一化权重 (在每个 graph 内部归一化)
        # alpha shape: [sum_nodes]
        alpha = softmax(masked_logits, big_batch.batch)
        
        # 步骤 C: 加权
        x_weighted = x * alpha.unsqueeze(-1)
        
        # 步骤 D: 聚合 (Add Pool)
        # 结果 shape: [total_timesteps, gnn_dim]
        graph_emb = global_add_pool(x_weighted, big_batch.batch) # [T * N ,D] -> [T, D]

        # ====== 7. 还原序列结构 (Pad & Pack) ======
        max_T = max(seq_lens) if len(seq_lens) > 0 else 1
        padded_seq = torch.zeros(batch_size, max_T, self.gnn_dim, device=device) # [B,T,D]
        
        cursor = 0
        for pid in range(batch_size):
            L = seq_lens[pid] # 每位患者的时间步骤长度 
            if L == 0: continue
            padded_seq[pid, :L, :] = graph_emb[cursor:cursor + L] # 来个游标
            cursor += L

        # ====== 8. 时间建模 (GRU) ======
        # temporal_out: [B, T, D]
        temporal_out, _ = self.temporal_gru(padded_seq)

        # ====== 9. 提取最后有效时间步 ======
        seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.long)
        # 避免索引为 -1 (针对长度为0的序列)
        seq_lens_clamped = seq_lens_tensor.clamp(min=1) - 1
        
        # gather 提取: [B, 1, D] -> [B, D]
        indices = seq_lens_clamped.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.gnn_dim)
        dyn_graph_embed = torch.gather(temporal_out, dim=1, index=indices).squeeze(1)
        
        # 将空序列产生的 embedding 置零
        zero_mask = (seq_lens_tensor == 0).unsqueeze(1)
        dyn_graph_embed = dyn_graph_embed.masked_fill(zero_mask, 0.0)

        # ====== 10. 最终投影 ======
        dyn_graph_embed = self.out_proj(dyn_graph_embed)

        return dyn_graph_embed

    def _process_edge_attr(self, edge_attr):
        if self.edge_weight_mode == 'full':
            return edge_attr
        elif self.edge_weight_mode == 'none':
            return None
        elif self.edge_weight_mode == 'random':
            if edge_attr is None: return None
            return edge_attr[torch.randperm(edge_attr.size(0), device=edge_attr.device)]
        elif self.edge_weight_mode == 'norm':
            # 将所有边的权重设置为相同的值（例如，1.0），然后归一化
            edge_attr_normalized = torch.ones_like(edge_attr)  # 赋值为1.0
            edge_attr_normalized = edge_attr_normalized / edge_attr_normalized.sum()  # 归一化
            return edge_attr_normalized
        return edge_attr

class DySATDynamicGraphEncoder_V2(nn.Module):
    """
    DySAT 风格动态图编码器（适配 DCGLLM 节点加权逻辑）
    
    输入：
        dynamic_graph_batch: List[List[Data]]
        node_embedding: nn.Embedding 对象，外部传入，用于查找节点向量
    """

    def __init__(
        self,
        gnn_dim: int = 256,        # 输入 Embedding 维度 & GNN 隐层维度
        out_dim: int = 256,        # 最终输出维度
        num_heads: int = 8,        
        num_temporal_layers: int = 1,  
        dropout: float = 0.1,
        max_timesteps: int = 5,    
        edge_weight_mode: str = 'full', 
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.out_dim = out_dim
        self.max_timesteps = max_timesteps
        self.edge_weight_mode = edge_weight_mode

        # ====== 1. 结构层 (GNN) ======
        # 输入维度直接对齐 Embedding 维度 (gnn_dim)
        self.gnn = GATConv(
            in_channels=gnn_dim,   
            # out_channels=gnn_dim // num_heads,
            out_channels=gnn_dim,
            # heads=num_heads,
            heads=1,
            # concat=True,    # 输出维度 = (gnn_dim // heads) * heads = gnn_dim
            concat=False,    # 输出维度 = (gnn_dim // heads) * heads = gnn_dim
            dropout=dropout,
            edge_dim=1      
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ====== 2. 时间层 (Transformer / DySAT-style temporal attention) ======
        self.temporal_mha = nn.MultiheadAttention(
            embed_dim=gnn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # 输入输出都是 [B, T, D]
        )

        # 可选：DySAT 里还有 position embedding，这里最小实现也加上（强烈建议）
        self.use_pos_emb = True
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(max_timesteps if max_timesteps is not None else 512, gnn_dim)

        # 可选：一个很轻的 FFN（DySAT 的 position_ffn 类似）
        self.temporal_ffn = nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_dim, gnn_dim),
        )

        self.temporal_norm1 = nn.LayerNorm(gnn_dim)
        self.temporal_norm2 = nn.LayerNorm(gnn_dim)

        # ====== 3. 输出映射 ======
        self.out_proj = nn.Linear(gnn_dim, out_dim)

    def forward(self, dynamic_graph_batch, node_embedding):
        """
        Args:
            dynamic_graph_batch: List[List[Data]]
            node_embedding: nn.Embedding or callable, shape [num_all_nodes, gnn_dim]
        """
        device = self.out_proj.weight.device

        # ====== 1. 预处理：处理空 Batch ======
        if dynamic_graph_batch is None or len(dynamic_graph_batch) == 0:
            return torch.zeros(1, self.out_dim, device=device)

        batch_size = len(dynamic_graph_batch)

        # ====== 2. 展平：List[List[Data]] -> List[Data] ======
        flat_graphs = []
        seq_lens = [] 
        for seq in dynamic_graph_batch:
            if len(seq) == 0:
                seq_lens.append(0)
                continue
            # 截断长序列
            if (self.max_timesteps is not None) and (len(seq) > self.max_timesteps):
                seq = seq[-self.max_timesteps:]
            seq_lens.append(len(seq))
            flat_graphs.extend(seq)

        if len(flat_graphs) == 0:
            return torch.zeros(batch_size, self.out_dim, device=device)

        # ====== 3. 构建大图 Batch ======
        big_batch = Batch.from_data_list(flat_graphs).to(device)
        
        # 解析特征 x: [sum_nodes, 3] -> [node_id, cfipf, participation]
        x_raw = big_batch.x
        x_index = x_raw[:, 0].long()     # 节点 ID
        cfipf_weight = x_raw[:, 1]       # TF-IDF 权重
        participation = x_raw[:, 2]      # 参与度 (0 or 1)

        edge_index = big_batch.edge_index.long()
        edge_attr = getattr(big_batch, "edge_attr", None)
        edge_attr_for_gnn = self._process_edge_attr(edge_attr)

        # ====== 4. Embedding 查找 ======
        # 使用外部传入的 node_embedding
        x = node_embedding(x_index) # [sum_nodes, gnn_dim]

        # ====== 5. GNN 结构编码 ======
        x = self.gnn(x, edge_index, edge_attr_for_gnn)
        x = self.relu(x)
        x = self.dropout(x)

        # ====== 6. DCGLLM 核心逻辑: Diagnostic Masked Pooling ======
        # 目标: softmax(cfipf) * x，但必须 mask 掉 participation=0 的节点
        
        # 步骤 A: 构造 Logits 
        # 将未参与的节点权重设为负无穷，Softmax 后变为 0
        mask_value = -1e9
        masked_logits = cfipf_weight.masked_fill(participation == 0, mask_value)
        
        # 步骤 B: 计算归一化权重 (在每个 graph 内部归一化)
        # alpha shape: [sum_nodes]
        alpha = softmax(masked_logits, big_batch.batch)
        
        # 步骤 C: 加权
        x_weighted = x * alpha.unsqueeze(-1)
        
        # 步骤 D: 聚合 (Add Pool)
        # 结果 shape: [total_timesteps, gnn_dim]
        graph_emb = global_add_pool(x_weighted, big_batch.batch) # [T * N ,D] -> [T, D]

        # ====== 7. 还原序列结构 (Pad & Pack) ======
        max_T = max(seq_lens) if len(seq_lens) > 0 else 1
        padded_seq = torch.zeros(batch_size, max_T, self.gnn_dim, device=device) # [B,T,D]
        
        cursor = 0
        for pid in range(batch_size):
            L = seq_lens[pid] # 每位患者的时间步骤长度 
            if L == 0: continue
            padded_seq[pid, :L, :] = graph_emb[cursor:cursor + L] # 来个游标
            cursor += L

        # ====== 8. 时间建模 (DySAT-style temporal self-attention) ======
        B, T, D = padded_seq.shape

        # (a) position embedding（最小实现）
        temporal_in = padded_seq
        if getattr(self, "use_pos_emb", False):
            # 位置索引 [0..T-1]
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            temporal_in = temporal_in + self.pos_emb(pos_ids)

        # (b) key_padding_mask: True 表示要 mask（padding）
        # lengths: [B], mask shape [B, T]
        seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.long)
        time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        key_padding_mask = time_ids >= seq_lens_tensor.unsqueeze(1)  # padding位置为 True

        # (c) causal mask: 上三角为 True（不允许看未来）
        # 注意：PyTorch 的 attn_mask 支持 bool mask：True=禁止attend
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        # (d) MHA + 残差 + LN（结构和 Transformer encoder 很像，改动很小）
        attn_out, _ = self.temporal_mha(
            temporal_in, temporal_in, temporal_in,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.temporal_norm1(temporal_in + attn_out)

        # (e) 轻量FFN（可选但推荐）
        ffn_out = self.temporal_ffn(x)
        temporal_out = self.temporal_norm2(x + ffn_out)  # [B, T, D]
        
        # ====== 9. 提取最后有效时间步 ======
        seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.long)
        # 避免索引为 -1 (针对长度为0的序列)
        seq_lens_clamped = seq_lens_tensor.clamp(min=1) - 1
        
        # gather 提取: [B, 1, D] -> [B, D]
        indices = seq_lens_clamped.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.gnn_dim)
        dyn_graph_embed = torch.gather(temporal_out, dim=1, index=indices).squeeze(1)
        
        # 将空序列产生的 embedding 置零
        zero_mask = (seq_lens_tensor == 0).unsqueeze(1)
        dyn_graph_embed = dyn_graph_embed.masked_fill(zero_mask, 0.0)

        # ====== 10. 最终投影 ======
        dyn_graph_embed = self.out_proj(dyn_graph_embed)

        return dyn_graph_embed

    def _process_edge_attr(self, edge_attr):
        if self.edge_weight_mode == 'full':
            return edge_attr
        elif self.edge_weight_mode == 'none':
            return None
        elif self.edge_weight_mode == 'random':
            if edge_attr is None: return None
            return edge_attr[torch.randperm(edge_attr.size(0), device=edge_attr.device)]
        elif self.edge_weight_mode == 'norm':
            # 将所有边的权重设置为相同的值（例如，1.0），然后归一化
            edge_attr_normalized = torch.ones_like(edge_attr)  # 赋值为1.0
            edge_attr_normalized = edge_attr_normalized / edge_attr_normalized.sum()  # 归一化
            return edge_attr_normalized
        return edge_attr