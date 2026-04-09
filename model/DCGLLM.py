import torch.nn as nn
import torch.nn.functional as F
from model.RNN import RNN
import math
import torch
from model.GraphEncoder import GraphEncoder
from model.StageNet import StageNet
from model.hita_transformer import HitaTransformer
from model.Transformer import TransformerModel
from model.EvolveGCN import DySATDynamicGraphEncoder, DySATDynamicGraphEncoder_V1,DySATDynamicGraphEncoder_V2


import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class CrossAttentionBlock(nn.Module):
    """
    实现了您图中 所示的交叉注意力模块。
    它包含: MHA -> Add & Norm -> FeedForward -> Add & Norm
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        """
        初始化交叉注意力块。
        参数:
        d_model (int): 模型的嵌入维度 (patient_embed 和 graph_embed 的维度)
        n_heads (int): 多头注意力的头数 (例如 8)
        d_ff (int): FeedForward 层的内部维度 (通常是 d_model * 4)
        dropout (float): Dropout 比例
        """
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4  # FF层默认维度

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        # FeedForward 网络
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 规范化层 (LayerNorm 更适用于Transformer结构)
        # 注意：图中 是 BatchNorm，但 LayerNorm 在此更标准且常用
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        q (Query), k (Key), v (Value)
        q 来自一个源, k 和 v 来自另一个源
        """

        # 2. 多头交叉注意力
        # attn_output 形状为 (B, 1, D)
        attn_output, _ = self.mha(query, key, value)
        
        # 3. Add & Norm (第一个残差连接)
        x = query + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 4. Feed Forward
        ff_output = self.ff(x)
        
        # 5. Add & Norm (第二个残差连接)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        # 6. 恢复维度: (B, 1, D) -> (B, D)
        return x.squeeze(1)

class DCGLLM(nn.Module):
    def __init__(self, args,
                 **kwargs):
        super(DCGLLM, self).__init__()
        self.device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
        self.dataset = kwargs["dataset"]
        self.feature_keys=kwargs["feature_keys"]
        self.label_key=kwargs["label_key"]
        self.mode=kwargs["mode"]
        self.modeltype = args.modeltype
        self.train_dropout_rate = args.train_dropout_rate
        self.hidden_dim = args.hidden_dim
        self.embed_dim = args.embed_dim
        self.use_dynamic_graph = args.use_dynamic_graph
        self.use_cross_attention = args.use_cross_attention
        
        # Initialize loss function
        self.use_focal_loss = args.use_focal_loss
        self.use_weighted_loss = args.use_weighted_loss
        
        if self.use_focal_loss:
            # self.loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
            self.pos_weight = None
        elif self.use_weighted_loss:
            self.loss_fn = None
            self.pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32)
        else:
            self.loss_fn = None
            self.pos_weight = None

        if args.modeltype == "GRU":
            self.model = RNN(
                dataset=self.dataset,
                feature_keys=self.feature_keys,
                label_key=self.label_key,
                mode=self.mode,
                embedding_dim = self.embed_dim,
                hidden_dim = self.hidden_dim,
                train_dropout_rate = self.train_dropout_rate,
                rnn_type = "GRU",
                num_layers= args.encoder_layer
            )
            self.patient_dim = len(self.model.feature_keys) * self.model.hidden_dim

        elif args.modeltype == "StageNet":
            time_keys = None
            if "delta_days" in self.dataset.input_info:
                time_keys = ["delta_days"] * len(self.feature_keys)
            self.model = StageNet(
                dataset=self.dataset,
                feature_keys=self.feature_keys,
                label_key=self.label_key,
                mode=self.mode,
                time_keys=time_keys,
                embedding_dim = self.embed_dim,
                train_dropout_rate = self.train_dropout_rate,
                chunk_size = args.chunk_size,
                levels = args.levels
            )
            self.patient_dim = len(self.model.feature_keys) * args.chunk_size * args.levels

        elif args.modeltype == "HiTANet":
            self.model = HitaTransformer(
                dataset=self.dataset,
                feature_keys=self.feature_keys,
                label_key=self.label_key,
                mode=self.mode,
                embedding_dim = self.embed_dim,
                train_dropout_rate = self.train_dropout_rate,
                num_layers= args.encoder_layer,
                num_heads = args.encoder_head,
                device = self.device
            )
            self.patient_dim = len(self.model.feature_keys) * self.model.embedding_dim
        
        elif args.modeltype == "Transformer":
            self.model = TransformerModel(
                dataset=self.dataset,
                feature_keys=self.feature_keys,
                label_key=self.label_key,
                mode=self.mode,
                embedding_dim = self.embed_dim,
                hidden_dim = self.hidden_dim,
                train_dropout_rate = self.train_dropout_rate,
                num_layers= args.encoder_layer,
                num_heads = args.encoder_head
            )
            self.patient_dim = len(self.model.feature_keys) * self.model.embedding_dim
        
        self.node_num = kwargs["node_num"]
        self.gencoder_dim_list = [args.node_dim] + eval(args.gencoder_dim_list)
        self.n_layers_gencoder = len(eval(args.gencoder_dim_list))
        self.node_embedding = nn.Embedding(self.node_num, args.node_dim)
        
        # 构造疾病-疾病节点图后的图编码器
        self.graph_model = GraphEncoder(self.gencoder_dim_list, self.n_layers_gencoder)
        
        # 批量并行动态图编码器（参考 GraphEncoder 风格）
        # self.dynamic_graph_model = DySATDynamicGraphEncoder(
        #     in_dim=2,                                    # 节点特征：cfipf + participation
        #     gnn_dim=self.gencoder_dim_list[-1],          # 结构 GNN 输出维度，和静态 GraphEncoder 对齐
        #     out_dim=self.gencoder_dim_list[-1],          # 最终输出维度，直接对齐 classifier 预期
        #     num_heads=args.encoder_head if hasattr(args, "encoder_head") else 4,
        #     num_temporal_layers=1,
        #     dropout=self.train_dropout_rate,
        #     max_timesteps=None,                          # 如果太慢可以改成 3/5，仅用最近K次visit
        #     edge_weight_mode=getattr(args, 'edge_weight_mode', 'full'),  # 消融实验参数
        # )

        self.dynamic_graph_model = DySATDynamicGraphEncoder_V1(
            gnn_dim=self.gencoder_dim_list[-1],          # 结构 GNN 输出维度，和静态 GraphEncoder 对齐
            out_dim=self.gencoder_dim_list[-1],          # 最终输出维度，直接对齐 classifier 预期
            num_heads=args.encoder_head if hasattr(args, "encoder_head") else 4,
            num_temporal_layers=1,
            dropout=self.train_dropout_rate,
            max_timesteps=None,                          # 如果太慢可以改成 3/5，仅用最近K次visit
            edge_weight_mode=getattr(args, 'edge_weight_mode', 'full'),  # 消融实验参数
        )
        # 如果使用动态图，添加维度投影层
        # patient_dim + hidden_dim → patient_dim + gencoder_dim_list[-1]
        if self.use_dynamic_graph:
            self.dynamic_proj = nn.Linear(
                self.patient_dim + args.hidden_dim,
                self.patient_dim + self.gencoder_dim_list[-1]
            )
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.train_dropout_rate)

        # 在最后的特征融合层添加交叉注意力模块取代简单的Concat+MLP分类
        # 使用双向交叉注意力进行特征融合
        self.proj_patient = nn.Linear(self.patient_dim, self.embed_dim // 2)
        self.proj_graph = nn.Linear(self.gencoder_dim_list[-1], self.embed_dim // 2)
        d_half = self.embed_dim // 2

        # self.cross_attn_1 = CrossAttentionBlock(
        #     d_model=d_half,
        #     n_heads=args.encoder_head,
        #     d_ff=4 * d_half,
        #     dropout=self.train_dropout_rate
        # )

        # self.cross_attn_2 = CrossAttentionBlock(
        #     d_model=d_half,
        #     n_heads=args.encoder_head,
        #     d_ff=4 * d_half,
        #     dropout=self.train_dropout_rate
        # )
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=self.embed_dim // 2  , num_heads=args.encoder_head, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=self.embed_dim //2 , num_heads=args.encoder_head, batch_first=True)
        self.fusion_proj = nn.Linear(2 * self.embed_dim, self.embed_dim)

        output_size = self.model.get_output_size(self.model.label_tokenizer)
        assert len(self.gencoder_dim_list) > 0
        # 统一使用 patient_dim + graph_dim 作为 classifier 输入维度
        # self.classfier_fc = nn.Linear(self.patient_dim + self.gencoder_dim_list[-1], output_size)
        self.classfier_fc = nn.Linear((self.patient_dim + self.gencoder_dim_list[-1]) // 2, output_size)
        
        # 门控机制：用于动态控制 graph_embed 的贡献
        # patient_embed 是主体，graph_embed 是辅助
        self.gate_fc = nn.Sequential(
            nn.Linear(self.patient_dim + self.gencoder_dim_list[-1], self.gencoder_dim_list[-1]),
            nn.Tanh(),
            nn.Linear(self.gencoder_dim_list[-1], self.gencoder_dim_list[-1]),
            nn.Sigmoid()
        )
        
        # 可学习的超参数：控制 graph_embed 的整体贡献权重
        # 初始化为 1.0，模型可以学习调整这个值
        self.graph_weight = nn.Parameter(torch.tensor(1.0))

    def _init_weight(self):
        self.model.apply(init_weights)
        self.classfier_fc.apply(init_weights())
        nn.init.xavier_uniform_(self.node_embedding)


    def forward(self, graph_batch, dynamic_graph_batch=None, **data):
        patient_embed, patient_emb_all_step = self.model(**data)
        # 真值标签准备
        patient_y_true = self.model.prepare_labels(data[self.model.label_key], self.model.label_tokenizer)
        # 图嵌入表示获得
        # graph_embed = self.graph_model(graph_batch, self.node_embedding)
        
        # 获取图嵌入（动态图或静态图）
        if self.use_dynamic_graph and dynamic_graph_batch is not None:
            graph_embed = self.dynamic_graph_model(dynamic_graph_batch,self.node_embedding)
        else:
            # 如果不使用动态图，使用静态图嵌入
            graph_embed = self.graph_model(graph_batch, self.node_embedding)
        
        if self.use_cross_attention == False:
            # 使用门控机制：patient_embed 是主体，graph_embed 是辅助
            # 根据 patient_embed 和 graph_embed 计算门控值
            # concat_features = torch.cat((patient_embed, graph_embed), dim=1)
            # gate = self.gate_fc(concat_features)  # gate shape: (batch, graph_dim)
            # 对 graph_embed 进行门控
            # gated_graph_embed = self.graph_weight * gate * graph_embed
            # 使用可学习的超参数进一步控制 graph_embed 的整体贡献
            # 拼接主体特征和门控后的辅助特征
            # all_embed = torch.cat((patient_embed, gated_graph_embed), dim=1)
            # Project to half dims first to match classifier input dim
            patient_proj = self.proj_patient(patient_embed)
            graph_proj = self.proj_graph(graph_embed)
            all_embed = torch.cat((patient_proj, graph_proj), dim=1)
        else:
            # 交叉注意力：投影到统一维度
            patient_proj = self.proj_patient(patient_embed).unsqueeze(1)  # [B, 1, embed_dim]
            graph_proj = self.proj_graph(graph_embed).unsqueeze(1)  # [B, 1, embed_dim]
            
            # 双向交叉注意力
            # patient作为query，graph作为key/value
            attn_output_1, _ = self.cross_attn_1(query=patient_proj, key=graph_proj, value=graph_proj)
            # attn_output_1 = self.cross_attn_1(query=patient_proj, key=graph_proj, value=graph_proj)
            # graph作为query，patient作为key/value
            attn_output_2, _ = self.cross_attn_2(query=graph_proj, key=patient_proj, value=patient_proj)
            # attn_output_2 = self.cross_attn_2(query=graph_proj, key=patient_proj, value=patient_proj)
            
            # 拼接并投影
            all_embed = torch.cat((attn_output_1, attn_output_2), dim=-1).squeeze(dim=1)  # [B, 2 * embed_dim]
            
            # # 保持你的输出维度一致
            # all_embed = self.fusion_proj(all_embed)  # [B, embed_dim]


        all_embed = self.dropout(all_embed)
        cls_logits = self.classfier_fc(all_embed)

        patient_y_prob = self.model.prepare_y_prob(cls_logits)
        
        # Use appropriate loss function
        if self.use_focal_loss:
            loss_cls = self.loss_fn(cls_logits, patient_y_true)
        elif self.use_weighted_loss:
            pos_weight = self.pos_weight.to(cls_logits.device)
            loss_cls = F.binary_cross_entropy_with_logits(
                cls_logits, patient_y_true, pos_weight=pos_weight
            )
        else:
            loss_cls = F.binary_cross_entropy_with_logits(cls_logits, patient_y_true)

        return loss_cls, cls_logits, patient_y_true, patient_y_prob, patient_embed