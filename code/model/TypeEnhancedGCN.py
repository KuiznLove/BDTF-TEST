import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeEnhancedGCN(nn.Module):
    """
    依赖类型增强的图卷积网络（TE-GCN）模块
    实现论文中的公式4-9，通过依赖类型信息增强语法图卷积过程

    功能：
    - 将依赖类型转换为嵌入向量（公式4）
    - 组合节点特征和依赖类型嵌入（公式5-6）
    - 计算语法相关性得分（公式7）
    - 生成增强的邻接矩阵（公式8）
    - 多层图卷积操作（公式9）
    """

    def __init__(self, hidden_dim, dep_type_dim, out_dim, ds_dim=768, num_dep_types=45, num_layers=1):
        """
        初始化TE-GCN模块

        参数：
        hidden_dim   : BiLSTM输出的节点嵌入维度
        dep_type_dim : 依赖类型嵌入维度
        ds_dim       : 中间特征表示维度
        num_dep_types: 依赖类型词典大小
        num_layers   : GCN层数（默认2层）
        """
        super(TypeEnhancedGCN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 依赖类型嵌入层（公式4）
        self.dep_type_embedding = nn.Embedding(num_dep_types, dep_type_dim)

        # 线性变换层（公式5-6）
        self.W1 = nn.Linear(hidden_dim + dep_type_dim, ds_dim)
        self.W2 = nn.Linear(hidden_dim + dep_type_dim, ds_dim)

        self.Wo = nn.Linear(ds_dim, out_dim)

        # GCN层参数（公式9）
        self.gcn_layers = nn.ModuleList()  # 线性变换层列表
        self.gcn_biases = nn.ParameterList()  # 偏置项列表

        # 初始化各层参数
        for l in range(num_layers):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.gcn_biases.append(nn.Parameter(torch.zeros(hidden_dim)))

    def forward(self, h, dep_type_matrix):
        """
        参数：
        h              : bert输出的节点特征，形状 [batch_size, seq_len, hidden_dim]
        dep_type_matrix: 依赖类型ID矩阵，形状 [batch_size, seq_len, seq_len]

        返回：
        更新后的节点表示，形状 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = h.size()

        adj_matrix = (dep_type_matrix != 0).int()

        # === 依赖类型嵌入（公式4） ===
        flat_dep_types = dep_type_matrix.reshape(-1)  # 展平依赖类型矩阵
        dep_type_emb = self.dep_type_embedding(flat_dep_types)  # 嵌入查询
        dep_type_emb = dep_type_emb.view(batch_size, seq_len, seq_len, -1)  # 恢复形状

        # === 构建节点对表示 ===
        # 扩展节点特征维度用于配对计算
        h_i = h.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq, seq, hidden]
        h_j = h.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, seq, seq, hidden]

        # === 组合特征和依赖嵌入（公式5-6） ===
        h_i_prime = torch.cat([h_i, dep_type_emb], dim=-1)  # 拼接节点i特征和依赖嵌入
        h_j_prime = torch.cat([h_j, dep_type_emb], dim=-1)  # 拼接节点j特征和依赖嵌入

        # === 线性变换 ===
        # 展平处理用于批量线性变换
        h_i_prime_flat = h_i_prime.reshape(-1, self.hidden_dim + dep_type_emb.size(-1))
        h_j_prime_flat = h_j_prime.reshape(-1, self.hidden_dim + dep_type_emb.size(-1))

        # 应用线性变换（公式5-6）
        h_i_transformed = self.W1(h_i_prime_flat).view(batch_size, seq_len, seq_len, -1)
        h_j_transformed = self.W2(h_j_prime_flat).view(batch_size, seq_len, seq_len, -1)

        # === 计算相关性得分（公式7） ===
        relevance = torch.sum(h_i_transformed * h_j_transformed, dim=-1)

        # === 邻接矩阵归一化（公式8） ===
        masked_relevance = relevance * adj_matrix  # 应用邻接矩阵掩码
        masked_relevance = masked_relevance.masked_fill(adj_matrix == 0, -1e9)  # 填充负无穷
        A = F.softmax(masked_relevance, dim=2)  # 行方向softmax归一化

        # === 多层图卷积（公式9） ===
        h_syn = h  # 初始化为输入特征
        for l in range(self.num_layers):
            # 邻居特征聚合
            aggregated = torch.bmm(A, h_syn)
            # 线性变换 + 偏置
            transformed = self.gcn_layers[l](aggregated) + self.gcn_biases[l].unsqueeze(0).unsqueeze(1)
            # 非线性激活
            h_syn = F.relu(transformed)

        # assert not torch.isnan(h_syn).any, "h_syn包含NaN"
        #------------------------------------
        h_syn = self.Wo(h_syn)  # 输出层线性变换
        #------------------------------------

        return h_syn

# 新
class TypeEnhancedGCNDeepseek(nn.Module):
    def __init__(self, in_dim, out_dim, dep_type_num=45, dep_emb_dim=16):
        super().__init__()
        self.dep_embedding = nn.Embedding(dep_type_num, dep_emb_dim)
        self.W1 = nn.Linear(in_dim + dep_emb_dim, out_dim)
        self.W2 = nn.Linear(in_dim + dep_emb_dim, out_dim)
        self.W_gcn = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W_gcn.weight)

    def forward(self, h, dep_type_matrix, adj=None, aspect_mask=None):
        batch_size, seq_len, _ = h.size()
        if adj is None:
            adj = (dep_type_matrix != 0).int()

        if aspect_mask is not None:
            h = position_aware_transformation(h, aspect_mask)

        dep_emb = self.dep_embedding(dep_type_matrix)
        h_expanded = h.unsqueeze(2).expand(-1, -1, seq_len, -1)
        combined = torch.cat([h_expanded, dep_emb], dim=-1)

        h_prime_i = self.W1(combined)
        h_prime_j = self.W2(combined)

        # 计算缩放后的点积得分
        d_k = h_prime_i.size(-1)
        scores_raw = torch.sum(h_prime_i * h_prime_j, dim=-1) / (d_k ** 0.5)

        # 最大值归一化
        max_scores = torch.max(scores_raw, dim=-1, keepdim=True).values
        scores = torch.exp(scores_raw - max_scores)  # 数值稳定

        # 应用邻接矩阵过滤
        masked_scores = scores * adj

        # 行归一化
        norm_factor = torch.sum(masked_scores, dim=-1, keepdim=True) + 1e-6
        A = masked_scores / norm_factor

        # 图卷积操作
        h_trans = self.W_gcn(h)
        output = torch.bmm(A, h_trans) + self.bias
        output = F.relu(output)

        return output

#旧
class TypeEnhancedGCNDeepseek0(nn.Module):
    def __init__(self, in_dim, out_dim, dep_type_num=45, dep_emb_dim=16):
        super().__init__()
        # 依赖类型嵌入层
        self.dep_embedding = nn.Embedding(dep_type_num, dep_emb_dim)

        # 节点特征转换层
        self.W1 = nn.Linear(in_dim + dep_emb_dim, out_dim)
        self.W2 = nn.Linear(in_dim + dep_emb_dim, out_dim)

        # 图卷积参数
        self.W_gcn = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # 初始化参数
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W_gcn.weight)

    def forward(self, h, dep_type_matrix, adj=None, aspect_mask=None):
        """
        h: 节点特征 [batch_size, seq_len, in_dim]
        adj: 原始邻接矩阵 [batch_size, seq_len, seq_len]
        dep_type_matrix: 依赖类型矩阵 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = h.size()
        if adj is None:
            adj = (dep_type_matrix != 0).int()

        if aspect_mask is not None:
            h = position_aware_transformation(h, aspect_mask) #应用位置变换
        # 1. 获取依赖类型嵌入
        dep_emb = self.dep_embedding(dep_type_matrix)  # [batch, seq, seq, dep_emb_dim]

        # 2. 构建增强的邻接矩阵
        h_expanded = h.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq, seq, in_dim]

        combined = torch.cat([h_expanded, dep_emb], dim=-1)  # [batch, seq, seq, in_dim+dep_emb]

        # 计算相关性得分
        h_prime_i = self.W1(combined)  # [batch, seq, seq, out_dim]
        h_prime_j = self.W2(combined)  # [batch, seq, seq, out_dim]

        # 相关性得分计算
        scores = torch.exp(torch.sum(h_prime_i * h_prime_j, dim=-1))  # [batch, seq, seq]

        # 应用原始邻接矩阵过滤
        masked_scores = scores * adj
        # assert not torch.isinf(masked_scores).any(), "masked_scores中存在inf"

        # 行归一化
        norm_factor = torch.sum(masked_scores, dim=-1, keepdim=True) + 1e-6
        A = masked_scores / norm_factor  # [batch, seq, seq]
        assert not torch.isnan(A).any(), "A包含NaN"

        # 3. 图卷积操作
        h_trans = self.W_gcn(h)  # [batch, seq, out_dim]
        output = torch.bmm(A, h_trans) + self.bias  # [batch, seq, out_dim]

        output = F.relu(output)
        # assert not torch.isnan(output).any(), "output包含NaN"

        return output

def position_aware_transformation(hidden_states, aspect_mask):
    """
    hidden_states: 形状为(batch_size, seq_len, hidden_dim)的张量，表示当前层的隐藏状态。
    aspect_mask: 形状为(batch_size, seq_len)的布尔张量，True表示对应位置属于aspect词。
    返回：应用位置权重后的隐藏状态，形状同hidden_states。
    """
    batch_size, seq_len, hidden_dim = hidden_states.size()
    device = hidden_states.device

    # 从aspect_mask中提取aspect的起始和结束位置
    start = torch.argmax(aspect_mask.int(), dim=1)  # (batch_size,)
    flipped_mask = torch.flip(aspect_mask.int(), dims=[1])
    end = (seq_len - 1) - torch.argmax(flipped_mask, dim=1)  # (batch_size,)

    # 生成位置索引 [0, 1, ..., seq_len-1]
    pos_indices = torch.arange(seq_len, device=device).expand(batch_size, seq_len)  # (batch_size, seq_len)

    # 扩展start和end以匹配维度
    aspect_start = start.unsqueeze(1)  # (batch_size, 1)
    aspect_end = end.unsqueeze(1)  # (batch_size, 1)

    # 计算左侧权重
    left_mask = pos_indices < aspect_start
    left_distance = aspect_start - pos_indices  # 正值距离
    left_weights = 1.0 - (left_distance.float() / seq_len)

    # 计算右侧权重
    right_mask = pos_indices > aspect_end
    right_distance = pos_indices - aspect_end  # 正值距离
    right_weights = 1.0 - (right_distance.float() / seq_len)

    # aspect区域内的掩码（直接使用输入参数）
    aspect_mask = aspect_mask.bool()

    # 合并权重
    position_weights = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
    position_weights[left_mask] = left_weights[left_mask]
    position_weights[right_mask] = right_weights[right_mask]
    position_weights[aspect_mask] = 0.0  # 直接使用传入的aspect_mask覆盖

    # 扩展维度以便广播
    position_weights = position_weights.unsqueeze(2)  # (batch_size, seq_len, 1)

    # 应用位置感知转换
    transformed_states = hidden_states * position_weights

    return transformed_states