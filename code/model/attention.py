import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, output_dim, num_heads=1):
        """
        自注意力机制模块
        Args:
            d_model: 输入特征的维度
            num_heads: 注意力头的数量
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 自注意力参数
        self.Wq = nn.Linear(d_model, d_model)  # 查询变换
        self.Wk = nn.Linear(d_model, d_model)  # 键变换

        # 输出变换
        self.Wo = nn.Linear(d_model, output_dim)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, H):
        """
        Args:
            H: 输入特征 [batch_size, seq_len, d_model]
        Returns:
            注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = H.shape

        # ===== 自注意力计算 =====
        Q = self.Wq(H)  # [batch_size, seq_len, d_model]
        K = self.Wk(H)  # [batch_size, seq_len, d_model]

        # ===== 多头处理 =====
        # 重塑形状为 [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ===== 注意力得分计算 =====
        A = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        A = F.softmax(A, dim=-1)

        # ===== 值计算和聚合 =====
        V = H.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.matmul(A, V)

        # 重塑输出
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 最终变换
        return self.Wo(out)

class AspectAwareAttention(nn.Module):
    def __init__(self, d_model, output_dim, num_heads=1):
        """
        方面感知注意力层
        论文：基于方面感知注意力增强的方面情感三元组抽取 - 高龙涛
        Args:
            d_model: 输入特征的维度
            num_heads: 注意力头的数量
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 方面感知注意力参数
        self.Wa = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.bias = nn.Parameter(torch.zeros(1))

        # 自注意力参数
        self.Wq_self = nn.Linear(d_model, d_model)
        self.Wk_self = nn.Linear(d_model, d_model)

        # 输出变换
        # self.Wo = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, output_dim)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, H, aspect_mask):
        """
        Args:
            H: 输入特征 [batch_size, seq_len, d_model]
            aspect_mask: 方面项位置掩码 [batch_size, seq_len]
        Returns:
            注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = H.shape
        # ===== 方面感知注意力计算 =====
        # 提取方面项特征
        aspect_h = H * aspect_mask.unsqueeze(-1)
        aspect_sum = aspect_h.sum(dim=1)  # [batch_size, d_model]
        aspect_count = aspect_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        H_a = aspect_sum / aspect_count.clamp(min=1e-9)  # 平均池化
        # 扩展为序列长度
        H_a = H_a.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        # 线性变换
        Q_a = self.Wa(H_a)  # [batch_size, seq_len, d_model]
        K = self.Wk(H)  # [batch_size, seq_len, d_model]

        # ===== 自注意力计算 =====
        Q_self = self.Wq_self(H)
        K_self = self.Wk_self(H)

        # ===== 多头处理 =====
        # 重塑形状为 [batch_size, num_heads, seq_len, head_dim]
        Q_a = Q_a.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Q_self = Q_self.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_self = K_self.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ===== 注意力得分计算 =====
        # 方面感知注意力得分
        A_asp = torch.matmul(Q_a, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        A_asp = torch.tanh(A_asp + self.bias)
        # 自注意力得分
        A_self = torch.matmul(Q_self, K_self.transpose(-2, -1)) / (self.head_dim  ** 0.5)
        # 合并注意力
        # A = A_asp + A_self
        A = A_asp
        A = F.softmax(A, dim=-1)

        # ===== 值计算和聚合 =====
        V = H.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.matmul(A, V)
        # 重塑输出
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # 最终变换
        return self.Wo(out)


class AspectAwareAttention2(nn.Module):
    def __init__(self, d_model):
        """
        Args:
            d_model: 输入特征的维度
        """
        super().__init__()
        self.d_model = d_model

        # 方面感知注意力参数
        self.Wa = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.bias = nn.Parameter(torch.zeros(1))

        # 自注意力参数
        self.Wq_self = nn.Linear(d_model, d_model)
        self.Wk_self = nn.Linear(d_model, d_model)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, H, aspect_mask):
        """
        Args:
            H: 输入特征 [batch_size, seq_len, d_model]
            aspect_mask: 方面项位置掩码 [batch_size, seq_len]

        Returns:
            注意力得分矩阵 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = H.shape

        # ===== 方面项特征提取 =====
        # 使用mask加权平均替代原始的位置索引
        aspect_h = H * aspect_mask.unsqueeze(-1)
        aspect_sum = aspect_h.sum(dim=1)  # [batch_size, d_model]
        aspect_count = aspect_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        H_a = aspect_sum / aspect_count.clamp(min=1e-9)  # 平均池化 [batch_size, d_model]

        # ===== 方面感知注意力计算 =====
        # 线性变换
        Q_a = self.Wa(H_a).unsqueeze(1)  # [batch_size, 1, d_model]
        K = self.Wk(H)  # [batch_size, seq_len, d_model]

        # 注意力得分计算
        A_asp = torch.bmm(Q_a, K.transpose(1, 2))  # [batch_size, 1, seq_len]
        A_asp = torch.tanh(A_asp + self.bias)

        # ===== 自注意力计算 =====
        Q_self = self.Wq_self(H)  # [batch_size, seq_len, d_model]
        K_self = self.Wk_self(H)  # [batch_size, seq_len, d_model]
        A_self = torch.bmm(Q_self, K_self.transpose(1, 2)) / (self.d_model ** 0.5)

        # ===== 合并注意力 =====
        # 扩展方面感知注意力维度
        A_asp = A_asp.expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]
        A = A_asp + A_self

        # 归一化处理
        return F.softmax(A, dim=-1)