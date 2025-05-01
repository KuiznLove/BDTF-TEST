import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractiveAttention(nn.Module):
    """交互注意力层"""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 语义特征变换
        self.Wq_semantic = nn.Linear(hidden_dim, hidden_dim)
        self.Wk_semantic = nn.Linear(hidden_dim, hidden_dim)

        # 语法特征变换
        self.Wk_syntax = nn.Linear(hidden_dim, hidden_dim)
        self.Wv_syntax = nn.Linear(hidden_dim, hidden_dim)

        # 正则化组件
        self.dropout = nn.Dropout(dropout)
        self.norm_semantic = nn.LayerNorm(hidden_dim)
        self.norm_syntax = nn.LayerNorm(hidden_dim)

        # 门控机制
        self.gate_semantic = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_syntax = nn.Linear(2 * hidden_dim, hidden_dim)

        # 动态融合权重
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def split_heads(self, x):
        """分割多头"""
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        """合并多头"""
        B, H, L, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def compute_attention(self, query, key, value):
        """基础注意力计算"""
        # 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(self.dropout(attn_weights), value), attn_weights

    def cross_attention(self, feat_a, feat_b, Wq, Wk, Wv):
        """跨模态注意力计算"""
        q = self.split_heads(Wq(feat_a))  # [B, H, L, D]
        k = self.split_heads(Wk(feat_b))
        v = self.split_heads(Wv(feat_b))

        attn_output, _ = self.compute_attention(q, k, v)
        return self.combine_heads(attn_output)

    def gate_fusion(self, original, enhanced, gate_layer):
        """带残差的门控融合"""
        combined = torch.cat([original, enhanced], dim=-1)
        gate = torch.sigmoid(gate_layer(combined))
        return gate * enhanced + (1 - gate) * original

    def forward(self, h_b, h_p):
        # 保留原始特征
        identity_b = h_b
        identity_p = h_p

        # 语义特征增强
        enhanced_b = self.cross_attention(
            h_b, h_p,
            self.Wq_semantic, self.Wk_syntax, self.Wv_syntax
        )
        h_b_prime = self.gate_fusion(identity_b, enhanced_b, self.gate_semantic)
        h_b_prime = self.norm_semantic(h_b_prime + identity_b)

        # 语法特征增强
        enhanced_p = self.cross_attention(
            h_p, h_b,
            self.Wq_semantic, self.Wk_semantic, self.Wv_syntax
        )
        h_p_prime = self.gate_fusion(identity_p, enhanced_p, self.gate_syntax)
        h_p_prime = self.norm_syntax(h_p_prime + identity_p)

        # 动态权重融合
        fusion_weight = self.fusion_gate(torch.cat([h_b_prime, h_p_prime], dim=-1))
        final_rep = fusion_weight * h_b_prime + (1 - fusion_weight) * h_p_prime

        return final_rep


class InteractiveAttentionLayer(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h_b, h_p):
        """
        交互注意力层前向传播
        参数:
            h_b: 语义特征 [batch_size, seq_len, hidden_dim]
            h_p: 语法特征 [batch_size, seq_len, hidden_dim]
        返回:
            h_b_prime: 增强后的语义特征
            h_p_prime: 增强后的语法特征
        """
        # batch_size, seq_len, hidden_dim = h_b.shape
        # 计算语义自注意力矩阵
        attn_b = torch.matmul(h_b, h_b.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attn_b = F.softmax(attn_b, dim=-1)

        # 计算语法自注意力矩阵
        attn_p = torch.matmul(h_p, h_p.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attn_p = F.softmax(attn_p, dim=-1)

        # 交叉注意力增强
        h_b_prime = self.dropout(torch.matmul(attn_p, h_b)) + h_b  # 用语法注意力增强语义
        h_p_prime = self.dropout(torch.matmul(attn_b, h_p)) + h_p  # 用语义注意力增强语法

        h_out = h_b_prime + h_p_prime
        return h_out

# 改进的交互注意力层
class ImprovedInteractiveAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 多头注意力
        self.mha_b = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate)
        self.mha_p = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate)

        # 层归一化
        self.norm_b1 = nn.LayerNorm(hidden_dim)
        self.norm_p1 = nn.LayerNorm(hidden_dim)
        self.norm_b2 = nn.LayerNorm(hidden_dim)
        self.norm_p2 = nn.LayerNorm(hidden_dim)

        # 门控机制
        self.gate_b = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_p = nn.Linear(hidden_dim * 2, hidden_dim)

        # 特征融合
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h_b, h_p):
        """
        改进的交互注意力层
        参数:
            h_b: 语义特征 [batch_size, seq_len, hidden_dim]
            h_p: 语法特征 [batch_size, seq_len, hidden_dim]
        返回:
            h_out: 融合后的特征
        """
        batch_size, seq_len, _ = h_b.shape

        # 自注意力处理
        h_b_t = h_b.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        h_p_t = h_p.transpose(0, 1)

        # 语义自注意力
        attn_b, _ = self.mha_b(h_b_t, h_b_t, h_b_t)
        attn_b = attn_b.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        h_b = self.norm_b1(h_b + self.dropout(attn_b))

        # 语法自注意力
        attn_p, _ = self.mha_p(h_p_t, h_p_t, h_p_t)
        attn_p = attn_p.transpose(0, 1)
        h_p = self.norm_p1(h_p + self.dropout(attn_p))

        # 交叉注意力增强
        h_b_t = h_b.transpose(0, 1)
        h_p_t = h_p.transpose(0, 1)

        # 用语法注意力增强语义
        cross_b, _ = self.mha_b(h_b_t, h_p_t, h_p_t)
        cross_b = cross_b.transpose(0, 1)

        # 用语义注意力增强语法
        cross_p, _ = self.mha_p(h_p_t, h_b_t, h_b_t)
        cross_p = cross_p.transpose(0, 1)

        # 门控融合
        gate_b = torch.sigmoid(self.gate_b(torch.cat([h_b, cross_b], dim=-1)))
        h_b_prime = self.norm_b2(h_b + self.dropout(gate_b * cross_b))

        gate_p = torch.sigmoid(self.gate_p(torch.cat([h_p, cross_p], dim=-1)))
        h_p_prime = self.norm_p2(h_p + self.dropout(gate_p * cross_p))

        # 最终特征融合
        h_out = self.fusion(torch.cat([h_b_prime, h_p_prime], dim=-1))

        return h_out