import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from typing import Dict, Tuple


class TypeAwareEdgeEncoder(nn.Module):
    r"""依赖类型编码器：将依赖类型映射为嵌入向量"""

    def __init__(self, type_vocab_size: int, embed_dim: int):
        super(TypeAwareEdgeEncoder, self).__init__()
        self.type_embedding = nn.Embedding(type_vocab_size, embed_dim)

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        return self.type_embedding(edge_types)


class TypeAwareHAGATLayer(nn.Module):
    r"""融入类型感知的异构图注意力层"""

    def __init__(
            self,
            node_types: list[str],
            edge_types: list[tuple[str, str, str]],
            embed_dim: int,
            num_heads: int,
            type_vocab_size: int,
            use_type_aware: bool = True
    ):
        super(TypeAwareHAGATLayer, self).__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_type_aware = use_type_aware

        # 初始化依赖类型编码器
        self.type_encoder = TypeAwareEdgeEncoder(type_vocab_size, embed_dim)

        # 为每种边类型定义GATConv（带类型感知）
        self.convs = nn.ModuleDict()
        for edge_type in edge_types:
            src_type, rel_type, dst_type = edge_type
            self.convs[edge_type] = GATConv(
                (embed_dim, embed_dim),
                embed_dim,
                num_heads=num_heads,
                concat=False
            )

        # 节点特征线性变换
        self.node_transform = nn.ModuleDict({
            nt: nn.Linear(embed_dim, embed_dim * num_heads)
            for nt in node_types
        })

    def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict[tuple[str, str, str], torch.Tensor],
            edge_type_dict: Dict[tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        h_dict = {nt: self.node_transform[nt](x)
                  for nt, x in x_dict.items()}  # (N, H*D) -> (N, H, D)

        out_dict = {nt: torch.zeros_like(h) for nt, h in h_dict.items()}

        for edge_type in self.edge_types:
            src_type, rel_type, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]
            edge_types = edge_type_dict[edge_type]  # 获取边的依赖类型索引

            # 提取源节点和目标节点特征
            src_h = h_dict[src_type][edge_index[0]]  # (E, H, D)
            dst_h = h_dict[dst_type][edge_index[1]]  # (E, H, D)

            # 类型感知处理：将依赖类型嵌入融入注意力计算
            if self.use_type_aware:
                type_embeds = self.type_encoder(edge_types).unsqueeze(1)  # (E, 1, D)
                src_h = torch.cat([src_h, type_embeds], dim=-1)  # 拼接类型信息
                dst_h = torch.cat([dst_h, type_embeds], dim=-1)

            # 计算注意力并聚合
            edge_attention = self.convs[edge_type]((src_h, dst_h), edge_index)
            out_dict[dst_type] = out_dict[dst_type].scatter_add_(
                0, edge_index[1].unsqueeze(-1).repeat(1, self.num_heads, self.embed_dim),
                edge_attention
            )

        # 应用激活函数和归一化
        for nt in self.node_types:
            out_dict[nt] = F.elu(out_dict[nt])
            out_dict[nt] = F.dropout(out_dict[nt], p=0.6, training=self.training)

        return out_dict


class EnhancedHAGNN(nn.Module):
    r"""增强型异构图神经网络（融入类型感知）"""

    def __init__(
            self,
            node_types: list[str],
            edge_types: list[tuple[str, str, str]],
            type_vocab_size: int,
            embed_dim: int = 300,
            num_heads: int = 8,
            num_layers: int = 2
    ):
        super(EnhancedHAGNN, self).__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers

        # 初始化类型感知图注意力层
        self.layers = nn.ModuleList([
            TypeAwareHAGATLayer(
                node_types, edge_types, embed_dim, num_heads, type_vocab_size
            )
            for _ in range(num_layers)
        ])

        # 方面节点初始化（带类型感知增强）
        self.aspect_embedding = nn.Embedding(num_aspects, embed_dim)  # 假设num_aspects为方面数量

    def forward(
            self,
            hetero_data: HeteroData
    ) -> Dict[str, torch.Tensor]:
        x_dict = hetero_data.x_dict
        x_dict['aspect'] = self.aspect_embedding(hetero_data['aspect'].x)  # 初始化方面节点

        for layer in self.layers:
            x_dict = layer(
                x_dict,
                hetero_data.edge_index_dict,
                hetero_data.edge_type_dict  # 传入边类型信息
            )

        return x_dict


# 数据预处理示例：构建带依赖类型的异构图
def build_type_aware_hetero_data(sentences, aspects, dependency_parser):
    data = HeteroData()

    # 节点特征初始化（假设已通过BERT获取词和句子嵌入）
    data['word'].x = word_embeddings  # (V, D)
    data['sentence'].x = sent_embeddings  # (S, D)
    data['aspect'].x = aspect_indices  # (A, ) 方面类别索引

    # 构建边索引和类型（通过依赖解析获取）
    edge_index_dict = {}
    edge_type_dict = {}

    for sent_idx, sentence in enumerate(sentences):
        dependencies = dependency_parser(sentence)  # 获取依赖三元组 (head, dependent, type)
        for head, dep, t_type in dependencies:
            # 词-句边（假设每个词属于一个句子）
            edge_index = torch.tensor([[head], [sent_idx]])
            edge_type = torch.tensor([type_to_idx[t_type]])  # 转换为类型索引
            edge_index_dict[('word', 'in', 'sentence')] = torch.cat(
                [edge_index_dict.get(('word', 'in', 'sentence'), torch.empty((2, 0))), edge_index],
                dim=1
            )
            edge_type_dict[('word', 'in', 'sentence')] = torch.cat(
                [edge_type_dict.get(('word', 'in', 'sentence'), torch.empty((0,))), edge_type],
                dim=0
            )

    # 句-方面边（假设通过标签获取）
    for sent_idx, sent_aspects in enumerate(aspects):
        for aspect_idx in sent_aspects:
            edge_index = torch.tensor([[sent_idx], [aspect_idx]])
            edge_index_dict[('sentence', 'about', 'aspect')] = torch.cat(
                [edge_index_dict.get(('sentence', 'about', 'aspect'), torch.empty((2, 0))), edge_index],
                dim=1
            )
            edge_type_dict[('sentence', 'about', 'aspect')] = torch.cat(
                [edge_type_dict.get(('sentence', 'about', 'aspect'), torch.empty((0,))),
                 torch.tensor([type_to_idx['aspect_relation']])],  # 固定方面关系类型
                dim=0
            )

    data.edge_index_dict = edge_index_dict
    data.edge_type_dict = edge_type_dict
    return data

if __name__ == '__main__':
    # ====================== 超参数配置 ======================
    EMBED_DIM = 128  # 嵌入维度
    NUM_HEADS = 4  # 注意力头数
    NUM_LAYERS = 2  # 图注意力层数
    TYPE_VOCAB_SIZE = 3  # 依赖类型数量（示例：假设3种基础类型）
    BATCH_SIZE = 2  # 模拟批次大小


    # ====================== 模拟数据准备 ======================
    def mock_dependency_parser(sentence: list[str]) -> list[tuple[int, int, str]]:
        r"""模拟依赖解析器：返回(head_idx, dependent_idx, type)三元组"""
        num_words = len(sentence)
        return [
            (i, (i + 1) % num_words, f"type_{i % TYPE_VOCAB_SIZE}")  # 循环生成依赖类型
            for i in range(num_words)
        ]


    # 模拟输入数据
    sentences = [
        ["The", "food", "is", "delicious"],  # 句子1（4个词）
        ["Service", "was", "excellent"]  # 句子2（3个词）
    ]
    aspects = [
        [0, 1],  # 句子1涉及的方面索引（假设0:food, 1:taste）
        [2]  # 句子2涉及的方面索引（假设2:service）
    ]

    # 生成节点特征（模拟BERT输出）
    word_embeddings = torch.randn(len([word for sent in sentences for word in sent]), EMBED_DIM)
    sent_embeddings = torch.randn(len(sentences), EMBED_DIM)
    aspect_indices = torch.tensor([0, 1, 2], dtype=torch.long)  # 3个方面类别

    # ====================== 构建异构图数据 ======================
    data = build_type_aware_hetero_data(
        sentences=sentences,
        aspects=aspects,
        dependency_parser=mock_dependency_parser
    )

    # ====================== 初始化模型与训练配置 ======================
    node_types = ['word', 'sentence', 'aspect']
    edge_types = [('word', 'in', 'sentence'), ('sentence', 'about', 'aspect')]

    model = EnhancedHAGNN(
        node_types=node_types,
        edge_types=edge_types,
        type_vocab_size=TYPE_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )

    # ====================== 前向传播与输出 ======================
    model.eval()
    with torch.no_grad():
        outputs = model(data)

        # 打印关键节点输出维度
        print("=== 模型输出维度 ===")
        print(f"词节点特征: {outputs['word'].shape}")  # (总词数, NUM_HEADS*EMBED_DIM)
        print(f"句子节点特征: {outputs['sentence'].shape}")  # (总句子数, NUM_HEADS*EMBED_DIM)
        print(f"方面节点特征: {outputs['aspect'].shape}")  # (总方面数, NUM_HEADS*EMBED_DIM)

        # 打印具体数值示例（前2个句子节点特征）
        print("\n=== 句子节点特征示例 ===")
        print("句子1特征前5维:", outputs['sentence'][0, :5].numpy())
        print("句子2特征前5维:", outputs['sentence'][1, :5].numpy())

    # ====================== 完整数据流程说明 ======================
    """
    1. 数据准备：
       - 词嵌入：通过BERT/GloVe获取，形状为(总词数, EMBED_DIM)
       - 句子嵌入：通过CNN/LSTM提取，形状为(总句子数, EMBED_DIM)
       - 方面索引：每个方面对应唯一整数索引，形状为(总方面数,)

    2. 依赖解析：
       - 真实场景需替换为实际依赖解析工具（如spaCy的依存句法分析）
       - 解析结果需包含(head词索引, dependent词索引, 依赖类型字符串)

    3. 类型映射：
       - 建立依赖类型到索引的映射（如{'nsubj':0, 'det':1, ...}）
       - 示例中使用模拟类型"type_0", "type_1", "type_2"

    4. 模型输入：
       - HeteroData包含x_dict（节点特征）和edge_index_dict/edge_type_dict（边信息）

    5. 输出解释：
       - 句子节点特征可用于后续情感分类
       - 方面节点特征反映不同方面的语义表示
    """