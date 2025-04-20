import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.activations import ACT2FN


class Seq2Mat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.norm   = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act] #使用 ACT2FN 根据配置选择激活函数（如 ReLU、GELU 等）

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H] 批次大小、序列长度、特征维度
        """
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        t = torch.cat([x, y], dim=-1)
        t = self.W(t)
        t = self.activation(t)
        return t


class ContextSeq2Mat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.norm   = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H]
        """
        xmat = x.clone()
        batch_size = xmat.shape[0]
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        
        max_len = xmat.shape[1]
        xmat_t = xmat.transpose(1, 2)
        context = torch.ones_like(x).to('cuda')
        for i in range(max_len):
            diag = x.diagonal(dim1=1, dim2=2, offset=-i)
            xmat_t = torch.max(xmat_t[:, :, :max_len-i], diag)
            bb = [[b] for b in range(batch_size)]
            linexup = [[j for j in range(max_len-i)] for b in range(batch_size)] 
            lineyup = [[j+i for j in range(max_len-i)] for b in range(batch_size)] 
            linexdown = [[j+i for j in range(max_len-i)] for b in range(batch_size)]
            lineydown = [[j for j in range(max_len-i)] for b in range(batch_size)]   
            context[bb, linexup, lineyup, :] = xmat_t.permute(0, 2, 1)
            context[bb, linexdown, lineydown, :] = xmat_t.permute(0, 2, 1)
        
        t = torch.cat([x, y, context], dim=-1)
        t = self.W(t)
        t = self.activation(t)
        return t


class TensorSeq2Mat(nn.Module):
    """
    refernce: SOCHER R, PERELYGIN A, WU J, 等. Recursive deep models for semantic compositionality over a sentiment treebank[C]//Proceedings of the 2013 conference on empirical methods in natural language processing. 2013: 1631-1642.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h = config.num_attention_heads
        self.d = config.num_d
        self.W = nn.Linear(2*config.hidden_size+self.d, config.hidden_size)
        self.V = nn.Parameter(torch.Tensor(self.d, config.hidden_size, config.hidden_size))
        self.norm = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]
        self.init_weights()

    def init_weights(self):
        self.V.data.normal_(mean=0.0, std=self.config.initializer_range)

    def rntn(self, x, y):
        t = torch.cat([x, y], dim=-1)
        xv = torch.einsum('b m n p, k p d -> b m n k d', x, self.V)
        xvy = torch.einsum('b m n k d, b m n d -> b m n k', xv, y)
        t = torch.cat([t, xvy], dim=-1)
        tw = self.W(t)
        return tw

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H]
        """
        seq = x
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        t = self.rntn(x, y)
        t = self.activation(t)
        return t


class TensorcontextSeq2Mat(nn.Module):
    """
    refernce: SOCHER R, PERELYGIN A, WU J, 等. Recursive deep models for semantic compositionality over a sentiment treebank[C]//Proceedings of the 2013 conference on empirical methods in natural language processing. 2013: 1631-1642.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h = config.num_attention_heads
        self.d = config.num_d
        self.W = nn.Linear(3*config.hidden_size+self.d, config.hidden_size)
        self.V = nn.Parameter(torch.Tensor(self.d, config.hidden_size, config.hidden_size))
        self.norm   = T5LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]
        self.init_weights()

    def init_weights(self):
        if self.config.model_type=='bart' or self.config.model_type=='t5':
            self.V.data.normal_(mean=0.0, std=0.02)
        else:
            self.V.data.normal_(mean=0.0, std=self.config.initializer_range)

    def rntn(self, x, y, xmat):
        max_len = xmat.shape[1]
        xmat_t = xmat.transpose(1, 2)
        batch_size = xmat.shape[0]
        context = torch.ones_like(x).to('cuda')
        # context = torch.ones_like(x)
        for i in range(max_len):
            diag = x.diagonal(dim1=1, dim2=2, offset=-i)
            #---------------------------------------------------------------------
            # 检查数据类型一致性（新增断言）
            assert diag.dtype == xmat_t.dtype, \
                f"Data type mismatch: diag ({diag.dtype}) vs xmat_t ({xmat_t.dtype})"
            assert not torch.isnan(xmat_t).any(), "xmat_t contains NaN!"
            assert not torch.isinf(xmat_t).any(), "xmat_t contains Inf!"
            assert not torch.isnan(diag).any(), "diag contains NaN!"
            assert not torch.isinf(diag).any(), "diag contains Inf!"
            #----------------------------------------------------------------------
            xmat_t = torch.max(xmat_t[:, :, :max_len-i], diag)
            bb = [[b] for b in range(batch_size)]
            linexup = [[j for j in range(max_len-i)] for b in range(batch_size)] 
            lineyup = [[j+i for j in range(max_len-i)] for b in range(batch_size)] 
            linexdown = [[j+i for j in range(max_len-i)] for b in range(batch_size)]
            lineydown = [[j for j in range(max_len-i)] for b in range(batch_size)]
            try:
                context[bb, linexup, lineyup, :] = xmat_t.permute(0, 2, 1)
                context[bb, linexdown, lineydown, :] = xmat_t.permute(0, 2, 1)
            except Exception as e:
                print("max_len:", max_len ,"\n")
                print("i:", i ,"\n")
                print("diag:", diag.shape ,"\n")
                print("bb:", bb ,"\n")
                print("linexup:", linexup ,"\n")
                print("lineyup:", lineyup ,"\n")
                print("context:", context.shape ,"\n")
                print("xmat_t:", xmat_t.shape ,"\n")

        t = torch.cat([x, y, context], dim=-1)
        xvy = torch.einsum('b m n p, k p d, b m n d -> b m n k', x, self.V, y)
        t = torch.cat([t, xvy], dim=-1)
        tw = self.W(t)
        return tw

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H]
        """
        xmat = x
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        t = self.rntn(x, y, xmat)
        t = self.activation(t)
        return t


if __name__ == "__main__":
    # 定义配置类并在 main 中实例化
    class Config:
        def __init__(self):
            self.hidden_size = 64
            self.layer_norm_eps = 1e-6
            self.hidden_act = 'relu'
            self.num_attention_heads = 8
            self.num_d = 32
            self.initializer_range = 0.02
            self.model_type = 't5'

    # 初始化配置
    config = Config()
    
    # 设置设备（如果有可用的 GPU，则使用它）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输入张量，形状为 [B, L, H]，并将其移动到指定设备上
    B, L, H = 2, 10, config.hidden_size
    x = torch.rand(B, L, H).to(device)
    y = torch.rand(B, L, H).to(device)
    
    # 1. 测试 Seq2Mat 类
    seq2mat = Seq2Mat(config).to(device)
    output_seq2mat = seq2mat(x, y)
    print("Seq2Mat Output Shape:", output_seq2mat.shape)  # 预期输出形状: [B, L, L, H]
    print(output_seq2mat)
    
    # 2. 测试 ContextSeq2Mat 类
    context_seq2mat = ContextSeq2Mat(config).to(device)
    output_context_seq2mat = context_seq2mat(x, y)
    print("ContextSeq2Mat Output Shape:", output_context_seq2mat.shape)  # 预期输出形状: [B, L, L, H]
    print(output_context_seq2mat)
    
    # 3. 测试 TensorSeq2Mat 类
    tensor_seq2mat = TensorSeq2Mat(config).to(device)
    output_tensor_seq2mat = tensor_seq2mat(x, y)
    print("TensorSeq2Mat Output Shape:", output_tensor_seq2mat.shape)  # 预期输出形状: [B, L, L, H]
    print(output_tensor_seq2mat)
    
    # 4. 测试 TensorcontextSeq2Mat 类
    tensor_context_seq2mat = TensorcontextSeq2Mat(config).to(device)
    output_tensor_context_seq2mat = tensor_context_seq2mat(x, y)
    print("TensorcontextSeq2Mat Output Shape:", output_tensor_context_seq2mat.shape)  # 预期输出形状: [B, L, L, H]
    print(output_tensor_context_seq2mat)
