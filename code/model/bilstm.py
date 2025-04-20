import torch
import torch.nn as nn

class DualEncoderWithBiLSTM(nn.Module):
    def __init__(self, hidden_size=384, num_layers=1, dropout=0.5):
        super().__init__()

        # BiLSTM编码器
        self.bilstm = nn.LSTM(
            input_size=768,  # 输入维度与BERT输出一致
            hidden_size=hidden_size // 2,  # 因双向拼接，实际隐藏单元数减半
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 可选：调整BiLSTM输出维度（若需要与其他模块对齐）
        self.proj = nn.Linear(hidden_size, hidden_size*2)  # 384 -> 384（保持维度不变）

    def forward(self, bert_hidden):
        # BiLSTM编码
        lstm_out, _ = self.bilstm(bert_hidden)  # [batch_size, seq_len, hidden_size=384]
        lstm_out = self.proj(lstm_out)  # 投影到目标维度（可选）

        return lstm_out