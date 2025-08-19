import torch
import torch.nn as nn
# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.01)  # 添加dropout层，丢弃率为0.5
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = None

    def init_hidden(self, batch_size):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)

    def forward(self, x):
        if self.hidden is None:
            self.hidden = self.init_hidden(batch_size=x.size(0))
        out, self.hidden = self.lstm(x, self.hidden)
        self.hidden = None
        out = self.dropout(out)  # 在LSTM输出上应用dropout
        out = self.fc(out[:, -1, :])
        return out
