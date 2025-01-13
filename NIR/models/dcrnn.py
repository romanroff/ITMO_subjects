import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import DCRNN

class DCRNNModel(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, hidden_channels, num_layers, K):
        super(DCRNNModel, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.K = K

        # DCRNN слой
        self.dcrnn = DCRNN(
            in_channels=in_channels,  # Количество входных каналов
            out_channels=hidden_channels,
            K=K,
        )

        # Выходной слой
        self.output_layer = nn.Linear(hidden_channels, out_channels)  # Количество выходных каналов

    def forward(self, x, edge_index):
        # x: [batch, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, _ = x.shape

        # Инициализация скрытого состояния
        h = torch.zeros(self.num_layers, batch_size * num_nodes, self.hidden_channels).to(x.device)

        # Обработка временных шагов
        outputs = []
        for t in range(seq_len):
            # Вход на текущий временной шаг
            x_t = x[:, t, :, :].reshape(batch_size * num_nodes, self.in_channels)

            # Обновление скрытого состояния
            h = self.dcrnn(x_t, edge_index, h)  # edge_weight не используется

            # Выход на текущий временной шаг
            out = self.output_layer(h[-1])  # Используем последний слой RNN
            out = out.reshape(batch_size, num_nodes, self.out_channels)
            outputs.append(out)

        # Собираем выходы по временным шагам
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, num_nodes, out_channels]
        return outputs