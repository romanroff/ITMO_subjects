import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, seq_len, pre_len):
        super(TGCN, self).__init__()
        self.seq_len = seq_len
        # GCN для обработки пространственных зависимостей
        self.gcn = GCNConv(in_channels, hidden_channels)
        # GRU для обработки временных зависимостей
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        # Полносвязный слой для получения выходных значений
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.pre_len = pre_len

    def forward(self, x, edge_index):
        # x: (batch, seq_len, nodes, channels)
        batch_size, seq_len, num_nodes, _ = x.shape
        # Инициализация скрытого состояния GRU
        h = torch.zeros(1, batch_size * num_nodes, self.gru.hidden_size).to(x.device)

        # Обработка временной последовательности
        outputs = []
        for t in range(seq_len):
            # Текущий временной шаг: (batch, nodes, channels)
            x_t = x[:, t, :, :]
            # Применяем GCN: (batch * nodes, hidden_channels)
            x_t = x_t.reshape(batch_size * num_nodes, -1)  # (batch * nodes, channels)
            x_t = self.gcn(x_t, edge_index)  # (batch * nodes, hidden_channels)
            x_t = torch.relu(x_t)
            # Обновляем GRU
            x_t = x_t.unsqueeze(1)  # (batch * nodes, 1, hidden_channels)
            _, h = self.gru(x_t, h)  # h: (1, batch * nodes, hidden_channels)
            # Получаем выход
            out = self.fc(h.squeeze(0))  # (batch * nodes, out_channels)
            outputs.append(out)

        # Собираем выходы: (batch, seq_len, nodes, out_channels)
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, nodes * out_channels)
        outputs = outputs.reshape(batch_size, seq_len, num_nodes, -1)  # (batch, seq_len, nodes, out_channels)
        return outputs[:, -self.pre_len:, :, :]  # Возвращаем только предсказания на предсказательный период
    

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TGCNEmb(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, seq_len, pre_len, num_emb_1=288, num_emb_2=7, emb_dim=3, a=1, emb_speed=12):
        super(TGCNEmb, self).__init__()
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.a = a

        # Инициализация эмбеддингов
        self.T_i_D_emb = nn.Parameter(torch.empty(num_emb_1, emb_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(num_emb_2, emb_dim))
        self.reset_parameters()

        # GCN для обработки пространственных зависимостей
        self.gcn = GCNConv(in_channels - 2 + emb_dim + emb_dim + 1, hidden_channels)
        
        # GRU для обработки временных зависимостей
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        
        # FFT-related parameters
        self.fft_len = round(seq_len // 2) + 1
        self.Ex1 = nn.Parameter(torch.randn(self.fft_len, emb_speed), requires_grad=True)
        
        # Полносвязный слой для получения выходных значений
        self.fc = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        # Инициализация эмбеддингов
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def forward(self, x, edge_index):
        # x: (batch, seq_len, nodes, channels)
        batch_size, seq_len, num_nodes, _ = x.shape

        # Восстановление целочисленных значений для T_i_D и D_i_W
        T_i_D_indices = torch.round(x[:, :, :, 1] * 287).type(torch.LongTensor)  # 288 вариантов (0-287)
        D_i_W_indices = torch.round(x[:, :, :, 2] * 6).type(torch.LongTensor)    # 7 вариантов (0-6)

        # Получение эмбеддингов из данных
        T_D = self.T_i_D_emb[T_i_D_indices]  # Индексы для T_i_D
        D_W = self.D_i_W_emb[D_i_W_indices]  # Индексы для D_i_W

        # Извлечение скорости с x[:, :, :, 0]
        speed = x[:, :, :, 0].unsqueeze(-1)  # Данные о скорости

        # FFT преобразование
        xn1 = torch.fft.rfft(speed.permute(0, 3, 2, 1), dim=-1)
        xn1 = torch.abs(xn1)
        xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=1, eps=1e-12, out=None)
        xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=2, eps=1e-12, out=None) * self.a
        xn1 = torch.matmul(xn1, self.Ex1)
        xn1 = xn1.permute(0, 3, 2, 1)

        # Конкатенируем скорость с эмбеддингами и остальными данными
        embeddings = torch.cat([T_D, D_W], dim=-1)  # Конкатенируем эмбеддинги
        x_combined = torch.cat([speed, xn1, embeddings, x[:, :, :, 3:]], dim=-1)  # Конкатенируем с исходными данными

        # Инициализация скрытого состояния GRU
        h = torch.zeros(1, batch_size * num_nodes, self.gru.hidden_size).to(x.device)

        # Обработка временной последовательности
        outputs = []
        for t in range(seq_len):
            # Текущий временной шаг: (batch, nodes, channels)
            x_t = x_combined[:, t, :, :]
            # Применяем GCN: (batch * nodes, hidden_channels)
            x_t = x_t.reshape(batch_size * num_nodes, -1)  # (batch * nodes, channels)
            x_t = self.gcn(x_t, edge_index)  # (batch * nodes, hidden_channels)
            x_t = torch.relu(x_t)
            # Обновляем GRU
            x_t = x_t.unsqueeze(1)  # (batch * nodes, 1, hidden_channels)
            _, h = self.gru(x_t, h)  # h: (1, batch * nodes, hidden_channels)
            # Получаем выход
            out = self.fc(h.squeeze(0))  # (batch * nodes, out_channels)
            outputs.append(out)

        # Собираем выходы: (batch, seq_len, nodes, out_channels)
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, nodes * out_channels)
        outputs = outputs.reshape(batch_size, seq_len, num_nodes, -1)  # (batch, seq_len, nodes, out_channels)
        return outputs[:, -self.pre_len:, :, :]  # Возвращаем только предсказания на предсказательный период


if __name__ == "__main__":

    # Пример использования
    batch_size = 2
    seq_len = 12
    num_nodes = 207
    in_channels = 3
    hidden_channels = 16
    out_channels = 1
    pre_len = 3

    # Модель
    model = TGCN(in_channels, hidden_channels, out_channels, seq_len, pre_len)

    # Пример данных
    TrainX = torch.randn(batch_size, seq_len, num_nodes, in_channels)
    TrainY = torch.randn(batch_size, seq_len, num_nodes)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)  # Пример графа

    # Прогон через модель
    output = model(TrainX, edge_index)
    print(output.shape)  # (batch_size, seq_len, num_nodes, out_channels)   


        # Пример входных данных [batch, seq_len, nodes, channels]
    batch_size = 2
    seq_len = 12
    num_nodes = 207
    num_channels = 8
    input_data = torch.randn(batch_size, seq_len, num_nodes, num_channels)

    # Пример индексов ребер графа
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    # Инициализация модели
    model = TGCNEmb(
        in_channels=num_channels,
        hidden_channels=64,
        out_channels=1,
        seq_len=seq_len,
        pre_len=3,
        num_emb_1=288,
        num_emb_2=7,
        emb_dim=3,
        a=1,
        emb_speed=12
    )

    # Прогон модели через данные
    output = model(input_data, edge_index)

    print(output.shape)  # Проверка формы выходных данных



