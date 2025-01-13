import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, pre_len=3, dropout=0.1, use_norm=True):
        super(GRUModel, self).__init__()
        self.pre_len = pre_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_norm = use_norm
        # GRU layer
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        # Normalization layer (optional)
        if self.use_norm:
            self.norm = nn.LayerNorm(hidden_size)
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, seq_len, nodes, channels]
        batch, seq_len, nodes, channels = x.size()
        x = x.view(batch * nodes, seq_len, channels)
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # GRU forward pass
        h_gru, _ = self.gru(x, h0)
        # Apply normalization if enabled
        if self.use_norm:
            h_gru = self.norm(h_gru)
        # Select the last `pre_len` time steps
        h_gru = h_gru[:, -self.pre_len:, :]
        # Apply dropout
        h_gru = self.dropout(h_gru)
        # Fully connected layer
        out = self.fc(h_gru)
        # Reshape output to [batch, pre_len, nodes, output_size]
        out = out.view(batch, nodes, self.pre_len, -1).permute(0, 2, 1, 3)
        return out
    

class GRUModelEmb(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, seq_len=12, pre_len=3, num_emb=[288, 7], emb_dim=3, a=1, emb_speed=12):
        super(GRUModelEmb, self).__init__()
        self.pre_len = pre_len
        self.a = a
        
        # Инициализация эмбеддингов
        self.T_i_D_emb = nn.Parameter(torch.empty(num_emb[0], emb_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(num_emb[1], emb_dim))
        
        # Инициализация эмбеддингов
        self.reset_parameters()

        # GRU слой
        self.gru = nn.GRU(input_size - 2 + emb_dim + emb_dim + 1, hidden_size, num_layers, batch_first=True)
        
        # FFT-related parameters
        self.fft_len = round(seq_len // 2) + 1
        self.Ex1 = nn.Parameter(torch.randn(self.fft_len, emb_speed), requires_grad=True)
        
        # Fully connected слой
        self.fc = nn.Linear(hidden_size, output_size)

    def reset_parameters(self):
        # Инициализация эмбеддингов
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def forward(self, x):
        batch, seq_len, nodes, channels = x.size()

        # Восстановление целочисленных значений для T_i_D и D_i_W
        T_i_D_indices = torch.round(x[:, :, :, 1] * 287).type(torch.LongTensor)      # 24 варианта (0-23)
        D_i_W_indices = torch.round(x[:, :, :, 2] * 6).type(torch.LongTensor)       # 7 вариантов (0-6)

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
        x_combined = torch.cat([speed, xn1, embeddings, x[:, :, :, 3:]], dim=-1)  # Конкатенируем с исходными данными (скорость и другие данные)

        # Преобразование данных перед подачей в GRU
        x_combined = x_combined.view(batch * nodes, seq_len, -1)  # Merge batch and nodes dimensions

        # GRU
        h_gru, _ = self.gru(x_combined)  # h_gru shape: [batch * nodes, seq_len, hidden_size]
        h_gru = h_gru[:, -self.pre_len:, :]  # Take last `pre_len` steps

        # Fully connected слой
        out = self.fc(h_gru)  # shape: [batch * nodes, pre_len, output_size]
        out = out.view(batch, nodes, self.pre_len, -1).permute(0, 2, 1, 3)  # [batch, pre_len, nodes, output_size]

        return out  # Final shape: [batch, pre_len, nodes, output_size]
    

if __name__ == "__main__":
    # Пример входных данных [batch, seq_len, nodes, channels]
    batch_size = 2
    seq_len = 12
    num_nodes = 207
    num_channels = 8
    input_data = torch.randn(batch_size, seq_len, num_nodes, num_channels)

    # Инициализация модели
    model = GRUModelEmb(
        input_size=num_channels,
        hidden_size=64,
        output_size=1,
        num_layers=2,
        seq_len=seq_len,
        pre_len=3,
        num_emb_1=288,
        num_emb_2=7,
        emb_dim=3,
        a=1,
        emb_speed=12
    )

    # Прогон модели через данные
    output = model(input_data)

    print("Output shape:", output.shape)  # Проверка формы выходных данных