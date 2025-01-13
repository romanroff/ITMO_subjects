import torch
import torch.nn as nn

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.2, pre_len=3):
        super(TCNModel, self).__init__()
        self.pre_len = pre_len
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(num_channels, output_size)

    def forward(self, x):
        # x shape: [batch, seq_len, nodes, channels]
        batch, seq_len, nodes, channels = x.size()
        x = x.permute(0, 2, 3, 1)  # (batch, seq_len, nodes, channels) -> (batch, nodes, channels, seq_len)
        x = x.reshape(batch * nodes, channels, seq_len)  # Merge batch and nodes dimensions

        out = self.tcn(x)  # shape: [batch * nodes, num_channels, seq_len + kernel_size - 1]
        out = out[:, :, -self.pre_len:]  # Use last `pre_len` time steps, shape: [batch * nodes, num_channels, pre_len]
        
        out = out.permute(0, 2, 1)  # shape: [batch * nodes, pre_len, num_channels]
        out = self.fc(out)  # shape: [batch * nodes, pre_len, output_size]
        out = out.view(batch, nodes, self.pre_len, -1).permute(0, 2, 1, 3)  # [batch, pre_len, nodes, output_size]
        return out  # Final shape: [batch, pre_len, nodes, 1]
    



class TCNModelEmb(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.2, seq_len=12, pre_len=3, num_emb=[288, 7], emb_dim=3, a=1, emb_speed=12):
        super(TCNModelEmb, self).__init__()
        self.pre_len = pre_len
        self.a = a
        
        # Инициализация эмбеддингов
        self.T_i_D_emb = nn.Parameter(torch.empty(num_emb[0], emb_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(num_emb[1], emb_dim))
        
        # Инициализация эмбеддингов
        self.reset_parameters()

        self.tcn = nn.Sequential(
            nn.Conv1d(input_size-2+emb_dim+emb_dim+1, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fft_len = round(seq_len//2) + 1
        self.Ex1 = nn.Parameter(torch.randn(self.fft_len, emb_speed), requires_grad=True)
        self.fc = nn.Linear(num_channels, output_size)

    def reset_parameters(self):
        # Инициализация эмбеддингов
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def forward(self, x):
        batch, seq_len, nodes, channels = x.size()

        # Восстановление целочисленных значений для T_i_D и D_i_W
        T_i_D_indices = torch.round(x[:, :, :, 1] * 287).type(torch.LongTensor)  # 288 вариантов (0-287)
        D_i_W_indices = torch.round(x[:, :, :, 2] * 6).type(torch.LongTensor)    # 7 вариантов (0-6)

        # Получение эмбеддингов из данных
        T_D = self.T_i_D_emb[T_i_D_indices]  # Индексы для T_i_D
        D_W = self.D_i_W_emb[D_i_W_indices]  # Индексы для D_i_W


        # Извлечение скорости с x[:, :, :, 0]
        speed = x[:, :, :, 0].unsqueeze(-1)  # Данные о скорости

        xn1 = torch.fft.rfft(speed.permute(0, 3, 2, 1), dim=-1)
        xn1 = torch.abs(xn1)
        

        xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=1, eps=1e-12, out=None)
        xn1 = torch.nn.functional.normalize(xn1, p=2.0, dim=2, eps=1e-12, out=None) * self.a

        xn1 = torch.matmul(xn1, self.Ex1)
        xn1 = xn1.permute(0, 3, 2, 1)
        
        # Конкатенируем скорость с эмбеддингами и остальными данными
        embeddings = torch.cat([T_D, D_W], dim=-1)  # Конкатенируем эмбеддинги
        x_combined = torch.cat([xn1, embeddings, x[:, :, :, 3:]], dim=-1)  # Конкатенируем с исходными данными (скорость и другие данные)

        # Преобразование данных перед подачей в tcn
        x_combined = x_combined.permute(0, 2, 3, 1)  # (batch, seq_len, nodes, channels) -> (batch, nodes, channels, seq_len)
        x_combined = x_combined.reshape(batch * nodes, -1, seq_len)  # Merge batch and nodes dimensions

        out = self.tcn(x_combined)  # shape: [batch * nodes, num_channels, seq_len + kernel_size - 1]
        out = out[:, :, -self.pre_len:]  # Use last `pre_len` time steps, shape: [batch * nodes, num_channels, pre_len]
        
        out = out.permute(0, 2, 1)  # shape: [batch * nodes, pre_len, num_channels]
        out = self.fc(out)  # shape: [batch * nodes, pre_len, output_size]
        out = out.view(batch, nodes, self.pre_len, -1).permute(0, 2, 1, 3)  # [batch, pre_len, nodes, output_size]

        return out  # Final shape: [batch, pre_len, nodes, output_size]

if __name__ == "__main__":
    # Пример входных данных [batch, seq_len, nodes, channels]
    batch_size = 2
    seq_len = 12
    emb_dim = 3
    num_nodes = 207
    num_channels = 8
    input_data = torch.randn(batch_size, seq_len, num_nodes, num_channels)

    # Инициализация модели
    model = TCNModelEmb(input_size=num_channels, output_size=1, num_channels=32, kernel_size=3, seq_len=seq_len, pre_len=3, emb_speed=seq_len, emb_dim=10)

    # Прогон модели через данные
    output = model(input_data)

    print("Output shape:", output.shape)  # Проверка формы выходных данных
