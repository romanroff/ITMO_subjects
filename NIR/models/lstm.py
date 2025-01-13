import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, pre_len=3):
        super(LSTMModel, self).__init__()
        self.pre_len = pre_len
        self.dropout = nn.Dropout(0.5)  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch, seq_len, nodes, channels]
        batch, seq_len, nodes, channels = x.size()
        x = x.view(batch * nodes, seq_len, channels)  # Merge batch and nodes dimensions for LSTM
        
        h_lstm, _ = self.lstm(x)  # h_lstm shape: [batch * nodes, seq_len, hidden_size]
        h_lstm = self.dropout(h_lstm)  # Apply dropout
        h_lstm = h_lstm[:, -self.pre_len:, :]  # Take last `pre_len` steps
        
        out = self.fc(h_lstm)  # shape: [batch * nodes, pre_len, output_size]
        out = out.view(batch, nodes, self.pre_len, -1).permute(0, 2, 1, 3)  # [batch, pre_len, nodes, output_size]
        return out  # Final shape: [batch, pre_len, nodes, 1]