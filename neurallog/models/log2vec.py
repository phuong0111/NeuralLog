import torch
import torch.nn as nn

class Log2Vec(nn.Module):
    def __init__(self, h=10, no_events=500, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(300, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, no_events)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last sequence output
        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)