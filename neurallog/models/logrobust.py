import torch
import torch.nn as nn

class LogRobust(nn.Module):
    def __init__(self, max_len=75, num_hidden=128):
        super().__init__()
        self.bilstm = nn.LSTM(
            300, num_hidden, 
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        self.attention = nn.Sequential(
            nn.Linear(num_hidden * 2, 1),
            nn.Tanh()
        )
        self.fc = nn.Linear(num_hidden * 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, max_len, 300)
        lstm_out, _ = self.bilstm(x)
        # lstm_out shape: (batch_size, max_len, num_hidden * 2)
        
        # Compute attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=1)
        attention_weights = attention_weights.unsqueeze(-1)
        
        # Apply attention
        context = torch.sum(lstm_out * attention_weights, dim=1)
        output = torch.sigmoid(self.fc(context))
        
        return output