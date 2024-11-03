import torch
import torch.nn as nn
from .positional_encodings import PositionalEncoding

# embed_dim = 768  # Embedding size for each token
# num_heads = 12  # Number of attention heads
# ff_dim = 2048  # Hidden layer size in feed forward network inside transformer
# max_len = 20

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # PyTorch attention expects seq_len, batch, embed_dim
        x_t = x.transpose(0, 1)
        attn_output, _ = self.attention(x_t, x_t, x_t)
        attn_output = attn_output.transpose(0, 1)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(out1)
        return self.layernorm2(out1 + self.dropout2(ff_output))

class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, ff_dim, max_len, num_heads, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(max_len, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer(x)
        # Global pooling
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.softmax(self.fc2(x), dim=-1)