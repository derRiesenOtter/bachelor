import torch
import torch.nn as nn


class CNN2L_BiLSTM(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv1_out_channels,
        conv2_out_channels,
        kernel_size,
        lstm_hidden_size,
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=conv2_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(2 * lstm_hidden_size, num_classes)

    def forward(self, X):
        X = self.embedding(X)  # (batch, seq_len, embedding_dim)
        X = X.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)

        X = self.relu(self.conv1(X))  # (batch, conv1_out_channels, seq_len)
        X = self.pool(X)
        X = self.dropout(X)

        X = self.relu(self.conv2(X))  # (batch, conv2_out_channels, seq_len//2)
        X = self.pool(X)
        X = self.dropout(X)

        X = X.permute(0, 2, 1)  # (batch, seq_len, conv2_out_channels)
        lstm_out, _ = self.lstm(X)  # (batch, seq_len, 2 * hidden_size)

        # Use max over time (temporal max pooling)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, 2*hidden, seq_len)
        pooled = self.global_pool(lstm_out).squeeze(-1)  # (batch, 2*hidden)

        out = self.dropout(pooled)
        out = self.fc(out)  # (batch, num_classes)
        return out
