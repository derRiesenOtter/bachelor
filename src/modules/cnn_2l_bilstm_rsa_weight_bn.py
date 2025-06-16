import torch
import torch.nn as nn


class CNN2L(nn.Module):
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

        self.bn1 = nn.BatchNorm1d(conv1_out_channels)
        self.bn2 = nn.BatchNorm1d(conv2_out_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)

        # BLSTM Layer
        self.blstm = nn.LSTM(
            input_size=conv2_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Global pooling and classification
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(2 * lstm_hidden_size, num_classes)

    def forward(self, X, surface_availability):
        # Embedding + masking
        X = self.embedding(X)
        surface_availability = surface_availability.unsqueeze(-1)
        X = X * surface_availability

        # CNN layers
        X = X.permute(0, 2, 1)  # (B, embedding_dim, seq_len)
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)

        # Prepare for LSTM
        X = X.permute(0, 2, 1)  # (B, seq_len, conv2_out_channels)

        # BLSTM
        X, _ = self.blstm(X)  # Output shape: (B, seq_len, 2 * hidden_size)

        # Global pooling over time
        X = X.permute(0, 2, 1)  # (B, 2 * hidden_size, seq_len)
        X = self.global_pool(X).squeeze(-1)  # (B, 2 * hidden_size)

        # Dropout + FC
        X = self.dropout(X)
        X = self.fc1(X)
        return X
