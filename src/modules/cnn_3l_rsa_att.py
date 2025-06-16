import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3L(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv1_out_channels,
        conv2_out_channels,
        conv3_out_channels,
        kernel_size,
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim + 1,
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
        self.conv3 = nn.Conv1d(
            in_channels=conv2_out_channels,
            out_channels=conv3_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(conv3_out_channels, conv3_out_channels // 2),
            nn.ReLU(),
            nn.Linear(conv3_out_channels // 2, 1),
            nn.Softmax(dim=1),
        )

        # Pooling and dropout
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

        # Final layers
        self.fc1 = nn.Linear(conv3_out_channels, num_classes)

    def forward(self, X, surface_availability):
        # Embedding and feature concatenation
        X = self.embedding(X)  # (batch_size, seq_len, embedding_dim)
        surface_availability = surface_availability.unsqueeze(-1)
        X = torch.cat([X, surface_availability], dim=-1)
        X = X.permute(0, 2, 1)  # (batch_size, embedding_dim+1, seq_len)

        # First convolutional block
        X = self.relu(self.conv1(X))
        X = self.pool1(X)
        X = self.dropout(X)

        # Second convolutional block
        X = self.relu(self.conv2(X))
        X = self.pool2(X)

        # Third convolutional block
        X = self.relu(self.conv3(X))  # (batch_size, conv3_out_channels, seq_len)

        # Attention mechanism
        X_permuted = X.permute(0, 2, 1)  # (batch_size, seq_len, conv3_out_channels)
        attn_weights = self.attention(X_permuted)  # (batch_size, seq_len, 1)
        attn_weights = attn_weights.permute(0, 2, 1)  # (batch_size, 1, seq_len)

        # Apply attention weights
        X = torch.bmm(attn_weights, X_permuted).squeeze(
            1
        )  # (batch_size, conv3_out_channels)

        # Final classification
        X = self.dropout(X)
        X = self.fc1(X)
        return X
