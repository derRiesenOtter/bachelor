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
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

        # Additional linear layer to process surface availability values
        self.surface_linear = nn.Linear(1, embedding_dim)

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim
            * 2,  # Double channels to accommodate both embeddings
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_size,
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(conv2_out_channels, num_classes)

    def forward(self, X, surface_availability):
        # Process categorical input
        X_cat = self.embedding(X)

        # Process surface availability input
        # Add dimension to surface_availability to make it [batch_size, seq_length, 1]
        surface_availability = surface_availability.unsqueeze(-1)
        X_surf = self.surface_linear(surface_availability)

        # Combine both embeddings
        X_combined = torch.cat([X_cat, X_surf], dim=-1)

        # shape: (batch_size, seq_length, embedding_dim * 2)
        X_combined = X_combined.permute(0, 2, 1)
        # shape: (batch_size, embedding_dim * 2, seq_length)

        X = self.conv1(X_combined)
        X = self.relu(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.global_pool(X).squeeze(-1)
        X = self.dropout(X)
        X = self.fc1(X)
        return X
