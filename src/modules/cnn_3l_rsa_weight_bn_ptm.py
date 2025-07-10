import torch
import torch.nn as nn


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

        self.embedding_ptm = nn.Embedding(num_embeddings=10, embedding_dim=8)

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim + 8,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_size,
        )

        self.conv3 = nn.Conv1d(
            in_channels=conv2_out_channels,
            out_channels=conv3_out_channels,
            kernel_size=kernel_size,
        )
        self.bn1 = nn.BatchNorm1d(conv1_out_channels)
        self.bn2 = nn.BatchNorm1d(conv2_out_channels)
        self.bn3 = nn.BatchNorm1d(conv3_out_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(conv3_out_channels, num_classes)

    def forward(self, X, surface_availability, ptm):
        X = self.embedding(X)
        ptm_embedded = self.embedding_ptm(ptm)
        surface_availability = surface_availability.unsqueeze(-1)
        X = X * surface_availability
        X = torch.cat([X, ptm_embedded], dim=-1)
        X = X.permute(0, 2, 1)

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.pool1(X)

        X = self.conv3(X)
        X = self.bn3(X)
        X = self.relu(X)
        X = self.global_pool(X).squeeze(-1)

        X = self.dropout(X)
        X = self.fc1(X)
        return X
