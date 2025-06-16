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

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(conv3_out_channels, num_classes)

    def forward(self, X, surface_availability):
        X = self.embedding(X)

        surface_availability = surface_availability.unsqueeze(-1)
        X = torch.cat([X, surface_availability], dim=-1)

        X = X.permute(0, 2, 1)

        X = self.relu(self.conv1(X))
        X = self.pool1(X)
        X = self.dropout(X)

        X = self.relu(self.conv2(X))
        X = self.pool2(X)

        X = self.relu(self.conv3(X))
        X = self.global_pool(X).squeeze(-1)

        X = self.dropout(X)
        X = self.fc1(X)
        return X
