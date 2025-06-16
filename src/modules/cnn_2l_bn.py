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
        )

        self.bn1 = nn.BatchNorm1d(conv1_out_channels)
        self.bn2 = nn.BatchNorm1d(conv2_out_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(conv2_out_channels, num_classes)

    def forward(self, X):
        X = self.embedding(X)
        X = X.permute(0, 2, 1)

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.global_pool(X).squeeze(-1)

        X = self.dropout(X)
        X = self.fc1(X)

        return X
