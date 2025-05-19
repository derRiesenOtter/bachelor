import torch
import torch.nn as nn


class BDCNN1L(nn.Module):
    def __init__(
        self,
        num_channels,
        num_categories_per_channel,
        embedding_dim,
        conv1_out_channels,
        kernel_size,
        num_classes,
    ):
        super().__init__()
        self.num_channels = num_channels

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=num_categories,
                    embedding_dim=(num_categories + 1) // 2,
                )
                for num_categories in num_categories_per_channel
            ]
        )

        self.conv1 = nn.Conv1d(
            in_channels=num_channels * embedding_dim,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(conv1_out_channels, num_classes)

    def forward(self, X):
        embedded_channels = []
        for i in range(self.num_channels):
            embedding = self.embeddings[i](X[:, i])
            embedded_channels.append(embedding)
        X = torch.cat(embedded_channels, dim=-1)
        X = X.permute(0, 2, 1)

        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool1(X)

        X = self.fc1(X)
        return X
