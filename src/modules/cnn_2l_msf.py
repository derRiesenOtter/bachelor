import torch
import torch.nn as nn


class CNN2L(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv_out_channels,
        kernel_sizes,
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=conv_out_channels,
                        kernel_size=k,
                        padding=k // 2,
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for k in kernel_sizes
            ]
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(conv_out_channels * len(kernel_sizes), num_classes)

    def forward(self, X):
        X = self.embedding(X)

        X = X.permute(0, 2, 1)
        features = [self.global_pool(conv(X)).squeeze(-1) for conv in self.convs]
        X = torch.cat(features, dim=1)

        X = self.dropout(X)
        X = self.fc1(X)
        return X
