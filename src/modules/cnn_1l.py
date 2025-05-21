import torch.nn as nn


class CNN1L(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv1_out_channels,
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
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(conv1_out_channels, num_classes)

    def forward(self, X):
        embedded_channels = []
        X = self.embedding(X[:,])
        # shape: (batch_size, seq_length, num_channels * embedding_dim)
        X = X.permute(0, 2, 1)
        # shape: (batch_size, num_channels * embedding_dim, seq_length)
        X = self.conv1(X)
        # shape: (batch_size, conv1_out_channels, new_seq_length)
        X = self.relu(X)
        # shape: (batch_size, conv1_out_channels, new_seq_length)
        X = self.pool1(X)
        # shape: (batch_size, conv1_out_channels, pooled_seq_length)

        X = self.global_pool(X).squeeze(-1)
        # shape: (batch_size, conv1_out_channels)
        X = self.dropout(X)
        X = self.fc1(X)
        # shape: (batch_size, num_classes)
        return X
