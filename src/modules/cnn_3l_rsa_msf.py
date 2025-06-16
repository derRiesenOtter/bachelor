import torch
import torch.nn as nn


class CNN3L(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv_out_channels,
        kernel_sizes,  # List of different kernel sizes, e.g., [3, 5, 7]
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Create multiple conv layers for multiscale feature extraction
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim + 1,
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

        # Adaptive pooling after all convolutions
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layer after concatenation
        self.fc = nn.Linear(conv_out_channels * len(kernel_sizes), num_classes)

    def forward(self, X, surface_availability):
        X = self.embedding(X)  # (batch, seq_len, embed_dim)
        surface_availability = surface_availability.unsqueeze(-1)  # (batch, seq_len, 1)
        X = torch.cat(
            [X, surface_availability], dim=-1
        )  # (batch, seq_len, embed_dim+1)

        X = X.permute(0, 2, 1)  # (batch, channels, seq_len)

        # Apply all convolutional layers
        features = [self.global_pool(conv(X)).squeeze(-1) for conv in self.convs]
        X = torch.cat(features, dim=1)  # (batch, conv_out_channels * len(kernel_sizes))

        X = self.dropout(X)
        X = self.fc(X)
        return X
