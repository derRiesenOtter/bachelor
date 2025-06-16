import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels)
        attn = self.fc1(x)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        return x * attn


class MultiScaleCNNWithAttention(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv_out_channels,
        kernel_sizes,  # e.g., [3, 5, 7]
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Multiscale conv branches with batch normalization
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim + 1,
                        out_channels=conv_out_channels,
                        kernel_size=k,
                        padding=k // 2,
                    ),
                    nn.BatchNorm1d(conv_out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for k in kernel_sizes
            ]
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Attention module on concatenated features
        total_channels = conv_out_channels * len(kernel_sizes)
        self.attention = SEBlock(channel=total_channels)

        self.fc = nn.Linear(total_channels, num_classes)

    def forward(self, X, surface_availability):
        # Embedding and concatenation
        X = self.embedding(X)  # (batch, seq_len, embed_dim)
        surface_availability = surface_availability.unsqueeze(-1)  # (batch, seq_len, 1)
        X = torch.cat(
            [X, surface_availability], dim=-1
        )  # (batch, seq_len, embed_dim+1)

        X = X.permute(0, 2, 1)  # (batch, channels, seq_len)

        # Multiscale convs
        features = [self.global_pool(conv(X)).squeeze(-1) for conv in self.convs]

        # Concatenate all features
        X = torch.cat(features, dim=1)  # (batch, total_channels)

        # Attention
        X = self.attention(X)

        # Final layers
        X = self.dropout(X)
        X = self.fc(X)
        return X
