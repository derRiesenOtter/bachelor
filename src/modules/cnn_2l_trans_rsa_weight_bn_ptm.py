import torch
import torch.nn as nn


class CNN2LWithTransformer(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv1_out_channels,
        conv2_out_channels,
        kernel_size,
        num_classes,
        transformer_dim=64,
        num_heads=4,
        num_transformer_layers=1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )
        self.embedding_ptm = nn.Embedding(num_embeddings=10, embedding_dim=8)

        self.input_dim = embedding_dim + 8

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        self.conv1 = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(conv1_out_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_size,
        )
        self.bn2 = nn.BatchNorm1d(conv2_out_channels)

        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(conv2_out_channels, num_classes)

    def forward(self, X, surface_availability, ptm):
        # Embeddings
        X = self.embedding(X)
        ptm_embedded = self.embedding_ptm(ptm)
        surface_availability = surface_availability.unsqueeze(-1)
        X = X * surface_availability
        X = torch.cat([X, ptm_embedded], dim=-1)  # [B, L, D]

        # Transformer expects [B, L, D] if batch_first=True
        X = self.transformer(X)  # Apply Transformer Encoder

        # Prepare for Conv1d: [B, D, L]
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
