import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionBias(nn.Module):
    def __init__(self, max_distance, num_heads):
        super().__init__()
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, seq_len):
        pos = torch.arange(seq_len, device=self.relative_bias.weight.device)
        rel_pos = pos[None, :] - pos[:, None]
        rel_pos = (
            rel_pos.clamp(-self.max_distance, self.max_distance) + self.max_distance
        )
        # Output shape: [L, L, H]
        return self.relative_bias(rel_pos)  # [L, L, H]


class TransformerEncoderLayerRPE(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=256, dropout=0.1, max_distance=32
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.rpe = RelativePositionBias(max_distance, nhead)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: [B, L, D]
        B, L, D = src.shape
        rel_bias = self.rpe(L)  # [L, L, H]
        rel_bias = rel_bias.permute(2, 0, 1)  # [H, L, L]

        attn_output, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
            attn_bias=rel_bias,  # supported in recent PyTorch versions; else patch manually
        )

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


class CNN2LWithRPE(nn.Module):
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
        max_distance=32,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=embedding_dim
        )
        self.embedding_ptm = nn.Embedding(num_embeddings=10, embedding_dim=8)

        self.input_dim = embedding_dim + 8

        # Transformer with RPE
        self.transformer = TransformerEncoderLayerRPE(
            d_model=self.input_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim,
            max_distance=max_distance,
        )

        # CNN Layers
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

        # Transformer with RPE
        X = self.transformer(X)  # [B, L, D]

        # Prepare for Conv1d
        X = X.permute(0, 2, 1)  # [B, D, L]

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
