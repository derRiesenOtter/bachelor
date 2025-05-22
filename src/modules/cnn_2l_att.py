import torch
import torch.nn as nn


class CNN2L_AttentionPooling(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        conv1_out,
        conv2_out,
        kernel_size,
        attn_dim,
        num_classes,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

        self.conv1 = nn.Conv1d(
            embedding_dim, conv1_out, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            conv1_out, conv2_out, kernel_size, padding=kernel_size // 2
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Attention pooling
        self.attn_fc = nn.Linear(conv2_out, attn_dim)
        self.attn_vector = nn.Linear(attn_dim, 1)

        self.classifier = nn.Linear(conv2_out, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, emb)
        x = x.permute(0, 2, 1)  # (batch, emb, seq_len)

        x = self.relu(self.conv1(x))  # (batch, conv1_out, seq_len)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.conv2(x))  # (batch, conv2_out, seq_len//2)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len//2, conv2_out)

        # Attention pooling
        attn_weights = torch.tanh(self.attn_fc(x))  # (batch, seq_len, attn_dim)
        attn_weights = self.attn_vector(attn_weights)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)

        attended = (x * attn_weights).sum(dim=1)  # (batch, conv2_out)

        out = self.dropout(attended)
        return self.classifier(out)
