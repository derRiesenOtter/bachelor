import torch
import torch.nn as nn


class LightTransformer(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        num_heads,
        ff_dim,
        num_layers,
        num_classes,
        max_len=2700,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)
