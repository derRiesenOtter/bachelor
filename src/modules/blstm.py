import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_categories,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)
