import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Lambda, ToTensor

with open("./data/intermediate_data/llps_data_ppmclab_bd.pkl", "rb") as f:
    df = pickle.load(f)

num_categories_per_channel = [
    df[mapping].explode().nunique() + 1 for mapping in df.columns if "vec" in mapping
]


class ODCNNDataSet(Dataset):
    def __init__(self, df, label):
        self.df = df
        self.feature_columns = [col for col in self.df.columns if "vec" in col]
        self.label = df[label]
        self.max_len = self.get_max_len()

    def get_max_len(self):
        arrays = [
            torch.tensor(x) for col in df.columns if "vec" in col for x in df[col]
        ]
        max_len = max(arr.shape[0] for arr in arrays)
        return max_len

    def label_transformer(self, label):
        return torch.zeros(len(set(self.label))).scatter_(
            dim=0, index=torch.tensor(label), value=1
        )

    def create_tensor(self, idx):
        channels = [self.df[channel][idx] for channel in self.feature_columns]
        length = len(channels[0])
        padded_channels = [
            torch.tensor(np.append(channel, np.repeat(-2, self.max_len - length)))
            for channel in channels
        ]
        return torch.stack(padded_channels)

    def __len__(self):
        return len(df)

    def __getitem__(self, idx):
        return self.create_tensor(idx), self.label_transformer(int(self.label[idx]))


train_data_set = ODCNNDataSet(df, "PS")
train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        num_channels,
        num_categories_per_channel,
        embedding_dim,
        conv_out_channels,
        kernel_size,
        num_classes,
        dropout=0.3,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories in num_categories_per_channel
            ]
        )
        self.conv = nn.Conv1d(
            in_channels=embedding_dim * num_channels,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
        )
        self.relu = nn.ReLU
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_out_channels, num_classes)

    def forward(self, x):
        batch_size, num_channels, seq_length = x.size()
        embedded_channels = []
        for i in range(num_channels):
            emb = self.embeddings[i](x[:, i])
            embedded_channels.append(emb)
        x_emb = torch.cat(embedded_channels, dim=1)
        x_emb = x_emb.permute(0, 2, 1)
        x = self.conv(x_emb)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


model = NeuralNetwork(
    num_channels=14,
    num_categories_per_channel=num_categories_per_channel,
    embedding_dim=10,
    conv_out_channels=32,
    kernel_size=5,
    num_classes=2,
)

device = torch.device("mps")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}")
