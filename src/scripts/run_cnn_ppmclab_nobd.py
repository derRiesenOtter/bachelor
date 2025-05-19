import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

with open("./data/intermediate_data/ppmclab.pkl", "rb") as f:
    df = pickle.load(f)


class ODCNNDataSet(Dataset):
    def __init__(self, df, label_col):
        self.df = df.reset_index(drop=True)
        self.feature_column = "mapped_seq"
        self.label = self.df[label_col].astype(int)
        self.max_len = self.get_max_len()

    def get_max_len(self) -> int:
        """
        Returns the max length of all sequences. This length will be used for
        padding.

        :return: Max Sequence Length.
        :rtype: int
        """
        arrays = [torch.tensor(x) for x in df[self.feature_column]]
        return max(arr.shape[0] for arr in arrays)

    def create_tensor(self, idx):
        channel = self.df[self.feature_column][idx]
        length = len(channel)
        padded_channels = [
            torch.tensor(np.append(channel, np.repeat(-1, self.max_len - length)))
        ]
        return torch.stack(padded_channels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.create_tensor(idx), (self.label[idx])


# Split data into training data and validation data. Create DataLoaders that are
# responsible to feed the data into the model


train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["PS"], random_state=42
)

train_data_set = ODCNNDataSet(train_df, "PS")
train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
val_data_set = ODCNNDataSet(val_df, "PS")
val_loader = DataLoader(val_data_set, batch_size=32, shuffle=False)

# Create a model class


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
        lstm_hidden_size=64,
        lstm_layers=1,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories in num_categories_per_channel
            ]
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim * num_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # CNN blocks (input channels = 2 * lstm_hidden_size due to bidirectional LSTM)
        self.conv1 = nn.Conv1d(
            in_channels=2 * lstm_hidden_size,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=conv_out_channels,
            out_channels=conv_out_channels * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(conv_out_channels * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, num_channels, seq_length = x.size()

        # Embedding
        embedded_channels = []
        for i in range(num_channels):
            emb = self.embeddings[i](x[:, i])  # (batch, seq_len, emb_dim)
            embedded_channels.append(emb)

        x_emb = torch.cat(embedded_channels, dim=-1)  # (batch, seq_len, total_emb_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(x_emb)  # (batch, seq_len, 2 * hidden_size)

        # CNN expects (batch, channels, seq_len)
        x = lstm_out.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Create the model
num_categories_per_channel = [
    df[mapping].explode().nunique() + 1
    for mapping in df.columns
    if "mapped_seq" in mapping
]

model = NeuralNetwork(
    num_channels=len(num_categories_per_channel),
    num_categories_per_channel=num_categories_per_channel,
    embedding_dim=40,
    conv_out_channels=32,
    kernel_size=5,
    num_classes=2,
)

# Hand the model to the device

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(df["PS"]), y=df["PS"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    # --- Training ---
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

    train_accuracy = correct / total
    print(
        f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
    )

    # --- Validation ---
    model.eval()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)

            # Get predicted class and probabilities
            probs = torch.softmax(val_outputs, dim=1)[:, 1]  # Probability of class 1
            _, val_predicted = torch.max(val_outputs, 1)

            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0)

            all_preds.extend(val_predicted.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_accuracy = val_correct / val_total
    print(f"Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}")

    # --- Confusion Matrix and Classification Report ---
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print(
        "Classification Report:\n",
        classification_report(all_labels, all_preds, digits=4),
    )

    # --- ROC-AUC ---
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # --- Precision-Recall AUC ---
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/roc_curve.png")
    plt.close()

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/pr_curve.png")
    plt.close()

    print(f"ROC and PR curves saved to: results/plots/")
