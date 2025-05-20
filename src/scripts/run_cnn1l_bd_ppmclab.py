import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.modules.bd_cnn_1l import BDCNN1L
from src.modules.bd_sequence_dataset import BDSequenceDataSet

# Opening the data containing the block decomposition
with open("./data/intermediate_data/ppmclab_bd.pkl", "rb") as f:
    df = pickle.load(f)

# Split data into training data and validation data.
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["ps_label"], random_state=13
)

# Create DataLoaders that are
# responsible for feeding the data into the model
train_data_set = BDSequenceDataSet(train_df, "ps_label")
train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
val_data_set = BDSequenceDataSet(val_df, "ps_label")
val_loader = DataLoader(val_data_set, batch_size=32, shuffle=False)

# get the number of categories per channel to later create the embeddings
num_categories_per_channel = [
    df[mapping].explode().nunique() + 1 for mapping in df.columns if "_vec" in mapping
]

# Create the model
model = BDCNN1L(
    num_channels=14,
    num_categories_per_channel=num_categories_per_channel,
    embedding_dim=3,
    conv1_out_channels=70,
    kernel_size=10,
    num_classes=2,
)

# Hand the model to the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# create class weights as the data sets are not uniform and give them to the
# device
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(df["ps_label"]), y=df["ps_label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Create a criterion that takes the class weights into consideration and an optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Run the training loop
def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 5
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

    return last_loss


current_file = Path(__file__).stem
epoch_number = 0

EPOCHS = 50

best_vloss = 1000000

train_loss_list = []
val_loss_list = []
all_preds = []
all_probs = []
all_labels = []

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    train_loss_list.append(avg_loss)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            _, preds = torch.max(voutputs, 1)
            probs = torch.sigmoid(voutputs)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(vlabels.cpu().numpy())

    avg_vloss = running_vloss / len(val_loader)
    val_loss_list.append(avg_vloss)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"./data/processed_data/model_{current_file[4:]}"
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

train_loss_list_cpu = [
    loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss
    for loss in train_loss_list
]
val_loss_list_cpu = [
    loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss
    for loss in val_loss_list
]


# create a loss over time plot
plt.scatter(np.arange(1, 1 + len(train_loss_list_cpu)), train_loss_list_cpu, color="r")
plt.scatter(np.arange(1, 1 + len(val_loss_list_cpu)), val_loss_list_cpu, color="b")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(f"./results/plots/{current_file[4:]}_loss")
plt.close()

# create a confusion_matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"./results/plots/{current_file[4:]}_cm")
plt.close()

all_probs = np.array(all_probs)[:, 1]

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = roc_auc_score(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(f"./results/plots/{current_file[4:]}_rocauc")
plt.close()

precision, recall, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = average_precision_score(all_labels, all_probs)

plt.figure()
plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig(f"./results/plots/{current_file[4:]}_prauc")
plt.close()

# train_accuracy = correct / total
# print(
#     f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
# )

# --- Validation ---
# model.eval()
# val_correct = 0
# val_total = 0
# all_preds = []
# all_labels = []
# all_probs = []
#
# with torch.no_grad():
#     for val_inputs, val_labels in val_loader:
#         val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
#         val_outputs = model(val_inputs)
#
#         # Get predicted class and probabilities
#         probs = torch.softmax(val_outputs, dim=1)[:, 1]  # Probability of class 1
#         _, val_predicted = torch.max(val_outputs, 1)
#
#         val_correct += (val_predicted == val_labels).sum().item()
#         val_total += val_labels.size(0)
#
#         all_preds.extend(val_predicted.cpu().numpy())
#         all_labels.extend(val_labels.cpu().numpy())
#         all_probs.extend(probs.cpu().numpy())
#
# val_accuracy = val_correct / val_total
# print(f"Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}")
#
# # --- Confusion Matrix and Classification Report ---
# cm = confusion_matrix(all_labels, all_preds)
# print("Confusion Matrix:\n", cm)
# print(
#     "Classification Report:\n",
#     classification_report(all_labels, all_preds, digits=4),
# )
#
# # --- ROC-AUC ---
# roc_auc = roc_auc_score(all_labels, all_probs)
# print(f"ROC-AUC: {roc_auc:.4f}")
#
# # --- Precision-Recall AUC ---
# precision, recall, _ = precision_recall_curve(all_labels, all_probs)
# pr_auc = auc(recall, precision)
# print(f"PR-AUC: {pr_auc:.4f}")
#
# # --- ROC Curve ---
# fpr, tpr, _ = roc_curve(all_labels, all_probs)
# roc_auc = auc(fpr, tpr)
#
# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
# plt.plot([0, 1], [0, 1], "k--", label="Random")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("results/plots/roc_curve.png")
# plt.close()
#
# # --- Precision-Recall Curve ---
# precision, recall, _ = precision_recall_curve(all_labels, all_probs)
# pr_auc = auc(recall, precision)
#
# plt.figure(figsize=(6, 5))
# plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("results/plots/pr_curve.png")
# plt.close()
#
# print(f"ROC and PR curves saved to: results/plots/")
#
#
#
# class NeuralNetwork(nn.Module):
#     def __init__(
#         self,
#         num_channels,
#         num_categories_per_channel,
#         embedding_dim,
#         conv_out_channels,
#         kernel_size,
#         num_classes,
#         dropout=0.3,
#         nhead=4,
#         num_layers=1,
#     ):
#         super().__init__()
#         # Embedding layers
#         self.embeddings = nn.ModuleList(
#             [
#                 nn.Embedding(num_categories, embedding_dim)
#                 for num_categories in num_categories_per_channel
#             ]
#         )
#
#         # First convolutional block
#         self.conv1 = nn.Conv1d(
#             in_channels=embedding_dim * num_channels,
#             out_channels=conv_out_channels,
#             kernel_size=kernel_size,
#             padding=kernel_size // 2,  # Maintains sequence length
#         )
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#
#         # Second convolutional block
#         self.conv2 = nn.Conv1d(
#             in_channels=conv_out_channels,
#             out_channels=conv_out_channels * 2,  # Double channels
#             kernel_size=kernel_size,
#             padding=kernel_size // 2,
#         )
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#
#         self.relu = nn.ReLU()
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout)
#
#         # Enhanced classifier
#         self.fc1 = nn.Linear(conv_out_channels * 2, 128)  # Additional hidden layer
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         batch_size, num_channels, seq_length = x.size()
#
#         # Embedding
#         embedded_channels = []
#         for i in range(num_channels):
#             emb = self.embeddings[i](x[:, i])
#             embedded_channels.append(emb)
#         x_emb = torch.cat(embedded_channels, dim=-1)
#         x_emb = x_emb.permute(0, 2, 1)
#
#         # First conv block
#         x = self.conv1(x_emb)
#         x = self.relu(x)
#         x = self.pool1(x)
#
#         # Second conv block
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool2(x)
#
#         # Global pooling and classification
#         x = self.global_pool(x)
#         x = x.squeeze(-1)
#         x = self.dropout(x)
#
#         # Enhanced classifier
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#
