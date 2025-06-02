import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.modules.cnn_2l import CNN2L
from src.modules.sequence_dataset import SequenceDataSet

with open("./data/intermediate_data/phasepdb.pkl", "rb") as f:
    df = pickle.load(f)

with open("./data/intermediate_data/PhaSepDB high-throughput.pkl", "rb") as f:
    df_mlo = pickle.load(f)

train_df = df.loc[df["type"] == "train"]
val_df = df.loc[(df["type"] == "test") & (df["ps_label"] == 0)]
val_df = pd.concat([df_mlo, val_df])

num_categories = df["mapped_seq"].explode().nunique() + 1

model = CNN2L(
    num_categories=num_categories,
    embedding_dim=10,
    conv1_out_channels=70,
    conv2_out_channels=140,
    kernel_size=10,
    num_classes=2,
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
state_dict = torch.load("./data/processed_data/model_run_cnn2l_phasepdb")
model.load_state_dict(state_dict)
model.to(device)
model.eval()

model_name = "eval_phasepdb_mlo_psdbht"
val_data_set = SequenceDataSet(val_df, "mapped_seq", "ps_label")
val_loader = DataLoader(val_data_set, batch_size=128, shuffle=False)

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(df["ps_label"]), y=df["ps_label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

running_vloss = 0.0
all_preds_tmp = []
all_probs_tmp = []
all_labels_tmp = []

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
        all_preds_tmp.extend(preds.cpu().numpy())
        all_probs_tmp.extend(probs.cpu().numpy())
        all_labels_tmp.extend(vlabels.cpu().numpy())

avg_vloss = running_vloss / len(val_loader)

# Track best performance, and save the model's state
all_preds = all_preds_tmp
all_probs = all_probs_tmp
all_labels = all_labels_tmp

# create a confusion_matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"./results/plots/{model_name}_cm")
plt.close()

all_probs = np.array(all_probs)[:, 1]

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = roc_auc_score(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.ylim(0, 1)
plt.legend()
plt.savefig(f"./results/plots/{model_name}_rocauc")
plt.close()

precision, recall, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = average_precision_score(all_labels, all_probs)

plt.figure()
plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.legend()
plt.savefig(f"./results/plots/{model_name}_prauc")
plt.close()
