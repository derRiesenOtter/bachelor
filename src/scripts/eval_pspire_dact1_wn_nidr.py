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

from src.modules.cnn_2l_rsa_weight_bn import CNN2L
from src.modules.sequence_dataset_rsa import SequenceDataSet

with open("./data/intermediate_data/pspire_rsa.pkl", "rb") as f:
    df = pickle.load(f)

with open(
    "./data/intermediate_data/pspire_DACT1-particulate proteome_rsa.pkl", "rb"
) as f:
    df_mlo = pickle.load(f)

# df_mlo = df_mlo.loc[df_mlo["Type"] == "noID-PSP"]

model_name = "eval_pspire_mlo_rsa_bn_dact1_nidr"
# with open("./data/intermediate_data/ppmclab.pkl", "rb") as f:
#     ppmclab_df = pickle.load(f)

train_df = df.loc[df["Datasets"] == "Training"]
val_df = df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 0)]
val_df = pd.concat([df_mlo, val_df])

val_df.__len__()

# ppmclab_df = ppmclab_df.rename(columns={"UniProt.Acc": "id"})
# ppmclab_df = ppmclab_df.loc[~ppmclab_df["id"].isin(val_df["id"])]
# train_df = train_df.loc[~train_df["id"].duplicated(keep=False)]
val_df = val_df[~val_df["UniprotEntry"].isin(train_df["UniprotEntry"])]
val_df.__len__()

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
state_dict = torch.load(
    "./data/processed_data/model_run_cnn2l_pspire_rsa_weight_bn_nidr"
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

val_data_set = SequenceDataSet(val_df, "mapped_seq", "rsa", "ps_label")
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
        vinputs, vrsa, vlabels = vdata
        vinputs, vrsa, vlabels = vinputs.to(device), vrsa.to(device), vlabels.to(device)
        voutputs = model(vinputs, vrsa)
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


idr_probs = np.array(all_probs)[
    (val_df["Type"] == "ID-PSP") | (val_df["ps_label"] == 0)
]
idr_labels = np.array(all_labels)[
    (val_df["Type"] == "ID-PSP") | (val_df["ps_label"] == 0)
]
idr_preds = np.array(all_preds)[
    (val_df["Type"] == "ID-PSP") | (val_df["ps_label"] == 0)
]

nidr_probs = np.array(all_probs)[
    (val_df["Type"] == "noID-PSP") | (val_df["ps_label"] == 0)
]
nidr_labels = np.array(all_labels)[
    (val_df["Type"] == "noID-PSP") | (val_df["ps_label"] == 0)
]
nidr_preds = np.array(all_preds)[
    (val_df["Type"] == "noID-PSP") | (val_df["ps_label"] == 0)
]

# create a confusion_matrix
cm = confusion_matrix(idr_labels, idr_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"./results/plots/{model_name}_cm_idr")
plt.close()

fpr, tpr, _ = roc_curve(idr_labels, idr_probs)
roc_auc = roc_auc_score(idr_labels, idr_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.ylim(0, 1)
plt.legend()
plt.savefig(f"./results/plots/{model_name}_rocauc_idr")
plt.close()

precision, recall, _ = precision_recall_curve(idr_labels, idr_probs)
pr_auc = average_precision_score(idr_labels, idr_probs)

plt.figure()
plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.legend()
plt.savefig(f"./results/plots/{model_name}_prauc_idr")
plt.close()

# create a confusion_matrix
cm = confusion_matrix(nidr_labels, nidr_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"./results/plots/{model_name}_cm_nidr")
plt.close()

fpr, tpr, _ = roc_curve(nidr_labels, nidr_probs)
roc_auc = roc_auc_score(nidr_labels, nidr_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.ylim(0, 1)
plt.legend()
plt.savefig(f"./results/plots/{model_name}_rocauc_nidr")
plt.close()

precision, recall, _ = precision_recall_curve(nidr_labels, nidr_probs)
pr_auc = average_precision_score(nidr_labels, nidr_probs)

plt.figure()
plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.legend()
plt.savefig(f"./results/plots/{model_name}_prauc_nidr")
plt.close()
