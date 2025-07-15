import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from captum.attr import Saliency
from pandas._libs.lib import u8max
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.modules.cnn_2l_rsa_weight import CNN2L
from src.modules.sequence_dataset_rsa import SequenceDataSet

# -------------------------------
# Load Data
# -------------------------------
with open("./data/intermediate_data/pspire_rsa.pkl", "rb") as f:
    df = pickle.load(f)

with open("./data/intermediate_data/pspire_DrLLPS_MLO_rsa.pkl", "rb") as f:
    df_mlo = pickle.load(f)

sum(df_mlo["Type"] == "noID-PSP")

train_df = df.loc[df["Datasets"] == "Training"]
val_df = df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 0)]
val_df = pd.concat([df_mlo, val_df])
val_df = val_df[~val_df["UniprotEntry"].isin(train_df["UniprotEntry"])]
val_df.reset_index(inplace=True)

val_df = pd.concat(
    [val_df.iloc[0:100, :], val_df.iloc[1500:1600, :], val_df.iloc[3000:3010, :]]
)

val_df.__len__()
# -------------------------------
# Model Setup
# -------------------------------
num_categories = df["mapped_seq"].explode().nunique() + 1


class CNN2LWithEmbeddingOut(CNN2L):
    def forward_from_embedding(self, X_embedded, surface_availability):
        surface_availability = surface_availability.unsqueeze(-1)
        X = X_embedded * surface_availability
        X = X.permute(0, 2, 1)

        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool1(X)

        X = self.conv2(X)
        X = self.relu(X)
        X = self.global_pool(X).squeeze(-1)

        X = self.dropout(X)
        X = self.fc1(X)
        return X


# Create model
model = CNN2LWithEmbeddingOut(
    num_categories=num_categories,
    embedding_dim=10,
    conv1_out_channels=70,
    conv2_out_channels=140,
    kernel_size=10,
    num_classes=2,
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
state_dict = torch.load("./data/processed_data/model_run_cnn2l_pspire_rsa_weight_idr")
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -------------------------------
# DataLoader
# -------------------------------
val_data_set = SequenceDataSet(val_df, "mapped_seq", "rsa", "ps_label")
val_loader = DataLoader(val_data_set, batch_size=210, shuffle=False)

# -------------------------------
# Get a Batch
# -------------------------------
vinputs, vrsa, vlabels = next(iter(val_loader))
vinputs = vinputs.to(device)
vrsa = vrsa.to(device)
vlabels = vlabels.to(device)

# Get embedded input and enable gradients
embedded_input = model.embedding(vinputs)
embedded_input.requires_grad_()


# -------------------------------
# Captum Wrapper
# -------------------------------
class SaliencyWrapper(nn.Module):
    def __init__(self, model, rsa):
        super().__init__()
        self.model = model
        self.rsa = rsa

    def forward(self, embedded_input):
        return self.model.forward_from_embedding(embedded_input, self.rsa)


wrapped_model = SaliencyWrapper(model, vrsa)
wrapped_model.to(device)
wrapped_model.eval()

# -------------------------------
# Compute Saliency
# -------------------------------
saliency = Saliency(wrapped_model)
attributions = saliency.attribute(embedded_input, target=1)


with torch.no_grad():
    logits = model.forward_from_embedding(embedded_input, vrsa)
    probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
    class1_probs = probabilities[:, 1].cpu().numpy()
# -------------------------------
# Visualize One Example
# -------------------------------
sal_list = [attributions[x].detach().cpu().numpy() for x in range(210)]
val_df["sal"] = sal_list
val_df["prob"] = class1_probs
val_df.columns

probs = val_df["prob"].to_numpy()[100:200]
top10_idx_idr = np.argsort(probs)[-10:][::-1]
top20_idx = np.append(100 + top10_idx_idr, [27, 63, 37, 51, 77, 29, 53, 11, 8, 73])
for row in top20_idx:
    example_attr = val_df.iat[row, 10]
    seq_len = len(val_df.iat[row, 2])
    protein_name = val_df.iat[row, 1]
    ps_label = val_df.iat[row, 3]
    idr_protein = val_df.iat[row, 4]
    prob = val_df.iat[row, 11]

    if example_attr.ndim == 2:
        example_attr = np.linalg.norm(example_attr, axis=1)
    plt.figure(figsize=(12, 3))
    sns.heatmap(example_attr[None, :], cmap="viridis", cbar=True)
    plt.xlabel("Sequence Position")
    plt.yticks([])
    plt.xticks(
        [x for x in range(0, seq_len, 10)], [str(x) for x in range(0, seq_len, 10)]
    )
    plt.tight_layout()
    plt.xlim(0, seq_len)
    plt.savefig(
        f"./results/plots/captum_idr_{protein_name}_{ps_label}_{prob}_{idr_protein}.png"
    )
