import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.modules.sequence_dataset import SequenceDataSet
from src.modules.train_eval import run_train_eval
from src.modules.transformer import LightTransformer

# Opening the data containing the mapped sequences
with open("./data/intermediate_data/pspire.pkl", "rb") as f:
    pspire_df = pickle.load(f)

with open("./data/intermediate_data/ppmclab.pkl", "rb") as f:
    ppmclab_df = pickle.load(f)

val_df = pspire_df.loc[pspire_df["Datasets"] == "Testing"]
pspire_train = pspire_df.loc[pspire_df["Datasets"] == "Training"]
ppmclab_df = ppmclab_df.rename(columns={"UniProt.Acc": "id"})
train_df = pd.concat([ppmclab_df, pspire_train])
train_df = train_df.loc[~train_df["id"].duplicated(keep=False)]
all_df = pd.concat([train_df, val_df])

# filename = Path(__file__).stem
# sys.stdout = open(f"./results/stdout/{filename}.txt", "wt")

# print(f"Length of the ppmclab data set{len(ppmclab_df)}")
# print(f"Length of the training data set {len(train_df)}")

# set a seed for reproducability
torch.manual_seed(13)

# Split data into training data and validation data.
# train_df, val_df = train_test_split(
#     df, test_size=0.2, stratify=df["ps_label"], random_state=13
# )

# Create DataLoaders that are
# responsible for feeding the data into the model
train_data_set = SequenceDataSet(train_df, "mapped_seq", "ps_label")
train_loader = DataLoader(train_data_set, batch_size=8, shuffle=True)
val_data_set = SequenceDataSet(val_df, "mapped_seq", "ps_label")
val_loader = DataLoader(val_data_set, batch_size=8, shuffle=False)

# get the number of categories
num_categories = pspire_df["mapped_seq"].explode().nunique() + 1

# Create the model
model = LightTransformer(
    num_categories=num_categories,
    embedding_dim=12,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    num_classes=2,
)

# Creat a device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# create class weights as the data sets are not uniform and give them to the
# device
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(pspire_df["ps_label"]),
    y=all_df["ps_label"],
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Create a loss function that takes the class weights into consideration and an optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# get the model name and define the epochs
model_name = Path(__file__).stem
epochs = 15
run_train_eval(
    model_name,
    model,
    device,
    epochs,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    val_df,
)
