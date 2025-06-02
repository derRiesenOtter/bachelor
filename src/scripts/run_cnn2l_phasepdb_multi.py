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

from src.modules.cnn_2l import CNN2L
from src.modules.sequence_dataset import SequenceDataSet
from src.modules.train_eval_multi import run_train_eval

# Opening the data containing the mapped sequences
with open("./data/intermediate_data/phasepdb.pkl", "rb") as f:
    df = pickle.load(f)

df = df[df["ps_sp_label"] < 3]

val_df = df.loc[df["type"] == "test"]
train_df = df.loc[df["type"] == "train"]

train_df.ps_sp_label.unique()

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
train_data_set = SequenceDataSet(train_df, "mapped_seq", "ps_sp_label")
train_loader = DataLoader(train_data_set, batch_size=128, shuffle=True)
val_data_set = SequenceDataSet(val_df, "mapped_seq", "ps_sp_label")
val_loader = DataLoader(val_data_set, batch_size=128, shuffle=False)

# get the number of categories
num_categories = df["mapped_seq"].explode().nunique() + 1

# Create the model
model = CNN2L(
    num_categories=num_categories,
    embedding_dim=10,
    conv1_out_channels=70,
    conv2_out_channels=140,
    kernel_size=8,
    num_classes=3,
)

# Creat a device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# create class weights as the data sets are not uniform and give them to the
# device
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["ps_sp_label"]),
    y=df["ps_sp_label"],
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Create a loss function that takes the class weights into consideration and an optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-3, weight_decay=0.5e-5)

# get the model name and define the epochs
model_name = Path(__file__).stem
epochs = 20
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
