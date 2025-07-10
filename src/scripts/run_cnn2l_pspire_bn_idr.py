import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.modules.cnn_2l_bn import CNN2L
from src.modules.sequence_dataset import SequenceDataSet
from src.modules.train_eval import run_train_eval

# Opening the data containing the mapped sequences
with open("./data/intermediate_data/pspire.pkl", "rb") as f:
    df = pickle.load(f)


df = df.loc[(df["idr_protein"] == 1) | (df["ps_label"] == 0)]
train_df = df.loc[df["Datasets"] == "Training"]
val_df = df.loc[df["Datasets"] == "Testing"]

# set a seed for reproducability
torch.manual_seed(13)

# Split data into training data and validation data.
# train_df, val_df = train_test_split(
#     df, test_size=0.2, stratify=df["ps_label"], random_state=13
# )

# Create DataLoaders that are
# responsible for feeding the data into the model
train_data_set = SequenceDataSet(train_df, "mapped_seq", "ps_label")
train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
val_data_set = SequenceDataSet(val_df, "mapped_seq", "ps_label")
val_loader = DataLoader(val_data_set, batch_size=32, shuffle=False)

# get the number of categories
num_categories = df["mapped_seq"].explode().nunique() + 1

# Create the model
model = CNN2L(
    num_categories=num_categories,
    embedding_dim=10,
    conv1_out_channels=70,
    conv2_out_channels=140,
    kernel_size=10,
    num_classes=2,
)

# Creat a device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# create class weights as the data sets are not uniform and give them to the
# device
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(df["ps_label"]), y=df["ps_label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Create a loss function that takes the class weights into consideration and an optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

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
    patience=10,
)
