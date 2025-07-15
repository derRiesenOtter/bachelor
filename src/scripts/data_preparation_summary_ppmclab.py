import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from src.modules.get_ptm import add_ptm

df = pd.read_csv("./data/raw_data/ppmclab.tsv", delimiter="\t")
len(df)

df.loc[df["Type"] == "non-PSP", "ps_label"] = 0
df.loc[df["Type"] == "noID-PSP", "ps_label"] = 1
df.loc[df["Type"] == "ID-PSP", "ps_label"] = 1

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["ps_label"], random_state=13
)

len(train_df.loc[train_df["ps_label"] == 1])
len(train_df.loc[train_df["ps_label"] == 0])

len(val_df.loc[val_df["ps_label"] == 1])
len(val_df.loc[val_df["ps_label"] == 0])

with open("./data/intermediate_data/ppmclab_rsa.pkl", "br") as f:
    df = pickle.load(f)

len(df)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["ps_label"], random_state=13
)

len(train_df.loc[train_df["ps_label"] == 1])
len(train_df.loc[train_df["ps_label"] == 0])

len(val_df.loc[val_df["ps_label"] == 1])
len(val_df.loc[val_df["ps_label"] == 0])

df = df.loc[df["rsa"].notna()]
len(df)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["ps_label"], random_state=13
)

len(train_df.loc[train_df["ps_label"] == 1])
len(train_df.loc[train_df["ps_label"] == 0])

len(val_df.loc[val_df["ps_label"] == 1])
len(val_df.loc[val_df["ps_label"] == 0])

df = add_ptm(df, "./data/intermediate_data/ptm_ppmclab.pkl", "UniProt.Acc")

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["ps_label"], random_state=13
)

len(df[df["ptm_profile"].apply(sum) > 0])

train_df = train_df[train_df["ptm_profile"].apply(sum) > 0]
val_df = val_df[val_df["ptm_profile"].apply(sum) > 0]

len(train_df.loc[train_df["ps_label"] == 1])
len(train_df.loc[train_df["ps_label"] == 0])

len(val_df.loc[val_df["ps_label"] == 1])
len(val_df.loc[val_df["ps_label"] == 0])
