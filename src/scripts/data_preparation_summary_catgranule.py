import pickle

import pandas as pd

from src.modules.get_ptm import add_ptm

df = pd.read_csv("./data/raw_data/catgranule2.csv")
df = df[(df["training"] == 1) | (df["test"] == 1)]
len(df)

df = df.rename(columns={"Unnamed: 0": "UniprotEntry", "known_LLPS": "ps_label"})

len(df.loc[(df["training"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["training"] == 1) & (df["ps_label"] == 0)])

len(df.loc[(df["test"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["test"] == 1) & (df["ps_label"] == 0)])

with open("./data/intermediate_data/catgranule2_rsa.pkl", "br") as f:
    df = pickle.load(f)

df = df[(df["training"] == 1) | (df["test"] == 1)]
len(df)

len(df.loc[(df["training"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["training"] == 1) & (df["ps_label"] == 0)])

len(df.loc[(df["test"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["test"] == 1) & (df["ps_label"] == 0)])

df = df.loc[df["rsa"].notna()]
len(df)

len(df.loc[(df["training"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["training"] == 1) & (df["ps_label"] == 0)])

len(df.loc[(df["test"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["test"] == 1) & (df["ps_label"] == 0)])

df = add_ptm(df, "./data/intermediate_data/catgranule2_ptm.pkl")

len(df[df["ptm_profile"].apply(sum) > 0])
df = df[df["ptm_profile"].apply(sum) > 0]

len(df.loc[(df["training"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["training"] == 1) & (df["ps_label"] == 0)])

len(df.loc[(df["test"] == 1) & (df["ps_label"] == 1)])
len(df.loc[(df["test"] == 1) & (df["ps_label"] == 0)])
