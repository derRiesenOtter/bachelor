import pickle

import pandas as pd

from src.modules.get_ptm import add_ptm

df = pd.read_csv("./data/raw_data/pspire.csv")
len(df)

df.loc[df["Type"] == "non-PSP", "ps_label"] = 0
df.loc[df["Type"] == "noID-PSP", "ps_label"] = 1
df.loc[df["Type"] == "ID-PSP", "ps_label"] = 1

len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 0)])

len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 0)])

with open("./data/intermediate_data/pspire_rsa.pkl", "br") as f:
    df = pickle.load(f)

len(df)

len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 0)])

len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 0)])

df = df.loc[df["rsa"].notna()]
len(df)

len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 0)])

len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 0)])

df = add_ptm(df)

len(df[df["ptm_profile"].apply(sum) > 0])
df = df[df["ptm_profile"].apply(sum) > 0]

len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Training") & (df["ps_label"] == 0)])

len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 1)])
len(df.loc[(df["Datasets"] == "Testing") & (df["ps_label"] == 0)])
