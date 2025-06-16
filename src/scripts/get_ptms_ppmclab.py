import pickle

import pandas as pd

from src.modules.feature_annotations import annot_PTMs

df = pd.DataFrame()

files = [
    "./data/intermediate_data/ppmclab_rsa.pkl",
]

for file in files:
    with open(file, "rb") as f:
        df_tmp = pickle.load(f)
    df = pd.concat([df, df_tmp["UniProt.Acc"]])

df = df.drop_duplicates(subset=["UniProt.Acc"], keep=False)

proteins = [protein for protein in df["UniProt.Acc"]]

annotated_df = annot_PTMs(proteins)

with open("./data/intermediate_data/ptm_ppmclab.pkl", "wb") as f:
    pickle.dump(annotated_df, f)
