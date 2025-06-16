import pickle

import pandas as pd

from src.modules.feature_annotations import annot_PTMs

df = pd.DataFrame()

files = [
    "./data/intermediate_data/pspire_rsa.pkl",
    "./data/intermediate_data/pspire_DACT1-particulate proteome_rsa.pkl",
    "./data/intermediate_data/pspire_DrLLPS_MLO_rsa.pkl",
    "./data/intermediate_data/pspire_G3BP1 proximity labeling_rsa.pkl",
    "./data/intermediate_data/pspire_PhaSepDB_MLO_rsa.pkl",
    "./data/intermediate_data/pspire_RNAgranuleDB_tier1_rsa.pkl",
]

for file in files:
    with open(file, "rb") as f:
        df_tmp = pickle.load(f)
    df = pd.concat([df, df_tmp["UniprotEntry"]])

df = df.drop_duplicates(subset=["UniprotEntry"], keep=False)

proteins = [protein for protein in df["UniprotEntry"]]

annotated_df = annot_PTMs(proteins)

with open("./data/intermediate_data/ptm.pkl", "wb") as f:
    pickle.dump(annotated_df, f)
