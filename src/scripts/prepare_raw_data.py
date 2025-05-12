import pickle

import pandas as pd

data = pd.read_csv("./data/raw_data/llps_data_ppmclab.tsv", delimiter="\t")

data.loc[~data["Datasets"].isin(["NP", "ND"]), "PS"] = 0
data.loc[data["Datasets"].isin(["NP", "ND"]), "PS"] = 1

with open("./data/intermediate_data/llps_data_ppmclab.pkl", "wb") as f:
    pickle.dump(data, f)
