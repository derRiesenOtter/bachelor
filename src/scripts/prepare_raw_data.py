import os
import pickle
import sys

import pandas as pd


def main():
    df = pd.read_csv("./data/raw_data/llps_data_ppmclab.tsv", delimiter="\t")

    df.loc[~df["Datasets"].isin(["NP", "ND"]), "PS"] = 1
    df.loc[df["Datasets"].isin(["NP", "ND"]), "PS"] = 0

    filtered_df = df[~df["Full.seq"].str.contains("X|U", na=False)]

    print(f"Size of the whole data set: {len(df)} proteins.")
    print(f"Size of the filtered data set: {len(filtered_df)} proteins.")
    print("Positive Samples:")
    print(len(df.loc[~df["Datasets"].isin(["NP", "ND"]), "PS"]))
    print("Negative Samples:")
    print(len(df.loc[df["Datasets"].isin(["NP", "ND"]), "PS"]))

    with open("./data/intermediate_data/llps_data_ppmclab.pkl", "wb") as f:
        pickle.dump(filtered_df, f)


if __name__ == "__main__":
    filename = os.path.basename(__file__)
    sys.stdout = open(f"./results/stdout/{filename}.txt", "wt")
    main()
