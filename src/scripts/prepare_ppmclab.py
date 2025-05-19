import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.modules.bd_tools import map_sequence
from src.modules.mappings import AAMapping


def main():
    df = pd.read_csv("./data/raw_data/ppmclab.tsv", delimiter="\t")
    size_unfiltered_df = len(df)
    print(f"Size of the whole data set: {len(df)} proteins.")

    # visualize sequence length
    sns.kdeplot(np.array(df["Full.seq"].apply(len)))
    plt.title("Density of the sequence length")
    plt.xlabel("Sequence Length in residues")
    plt.ylabel("Density")
    plt.savefig("./results/plots/ppmclab_sequence_length_density.png")
    plt.show()

    # filter sequences that are longer than 2700 residuals long
    df = df.loc[df["Full.seq"].str.len() <= 2700]

    # filter out rows that contain letters that are not in the amino acid seq
    df = df.loc[~df["Full.seq"].str.contains("X|U")]

    # add a label column with 0 for negatives and 1 for positives
    df.loc[~df["Datasets"].isin(["NP", "ND"]), "ps_label"] = 1
    df.loc[df["Datasets"].isin(["NP", "ND"]), "ps_label"] = 0

    # add a label column with 0 for negatives, 1 for clients and 2 for drivers
    # and 3 for client and driver proteins
    df.loc[df["Datasets"].isin(["NP", "ND"]), "ps_cd_label"] = 0
    df.loc[df["Datasets"].str.contains("CE"), "ps_cd_label"] = 1
    df.loc[df["Datasets"].str.contains("DE"), "ps_cd_label"] = 2
    df.loc[df["Datasets"].str.contains("C_D"), "ps_cd_label"] = 3

    # add a column that represents whether the protein has idr regions
    df["idr_protein"] = 0
    df.loc[df["Frac.Order"].gt(0), "idr_protein"] = 1

    # create a column for the mapped_seq
    df["mapped_seq"] = df["Full.seq"].apply(map_sequence, args=(AAMapping,))

    print(f"Size of the filtered data set: {len(df)} proteins.")
    print("Positive Samples:")
    print(len(df.loc[~df["Datasets"].isin(["NP", "ND"])]))
    print("Negative Samples:")
    print(len(df.loc[df["Datasets"].isin(["NP", "ND"])]))
    print("Client exclusive proteins:")
    print(len(df.loc[df["Datasets"].str.contains("CE")]))
    print("Driver exclusive proteins:")
    print(len(df.loc[df["Datasets"].str.contains("DE")]))
    print("Driver and Client proteins:")
    print(len(df.loc[df["Datasets"].str.contains("C_D")]))
    print("Proteins containing IDRs:")
    print(len(df.loc[df["idr_protein"].gt(0)]))

    x = ["PS", "nPS", "filtered out"]
    y1 = [0, 0, size_unfiltered_df - len(df)]
    y2 = np.array(
        [
            len(df.loc[df["Datasets"].str.contains("CE") & (df["idr_protein"] == 0)]),
            0,
            0,
        ]
    )
    y3 = np.array(
        [
            len(df.loc[df["Datasets"].str.contains("CE") & (df["idr_protein"] == 1)]),
            0,
            0,
        ]
    )
    y4 = np.array(
        [
            len(df.loc[df["Datasets"].str.contains("DE") & (df["idr_protein"] == 0)]),
            0,
            0,
        ]
    )
    y5 = np.array(
        [
            len(df.loc[df["Datasets"].str.contains("DE") & (df["idr_protein"] == 1)]),
            0,
            0,
        ]
    )
    y6 = np.array(
        [
            len(df.loc[df["Datasets"].str.contains("C_D") & (df["idr_protein"] == 0)]),
            0,
            0,
        ]
    )
    y7 = np.array(
        [
            len(df.loc[df["Datasets"].str.contains("C_D") & (df["idr_protein"] == 1)]),
            0,
            0,
        ]
    )
    y8 = np.array(
        [
            0,
            len(df.loc[df["Datasets"].isin(["NP", "ND"]) & (df["idr_protein"] == 0)]),
            0,
        ]
    )
    y9 = np.array(
        [
            0,
            len(df.loc[df["Datasets"].isin(["NP", "ND"]) & (df["idr_protein"] == 1)]),
            0,
        ]
    )

    colors = plt.color_sequences["Paired"]

    plt.ylim(0, 2500)
    plt.bar(x, y1, color=colors[8])
    plt.bar(x, y2, bottom=y1, color=colors[0])
    plt.bar(x, y3, bottom=y1 + y2, color=colors[1])
    plt.bar(x, y4, bottom=y1 + y2 + y3, color=colors[2])
    plt.bar(x, y5, bottom=y1 + y2 + y3 + y4, color=colors[3])
    plt.bar(x, y6, bottom=y1 + y2 + y3 + y4 + y5, color=colors[4])
    plt.bar(x, y7, bottom=y1 + y2 + y3 + y4 + y5 + y6, color=colors[5])
    plt.bar(x, y8, bottom=y1 + y2 + y3 + y4 + y5 + y6 + y7, color=colors[6])
    plt.bar(x, y9, bottom=y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8, color=colors[7])
    plt.legend(
        [
            "Filtered out",
            "nIDR CE",
            "IDR CE",
            "nIDR DE",
            "IDR DE",
            "nIDR C_D",
            "IDR C_D",
            "nIDR N",
            "IDR N",
        ]
    )
    plt.savefig("./results/plots/ppmclab_sample_dist.png")
    plt.show()

    with open("./data/intermediate_data/ppmclab.pkl", "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    filename = os.path.basename(__file__)
    sys.stdout = open(f"./results/stdout/{filename}.txt", "wt")
    main()
