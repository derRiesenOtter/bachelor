import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from src.modules.bd_tools import map_sequence
from src.modules.mappings import AAMapping


def main():
    df = pd.read_csv("./data/raw_data/pspire.csv")

    # requesting the sequences from uniprot
    df_with_seq = pd.DataFrame()
    counter = 0
    protein_list = []
    for protein in df["UniprotEntry"]:
        counter += 1
        protein_list.append(protein)
        if counter == 50:
            df_slice = get_sequences(protein_list)
            counter = 0
            df_with_seq = pd.concat([df_with_seq, df_slice])
            protein_list = []
    df_slice = get_sequences(protein_list)
    df_with_seq = pd.concat([df_with_seq, df_slice])
    df_joined = pd.merge(df, df_with_seq, left_on="UniprotEntry", right_on="id")
    sns.kdeplot(np.array(df_joined["seq"].apply(len)))
    plt.xlabel("Sequence Length in residues")
    plt.ylabel("Density")
    plt.savefig("./results/plots/pspire_sequence_length_density.png")
    plt.show()

    # creating column with mapped sequences
    df_joined["mapped_seq"] = df_joined["seq"].apply(map_sequence, args=(AAMapping,))
    unfiltered_len = len(df_joined)
    # Filtering too long and too short sequences
    print(f"This data set contains {unfiltered_len} entries")
    filtered_df = df_joined.loc[
        (df_joined["mapped_seq"].apply(len) <= 2700)
        & (df_joined["mapped_seq"].apply(len) > 9)
    ]
    print(
        f"After filtering out sequences that are longer than 2700 or shorter than 10 {len(filtered_df)} entries were kept"
    )
    # adding a ps label
    filtered_df.loc[filtered_df["Type"] == "non-PSP", "ps_label"] = 0
    filtered_df.loc[filtered_df["Type"] == "noID-PSP", "ps_label"] = 1
    filtered_df.loc[filtered_df["Type"] == "ID-PSP", "ps_label"] = 1

    # adding a idr label
    filtered_df.loc[filtered_df["Type"] == "non-PSP", "idr_protein"] = 0
    filtered_df.loc[filtered_df["Type"] == "noID-PSP", "idr_protein"] = 0
    filtered_df.loc[filtered_df["Type"] == "ID-PSP", "idr_protein"] = 1

    print(
        f"Of these filtered proteins {len(filtered_df.loc[filtered_df["ps_label"] == 0])} were negatives"
    )
    print(
        f"Of these filtered proteins {len(filtered_df.loc[filtered_df["ps_label"] == 1])} were positives"
    )
    print(
        f"Of these filtered proteins {len(filtered_df.loc[filtered_df["idr_protein"] == 1])} were positives with an idr"
    )
    print(
        f"Of these filtered proteins {len(filtered_df.loc[filtered_df["ps_label"] == 1])
    -len(filtered_df.loc[filtered_df["idr_protein"] == 1])} were positives with an idr"
    )

    x = ["PS", "nPS", "filtered out"]
    y1 = np.array([0, 0, unfiltered_len - len(filtered_df)])
    y2 = np.array([0, len(filtered_df.loc[filtered_df["ps_label"] == 0]), 0])
    y3 = np.array([len(filtered_df.loc[filtered_df["idr_protein"] == 1]), 0, 0])
    y4 = np.array(
        [
            len(filtered_df.loc[filtered_df["ps_label"] == 1])
            - len(filtered_df.loc[filtered_df["idr_protein"] == 1]),
            0,
            0,
        ]
    )
    colors = plt.color_sequences["Paired"]

    plt.ylim(0, 11000)
    plt.bar(x, y1, color=colors[8])
    plt.bar(x, y2, bottom=y1, color=colors[0])
    plt.bar(x, y3, bottom=y1 + y2, color=colors[4])
    plt.bar(x, y4, bottom=y1 + y2 + y3, color=colors[5])
    plt.legend(["Filtered out", "nPS", "IDR PS", "nIDR PS"])

    plt.savefig("./results/plots/pspire_sample_dist.png")
    plt.show()

    with open("./data/intermediate_data/pspire.pkl", "wb") as f:
        pickle.dump(filtered_df, f)


def get_sequences(protein_list: list) -> pd.DataFrame:
    query = " OR ".join(f"accession:{protein}" for protein in protein_list)
    params = {
        "query": query,
        "fields": ["sequence"],
        "sort": "accession desc",
        "size": "100",
    }
    headers = {"accept": "application/json"}
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    response = requests.get(base_url, headers=headers, params=params)
    if not response.ok:
        response.raise_for_status()
        sys.exit()

    data = response.json()
    ids = [id["primaryAccession"] for id in data["results"]]
    seq = []
    for seq_data in data["results"]:
        try:
            seq.append(seq_data["sequence"]["value"])
        except KeyError:
            seq.append("")

    df = pd.DataFrame({"id": ids, "seq": seq})

    return df


if __name__ == "__main__":
    filename = os.path.basename(__file__)
    sys.stdout = open(f"./results/stdout/{filename}.txt", "wt")
    main()
