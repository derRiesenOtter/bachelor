import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from src.modules.bd_tools import map_sequence
from src.modules.mappings import AAMapping


def main():

    df = pd.read_csv("./data/raw_data/pspire_mlo.csv")
    df.columns

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
    df_joined["ps_label"] = 1
    sns.kdeplot(np.array(df_joined["seq"].apply(len)))
    plt.xlabel("Sequence Length in residues")
    plt.ylabel("Density")
    plt.savefig("./results/plots/pspire_sequence_length_density.png")
    plt.show()

    df_joined["mapped_seq"] = df_joined["seq"].apply(map_sequence, args=(AAMapping,))
    for mlo_list in df_joined.drop(
        columns=["id", "UniprotEntry", "mapped_seq", "ps_label", "Type", "seq"]
    ):
        mlo_df = df_joined.loc[df_joined[mlo_list] == 1][
            ["UniprotEntry", "mapped_seq", "ps_label", "Type"]
        ]
        mlo_df = mlo_df.loc[
            (mlo_df["mapped_seq"].apply(len) <= 2700)
            & (mlo_df["mapped_seq"].apply(len) > 9)
            & (
                df_joined["mapped_seq"].apply(
                    lambda seq: all(not pd.isna(x) for x in seq)
                )
            )
        ]
        with open(f"./data/intermediate_data/pspire_{mlo_list}.pkl", "wb") as f:
            pickle.dump(mlo_df, f)


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
    filename = Path(__file__).stem
    sys.stdout = open(f"./results/stdout/{filename}.txt", "wt")
    main()
