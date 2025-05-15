import sys

import pandas as pd
import requests


def main():
    df = pd.read_csv("./data/raw_data/pspire.csv")

    df_with_seq = pd.DataFrame()
    counter = 0
    protein_list = []
    for protein in df["UniprotEntry"]:
        counter += 1
        protein_list.append(protein)
        if counter == 100:
            df_slice = get_sequences(protein_list)
            counter = 0
            df_with_seq._append(df_slice)
    print(df)


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
    seq = [seq["sequence"]["value"] for seq in data["results"]]

    df = pd.DataFrame({"id": ids, "seq": seq})

    return df


if __name__ == "__main__":
    main()
