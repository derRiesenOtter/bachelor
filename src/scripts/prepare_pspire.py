# import pickle
# import sys
# from io import StringIO
#
# import pandas as pd
# import requests
# from Bio import SeqIO
#
# df = pd.read_csv("./data/raw_data/ps_pire_data.csv")
#
# print(f"Data set contains {len(df)} sequences.")
#
#
# def get_sequence(id):
#
#     requestURL = (
#         f"https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size=1&accession={id}"
#     )
#     r = requests.get(requestURL, headers={"Accept": "text/x-fasta"})
#
#     if not r.ok:
#         r.raise_for_status()
#         sys.exit()
#
#     responseBody = r.text
#     fasta = SeqIO.read(StringIO(responseBody), "fasta")
#     return fasta.seq
#
#
# df["seq"] = df["UniprotEntry"].apply(get_sequence)
#
# df["seq"] = df["UniprotEntry"].apply(get_sequence)
#
# print(f"{len(df)} sequences were downloaded.")
#
# with open("./data/intermediate_data/ps_pire_data.pkl", "wb") as f:
#     pickle.dump(df, f)


import json
import sys

import requests

params = {
    "query": "accession:O95613 OR accession:P08047",
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

for item in data["results"]:
    print(item["primaryAccession"])
# print(data["results"][0]["primaryAccession"])
# sequence = data["results"][0]["sequence"]["value"]
# print(sequence)
# print(data["results"][1]["primaryAccession"])
# sequence = data["results"][1]["sequence"]["value"]
# print(sequence)
