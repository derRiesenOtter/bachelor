import pickle
import time
from pathlib import Path

import requests
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBParser import PDBParser

dataset = "ppmclab"

with open(f"./data/intermediate_data/{dataset}.pkl", "rb") as f:
    df = pickle.load(f)


df["rsa"] = None

for idx, row in df.iterrows():
    id = row["UniProt.Acc"]
    file = f"./data/raw_data/alpha_pdb/{id}.pdb"

    try:
        if not Path(file).exists():
            url = f"https://alphafold.com/api/prediction/{id}?key=AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94"
            r = requests.get(url)
            pdb_url = r.json()[0]["pdbUrl"]

            r2 = requests.get(pdb_url)

            with open(file, "wb") as f:
                f.write(r2.content)

            time.sleep(0.1)
        parser = PDBParser()
        structure = parser.get_structure("test", file)
        dssp = DSSP(structure[0], file)
        rsa_values = [dssp[key][3] for key in dssp.keys()]
        df.at[idx, "rsa"] = rsa_values
    except (KeyError, Exception):
        df.at[idx, "rsa"] = None


with open(f"./data/intermediate_data/{dataset}_rsa.pkl", "wb") as f:
    pickle.dump(df, f)
