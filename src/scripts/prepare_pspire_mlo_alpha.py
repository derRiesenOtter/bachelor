import os
import pickle
import time
from pathlib import Path

import requests
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBParser import PDBParser

files = os.listdir("./data/intermediate_data/")
relevant_files = [
    file
    for file in files
    if "pspire_" in file and not "bd" in file and not "dvc" in file
]

for file in relevant_files:
    with open(f"./data/intermediate_data/{file}", "rb") as f:
        df = pickle.load(f)

    df["rsa"] = None

    for idx, row in df.iterrows():
        id = row["UniprotEntry"]
        fil = f"./data/raw_data/alpha_pdb/{id}.pdb"

        if not Path(fil).exists():
            url = f"https://alphafold.com/api/prediction/{id}?key=AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94"

            r = requests.get(url)
            pdb_url = r.json()[0]["pdbUrl"]

            r2 = requests.get(pdb_url)

            with open(fil, "wb") as f:
                f.write(r2.content)

            time.sleep(0.1)

        parser = PDBParser()
        structure = parser.get_structure("test", fil)
        dssp = DSSP(structure[0], fil)
        rsa_values = [dssp[key][3] for key in dssp.keys()]
        df.at[idx, "rsa"] = rsa_values

    with open(f"./data/intermediate_data/{file[:-4]}_rsa.pkl", "wb") as f:
        pickle.dump(df, f)
