import pickle

import numpy as np
import pandas as pd

CATEGORY_MAP = {
    "phosphorylation": 1,
    "acetylation": 2,
    "methylation": 3,
    "ubiquitin_like": 4,
    "adp_ribosylation": 5,
    "glycosylation": 6,
    "oxidation": 7,
    "other": 8,
}


def classify_ptm(ptm_str: str) -> int:
    ptm_str = ptm_str.lower()
    if "phospho" in ptm_str:
        return CATEGORY_MAP["phosphorylation"]
    elif "acetyl" in ptm_str:
        return CATEGORY_MAP["acetylation"]
    elif "methyl" in ptm_str:
        return CATEGORY_MAP["methylation"]
    elif (
        "ubiquitin" in ptm_str
        or "sumo" in ptm_str
        or "nedd8" in ptm_str
        or "isg15" in ptm_str
    ):
        return CATEGORY_MAP["ubiquitin_like"]
    elif "adp-ribosyl" in ptm_str or "polyadp" in ptm_str:
        return CATEGORY_MAP["adp_ribosylation"]
    elif (
        "glcnac" in ptm_str
        or "galnac" in ptm_str
        or "glycos" in ptm_str
        or "xyl" in ptm_str
        or "fuc" in ptm_str
        or "man" in ptm_str
    ):
        return CATEGORY_MAP["glycosylation"]
    elif "sulfoxide" in ptm_str or "hydroxy" in ptm_str or "oxid" in ptm_str:
        return CATEGORY_MAP["oxidation"]
    else:
        return CATEGORY_MAP["other"]


def add_ptm(df, ptm_file="./data/intermediate_data/ptm.pkl", id="UniprotEntry"):
    with open(ptm_file, "rb") as f:
        ptm_df = pickle.load(f)

    ptm_df["category"] = ptm_df["ptm"].apply(classify_ptm)

    ptm_profiles = []

    for idx, row in df.iterrows():
        protein = row[id]
        length = len(row["mapped_seq"])
        profile = [0] * length

        prot_ptms = ptm_df[ptm_df["protein_name"] == protein]

        for _, ptm_entry in prot_ptms.iterrows():
            pos = int(ptm_entry["pos"])
            cat = ptm_entry["category"]

            if 0 < pos <= length:
                profile[pos - 1] = cat

        ptm_profiles.append(profile)

    df["ptm_profile"] = ptm_profiles
    return df
