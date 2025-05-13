import os
import pickle
import sys
from functools import partial
from multiprocessing import Pool, freeze_support

from src.modules import Words
from src.modules.block_decomposition_modifier import get_block_seq
from src.modules.Words.Mappings import (
    APNAMapping,
    IDRMapping,
    MM5Mapping,
    PIPIFMapping,
    PIPIGMapping,
    RG2Mapping,
    RGMapping,
)


def create_block_list(
    sequence: str, balance_threshold: int, mapping
) -> list[tuple[int, int]]:
    w = Words.Word.MappableWord([_ for _ in sequence], mapping)
    block_list = w.find_balanced_subwords(balance_threshold)
    return block_list


def main():
    with open("./data/intermediate_data/llps_data_ppmclab.pkl", "rb") as f:
        data = pickle.load(f)
    mappings = [
        APNAMapping,
        RGMapping,
        RG2Mapping,
        PIPIFMapping,
        PIPIGMapping,
        MM5Mapping,
        IDRMapping,
    ]
    with Pool(8) as p:
        for mapping in mappings:
            balance_threshold = 4
            func = partial(
                create_block_list, balance_threshold=balance_threshold, mapping=mapping
            )
            data[f"{mapping.__name__}_{balance_threshold}"] = p.map(
                func, data["Full.seq"]
            )

            data[f"{mapping.__name__}_{balance_threshold}_vec"] = [
                get_block_seq(seq, blocks, mapping.Forward)
                for seq, blocks in zip(
                    data["Full.seq"], data[f"{mapping.__name__}_{balance_threshold}"]
                )
            ]

            balance_threshold = 3
            func = partial(
                create_block_list, balance_threshold=balance_threshold, mapping=mapping
            )
            data[f"{mapping.__name__}_{balance_threshold}"] = p.map(
                func, data["Full.seq"]
            )
            data[f"{mapping.__name__}_{balance_threshold}_vec"] = [
                get_block_seq(seq, blocks, mapping.Forward)
                for seq, blocks in zip(
                    data["Full.seq"], data[f"{mapping.__name__}_{balance_threshold}"]
                )
            ]

    with open("./data/intermediate_data/llps_data_ppmclab_bd.pkl", "bw") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    freeze_support()
    filename = os.path.basename(__file__)
    sys.stdout = open(f"./results/stdout/{filename}.txt", "wt")
    main()
