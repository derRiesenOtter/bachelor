import pickle
from functools import partial
from multiprocessing import Pool, freeze_support

from src.modules import Words
from src.modules.bd_tools import get_block_seq
from src.modules.Words.Mappings import (
    APNAMapping,
    IDRMapping,
    MM5Mapping,
    PIPIFMapping,
    PIPIGMapping,
    RG2Mapping,
    RGMapping,
)


def main():
    with open("./data/intermediate_data/phasepdb.pkl", "rb") as f:
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
            data[f"{mapping.__name__}_{balance_threshold}"] = p.map(func, data["seq"])

            data[f"{mapping.__name__}_{balance_threshold}_vec"] = [
                get_block_seq(seq, blocks, mapping.Forward)
                for seq, blocks in zip(
                    data["seq"], data[f"{mapping.__name__}_{balance_threshold}"]
                )
            ]

            balance_threshold = 2
            func = partial(
                create_block_list, balance_threshold=balance_threshold, mapping=mapping
            )
            data[f"{mapping.__name__}_{balance_threshold}"] = p.map(func, data["seq"])
            data[f"{mapping.__name__}_{balance_threshold}_vec"] = [
                get_block_seq(seq, blocks, mapping.Forward)
                for seq, blocks in zip(
                    data["seq"], data[f"{mapping.__name__}_{balance_threshold}"]
                )
            ]

    with open("./data/intermediate_data/phasepdb_bd.pkl", "bw") as f:
        pickle.dump(data, f)


# def get_block_share_list(row) -> list:
#     block_share_list = []
#     column_list = [col for col in row.index if "_vec" in col]
#     counter = 0
#     mapping = 0
#     for block_seq in column_list:
#         block_share_list.extend(get_scalar_values(row[block_seq],
#                                                   mappings[mapping].Forward))
#         counter += 1
#         if counter % 2 == 0:
#             mapping += 1
#     return block_share_list


def create_block_list(
    sequence: str, balance_threshold: int, mapping
) -> list[tuple[int, int]]:
    w = Words.Word.MappableWord([_ for _ in sequence], mapping)
    block_list = w.find_balanced_subwords(balance_threshold)
    return block_list


if __name__ == "__main__":
    freeze_support()
    main()
