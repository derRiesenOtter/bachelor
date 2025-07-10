import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataSet(Dataset):
    def __init__(
        self,
        df,
        feature_col: str,
        ptm_column: str,
        label_col: str,
        max_len: int = 2700,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_column = feature_col
        self.ptm = self.df[ptm_column]
        self.label_col = label_col
        self.label = self.df[label_col].astype(int)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = self.df[self.feature_column][idx]
        padded_feature = torch.tensor(
            np.append(feature, np.repeat(-1, self.max_len - len(feature)))
        )
        ptm_feature = self.ptm[idx]
        padded_ptm_feature = torch.tensor(
            np.append(ptm_feature, np.repeat(-1, self.max_len - len(ptm_feature))),
            dtype=torch.int64,
        )

        return padded_feature, padded_ptm_feature, self.label[idx]
