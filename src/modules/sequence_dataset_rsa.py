import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataSet(Dataset):
    def __init__(
        self, df, feature_col: str, rsa_column: str, label_col: str, max_len: int = 2700
    ):
        self.df = df.reset_index(drop=True)
        self.feature_column = feature_col
        self.rsa = self.df[rsa_column]
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
        rsa_feature = self.rsa[idx]
        padded_rsa_feature = torch.tensor(
            np.append(rsa_feature, np.repeat(-1, self.max_len - len(rsa_feature))),
            dtype=torch.float32,
        )

        return padded_feature, padded_rsa_feature, self.label[idx]
