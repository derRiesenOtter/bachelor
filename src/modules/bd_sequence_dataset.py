import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BDSequenceDataSet(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, max_len: int = 2700):
        self.df = df.reset_index(drop=True)
        self.feature_columns = [col for col in self.df.columns if "feat" in col]
        self.label_col = label_col
        self.label = self.df[label_col].astype(int)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        channels = [self.df[channel][idx] for channel in self.feature_columns]
        length = len(channels[0])
        padded_channels = [
            torch.tensor(np.append(channel, np.repeat(-2, self.max_len - length)))
            for channel in channels
        ]
        stacked_channels = torch.stack(padded_channels)
        return stacked_channels, self.label[idx]
