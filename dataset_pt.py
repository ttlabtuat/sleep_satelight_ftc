# dataset_pt.py
import numpy as np
import torch
from torch.utils.data import Dataset
from data_loader import load_edfds


class SleepDataset(Dataset):
    """
    PyTorch Dataset converting numpy to torch, no processing changed.
    """

    def __init__(self, dataset_dir, indices, fs=100):
        # Load with the original function (no change)
        eeg_time, eeg_freq, eeg_label = load_edfds(dataset_dir, indices, fs)

        # Convert numpy → torch tensor
        self.eeg_time = torch.from_numpy(eeg_time).float()
        self.eeg_freq = torch.from_numpy(eeg_freq).float()
        self.labels = torch.from_numpy(eeg_label).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return shape exactly as Keras expected
        return self.eeg_time[idx], self.eeg_freq[idx], self.labels[idx]
