import glob
import os
from typing import Iterable, Tuple

import numpy as np
import nitime.algorithms as tsa


def edfds_paths(dataset_dir: str) -> list[str]:
    """
    Return sorted list of EDF-derived .npz file paths.

    Parameters
    ----------
    dataset_dir : str
        Directory where EDF-derived .npz files are stored.
    """
    npz_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    return sorted(npz_files)


def load_edfds(
    dataset_dir: str,
    indices: Iterable[int],
    fs: int = 100,
    n_classes: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load EDF-derived npz files and compute time-domain and frequency-domain inputs.

    This function encapsulates the logic that was previously duplicated in train_1.py
    so that both train_1.py and train_2.py can share the same implementation.

    Parameters
    ----------
    dataset_dir : str
        Directory where EDF-derived .npz files are stored.
    indices : Iterable[int]
        Iterable of file indices (corresponding to sorted npz paths) to load.
    fs : int, default 100
        Sampling frequency used when computing multi-taper PSD.
    n_classes : int, default 5
        Number of classes for one-hot encoding of labels.

    """
    paths = edfds_paths(dataset_dir)

    eeg_time_list = []
    eeg_freq_list = []
    eeg_label_list = []

    for i in indices:
        loaded_npz = np.load(paths[i])

        x = loaded_npz["x"]  # shape: (N, L)
        eeg_one = x[:, np.newaxis, :]  # (N, 1, L)

        # (N, 1, L, 1) for Conv2D
        eeg_time_list.append(eeg_one[:, :, :, np.newaxis])

        # multi-taper PSD
        freq, P_mt, nu = tsa.multi_taper_psd(eeg_one, Fs=fs)
        half_len = eeg_one.shape[-1] // 2
        amp_spectrum = np.sqrt(P_mt[:, :, :half_len])
        amp_spectrum_dB = 20 * np.log10(amp_spectrum)

        # (N, 1, L/2, 1)
        eeg_freq_list.append(amp_spectrum_dB[:, :, :, np.newaxis])

        eeg_label_list.append(loaded_npz["y"])

    # Concatenate data loaded from each npz file
    eeg_time = np.concatenate(eeg_time_list, axis=0)
    eeg_freq = np.concatenate(eeg_freq_list, axis=0)
    eeg_label = np.concatenate(eeg_label_list, axis=0)

    return eeg_time, eeg_freq, eeg_label




