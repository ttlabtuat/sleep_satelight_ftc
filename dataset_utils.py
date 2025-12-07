

def assign_nights(num_subjects: int, onenight_subs: list[int]) -> dict[int, list[int]]:
    """
    Create a mapping from subject index to 1- or 2-night PSG indices.

    Parameters
    ----------
    num_subjects : int
        Number of subjects in the dataset.
    onenight_subs : list[int]
        List of subject indices that have only one night of PSG.

    Returns
    -------
    dict[int, list[int]]
        A dictionary mapping each subject index to a list of night indices.
    """
    assignments: dict[int, list[int]] = {}
    current_value = 0
    for i in range(num_subjects):
        if i in onenight_subs:
            assignments[i] = [current_value]
            current_value += 1
        else:
            assignments[i] = [current_value, current_value + 1]
            current_value += 2
    return assignments


def get_dataset_params(dataset_name: str, fs: int = 50) -> dict:
    """
    dataset_name: 'edf20' or 'edf78'
    fs: sampling rate (default 50)
    """

    # SleepEDF parameters
    if dataset_name == "edf20":
        ids = list(range(39))
        assignments = assign_nights(20, [12])
        fold_num = 20

    elif dataset_name == "edf78":
        ids = list(range(153))
        assignments = assign_nights(78, [12, 35, 51])
        fold_num = 10

    else:
        raise ValueError(f"{dataset_name} not supported")

    return {
        "ids": ids,
        "num": len(ids),
        "fold_num": fold_num,
        "assignments": assignments,
        "fs": fs,
        "segment_length": 30,
        "overlap_ratio": 0.0,
        "freq_params": {
            "time_bandwidth": 4.0,
            "n_tapers": 7,
        }
    }




