from vap.utils import read_json
import torch
import numpy as np


"""
Just to show how I load the output  saved by `run_stereo.py`
"""


def load_np(path):
    d = read_json(path)
    for k, v in d.items():
        if k == "vad_list":
            continue
        d[k] = np.array(v)
    return d


def load_torch(path):
    d = read_json(path)
    for k, v in d.items():
        if k == "vad_list":
            continue
        d[k] = torch.tensor(v)
    return d


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Please provide `filepath` argument.")
        print("`python load_stereo.py her_output.json`")
        sys.exit(0)
    filepath = sys.argv[1]
    d = load_torch(filepath)

    print("-" * 50)
    print(filepath)
    print("-" * len(filepath))
    for k, v in d.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(f"{k}: {tuple(v.shape)}, {type(v)}")
        else:
            print(f"{k}: {len(v[0])}, {type(v)}")
    print("-" * 50)
