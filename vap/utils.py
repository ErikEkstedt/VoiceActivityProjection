import torch
from torch import Tensor
import json
from os.path import dirname
from typing import List, Optional, Tuple

from vap.audio import time_to_frames, load_waveform

VAD_LIST = List[List[List[float]]]


################################################
# Tensor Helpers
################################################
def add_zero_channel(w: Tensor) -> Tensor:
    """Add silent channel as speaker 'B'"""
    z = torch.zeros_like(w)
    return torch.cat((w, z), dim=-2)


def find_island_idx_len(
    x: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat(
        (torch.where(y)[0], torch.tensor(n - 1, device=x.device).unsqueeze(0))
    ).long()
    it = torch.cat((torch.tensor(-1, device=x.device).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(
        torch.cat((torch.tensor([0], device=x.device, dtype=torch.long), dur)), dim=0
    )[
        :-1
    ]  # positions
    return idx, dur, x[i]


def everything_deterministic():
    """
    -----------------------------
    Wav2Vec
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: replication_pad1d_backward_cuda does not have a deterministic
    implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can
    turn off determinism just for this operation if that's acceptable for your
    application. You can also file an issue at
    https://github.com/pytorch/pytorch/issues to help us prioritize adding
    deterministic support for this operation.


    -----------------------------
    CPC
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: Deterministic behavior was enabled with either
    `torch.use_deterministic_algorithms(True)` or
    `at::Context::setDeterministicAlgorithms(true)`, but this operation is not
    deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable
    deterministic behavior in this case, you must set an environment variable
    before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
    CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


    Set these ENV variables and it works with the above recipe

    bash:
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export CUBLAS_WORKSPACE_CONFIG=:16:8

    """
    from os import environ

    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def batch_to_device(batch, device="cuda"):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def tensor_dict_to_json(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, Tensor):
            v = v.tolist()
        elif isinstance(v, dict):
            v = tensor_dict_to_json(v)
        new_d[k] = v
    return new_d


################################################
# Voice Activity
################################################
def get_dialog_states(vad) -> torch.Tensor:
    """Vad to the full state of a 2 person vad dialog
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    assert vad.ndim >= 1
    return (2 * vad[..., 1] - vad[..., 0]).long() + 1


def get_vad_list_subset(
    vad_list: VAD_LIST, start_time: float, end_time: float
) -> List[List[List[float]]]:
    duration = end_time - start_time

    subset = [[], []]
    for ch, vv in enumerate(vad_list):
        for s, e in vv:
            if e < start_time:
                continue
            if s > end_time:
                break
            rel_start = round(s - start_time, 2)
            rel_end = round(e - start_time, 2)
            if start_time <= s and e <= end_time:
                subset[ch].append([rel_start, rel_end])
            elif s <= start_time and e < end_time:
                # start before region but end included
                subset[ch].append([0, rel_end])
            elif s <= start_time and e >= end_time:
                # Start before and ends after
                subset[ch].append([0, duration])
            elif s < end_time and e >= end_time:
                # start in region but end after
                subset[ch].append([rel_start, duration])

    return subset


def vad_list_to_onehot(
    vad_list: VAD_LIST,
    duration: float,
    hop_time: float = 0,
    frame_hz: float = 0,
    channel_first: bool = False,
) -> Tensor:
    assert (
        hop_time > 0 or frame_hz > 0
    ), "vad_list_to_onehot requires `frame_hz` or `hop_time`"

    if frame_hz > 0:
        hop_time = 1 / frame_hz

    n_frames = time_to_frames(duration, hop_time)
    vad_tensor = torch.zeros((n_frames, 2))
    for ch, ch_vad in enumerate(vad_list):
        for v in ch_vad:
            s = time_to_frames(v[0], hop_time)
            e = time_to_frames(v[1], hop_time)
            vad_tensor[s:e, ch] = 1.0

    if channel_first:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


def vad_onehot_to_vad_list(
    vad: Tensor,
    frame_hz: int = 50,
    ipu_thresh_time: float = 0.1,
) -> List[VAD_LIST]:
    assert (
        vad.ndim == 3
    ), f"Expects vad with batch-dim of shape (B, n_frames, 2) but got {vad.shape}"

    batch_vad_list = []
    for b in range(vad.shape[0]):
        vad_list = []
        for ch in range(2):
            idx, dur, val = find_island_idx_len(vad[b, :, ch])
            active = idx[val == 1]
            start_times = active / frame_hz
            ch_vad_list = []
            if len(start_times) == 0:
                vad_list.append(ch_vad_list)
                continue

            active_dur = dur[val == 1]
            dur_times = active_dur / frame_hz
            end_times = start_times + dur_times
            start_times = start_times.tolist()
            end_times = end_times.tolist()

            s, last_end = round(start_times[0], 2), round(end_times[0], 2)
            ch_vad_list.append([s, last_end])
            for s, e in zip(start_times[1:], end_times[1:]):
                s, e = round(s, 2), round(e, 2)
                if s - last_end < ipu_thresh_time:
                    ch_vad_list[-1][-1] = e
                else:
                    ch_vad_list.append([s, e])
                last_end = e
            vad_list.append(ch_vad_list)
        batch_vad_list.append(vad_list)
    return batch_vad_list


def vad_fill_silences(
    vad: Tensor, max_fill_time: float = 0.02, frame_hz: float = 50
) -> Tensor:
    assert vad.ndim == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    assert vad.shape[-1] == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    max_fill_frame = round(max_fill_time * frame_hz)
    for ch in range(2):
        starts, dur, on_off = find_island_idx_len(vad[:, ch])
        sil_starts = starts[on_off == 0]
        sil_durs = dur[on_off == 0]
        w = torch.where(sil_durs <= max_fill_frame)[0]
        fill_starts = sil_starts[w]
        fill_durs = sil_durs[w]
        for s, d in zip(fill_starts, fill_durs):
            vad[s : s + d, ch] = 1.0
    return vad


def vad_omit_spikes(
    vad: Tensor, max_omit_time: float = 0.02, frame_hz: float = 50
) -> Tensor:
    assert vad.ndim == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    assert vad.shape[-1] == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    max_omit_frame = round(max_omit_time * frame_hz)
    for ch in range(2):
        starts, dur, on_off = find_island_idx_len(vad[:, ch])
        sil_starts = starts[on_off == 1]
        sil_durs = dur[on_off == 1]
        w = torch.where(sil_durs <= max_omit_frame)[0]
        fill_starts = sil_starts[w]
        fill_durs = sil_durs[w]
        for s, d in zip(fill_starts, fill_durs):
            vad[s : s + d, ch] = 0.0
    return vad


################################################
# File system
################################################
def repo_root():
    """
    Returns the absolute path to the git repository
    """
    root = dirname(__file__)
    root = dirname(root)
    return root


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False)


def read_json(path, encoding="utf8"):
    with open(path, "r", encoding=encoding) as f:
        data = json.loads(f.read())
    return data


def write_txt(txt, name):
    """
    Argument:
        txt:    list of strings
        name:   filename
    """
    with open(name, "w") as f:
        f.write("\n".join(txt))


def read_txt(path, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data
