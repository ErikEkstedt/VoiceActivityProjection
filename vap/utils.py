import torch
import json
from os.path import dirname

from typing import List, Optional, Tuple

from vap.audio import time_to_frames, load_waveform
from vap_turn_taking.utils import vad_list_to_onehot


def load_sample(
    path: str,
    vad_list_path: Optional[str] = None,
    sample_rate: int = 16000,
    frame_hz=50,
    noise_scale: float = 0.005,
    force_stereo: bool = True,
    device: Optional[str] = None,
):
    waveform, _ = load_waveform(path, sample_rate=sample_rate)

    # Add channel with minimal noise as second channel
    if force_stereo and waveform.ndim == 2 and waveform.shape[0] == 1:
        z = torch.randn_like(waveform) * noise_scale
        # waveform = torch.stack((waveform, z), dim=1)
        waveform = torch.stack((waveform, z), dim=1)

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    vad = None
    if vad_list_path is not None:
        vad_list = read_json(vad_list_path)
        duration = waveform.shape[-1] / sample_rate
        vad = vad_list_to_onehot(
            vad_list, hop_time=int(1 / frame_hz), duration=duration, channel_last=True
        )

    if device is not None:
        waveform = waveform.to(device)
        if vad is not None:
            vad = vad.to(device)

    return waveform, vad


def find_island_idx_len(x):
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


def vad_output_to_vad_list(
    vad: torch.Tensor,
    frame_hz: int,
    vad_thresh: float = 0.5,
    ipu_thresh_time: float = 0.1,
):
    assert (
        vad.ndim == 3
    ), f"Expects vad with batch-dim of shape (B, n_frames, 2) but got {vad.shape}"

    # Threshold if probabilities (sigmoided logits)
    v = (vad >= vad_thresh).float()

    batch_vad_list = []
    for b in range(vad.shape[0]):
        vad_list = []
        for ch in range(2):
            idx, dur, val = find_island_idx_len(v[b, :, ch])
            active = idx[val == 1]
            active_dur = dur[val == 1]
            start_times = active / frame_hz
            dur_times = active_dur / frame_hz
            end_times = start_times + dur_times
            start_times = start_times.tolist()
            end_times = end_times.tolist()
            ch_vad_list = []
            if len(start_times) == 0:
                vad_list.append(ch_vad_list)
                continue
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


def repo_root():
    """
    Returns the absolute path to the git repository
    """
    root = dirname(__file__)
    root = dirname(root)
    return root


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


def vad_list_to_onehot(
    vad_list: List[Tuple[float, float]],
    hop_time: float,
    duration: float,
    channel_last: bool = False,
) -> torch.Tensor:
    n_frames = time_to_frames(duration, hop_time) + 1

    if isinstance(vad_list[0][0], list):
        vad_tensor = torch.zeros((len(vad_list), n_frames))
        for ch, ch_vad in enumerate(vad_list):
            for v in ch_vad:
                s = time_to_frames(v[0], hop_time)
                e = time_to_frames(v[1], hop_time)
                vad_tensor[ch, s:e] = 1.0
    else:
        vad_tensor = torch.zeros((1, n_frames))
        for v in vad_list:
            s = time_to_frames(v[0], hop_time)
            e = time_to_frames(v[1], hop_time)
            vad_tensor[:, s:e] = 1.0

    if channel_last:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


def load_vad_list(
    path: str, frame_hz: int = 50, duration: Optional[float] = None
) -> torch.Tensor:
    vad_hop_time = 1.0 / frame_hz
    vad_list = read_json(path)

    last_vad = -1
    for vad_channel in vad_list:
        if len(vad_channel) > 0:
            if vad_channel[-1][-1] > last_vad:
                last_vad = vad_channel[-1][-1]

    ##############################################
    # VAD-frame of relevant part
    ##############################################
    all_vad_frames = vad_list_to_onehot(
        vad_list,
        hop_time=vad_hop_time,
        duration=duration if duration is not None else last_vad,
        channel_last=True,
    )

    return all_vad_frames


def batch_to_device(batch, device="cuda"):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def tensor_dict_to_json(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.tolist()
        elif isinstance(v, dict):
            v = tensor_dict_to_json(v)
        new_d[k] = v
    return new_d


def load_hydra_conf(config_path="conf", config_name="config"):
    """https://stackoverflow.com/a/61169706"""
    from hydra import compose, initialize

    try:
        initialize(version_base=None, config_path=config_path)
    except:
        pass

    cfg = compose(config_name=config_name)
    return cfg
