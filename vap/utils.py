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


def read_txt(path, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


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
