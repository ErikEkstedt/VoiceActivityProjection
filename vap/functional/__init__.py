import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    raise ImportError(
        "Missing dependency 'praat-parselmouth'. Please install ('pip install praat-parselmouth') if you require praat-based augmentations."
    )

"""
* Praat: https://www.fon.hum.uva.nl/praat/
* Flatten pitch script: http://phonetics.linguistics.ucla.edu/facilities/acoustic/FlatIntonationSynthesizer.txt
* Parselmouth: https://parselmouth.readthedocs.io/en/latest/examples/pitch_manipulation.html
"""

# Defaults
SAMPLE_RATE: int = 16_000
F0_MIN: int = 60
F0_MAX: int = 500
HOP_TIME: float = 0.01  # 10ms


def torch_to_praat_sound(
    x: torch.Tensor, sample_rate: int = SAMPLE_RATE
) -> parselmouth.Sound:
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy().astype("float64")
    return parselmouth.Sound(x, sampling_frequency=sample_rate)


def praat_to_torch(sound: parselmouth.Sound) -> torch.Tensor:
    y = sound.as_array().astype("float32")
    return torch.from_numpy(y)


def f0_statistics(f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if f0.ndim > 3:
        raise NotImplementedError(f"f1_statistics not implemented for tensor.ndim > 3.")

    if f0.ndim == 1:
        f_non_zero = f0[f0 != 0.0]
        median = f_non_zero.median()
        mean = f_non_zero.mean()
        std = f_non_zero.std()
    elif f0.ndim == 2:
        median, mean, std = [], [], []
        for f0_tmp in f0:
            f_non_zero = f0_tmp[f0_tmp != 0.0]
            median.append(f_non_zero.median())
            mean.append(f_non_zero.mean())
            std.append(f_non_zero.std())
        median = torch.stack(median)
        mean = torch.stack(mean)
        std = torch.stack(std)
    else:
        median, mean, std = [], [], []
        for b in range(f0.shape[0]):
            cmedian, cmean, cstd = [], [], []
            for ch in range(f0.shape[1]):
                f0_tmp = f0[b, ch]
                f_non_zero = f0_tmp[f0_tmp != 0.0]
                cmedian.append(f_non_zero.median())
                cmean.append(f_non_zero.mean())
                cstd.append(f_non_zero.std())
            median.append(torch.stack(cmedian))
            mean.append(torch.stack(cmean))
            std.append(torch.stack(cstd))
        median = torch.stack(median)
        mean = torch.stack(mean)
        std = torch.stack(std)

    return mean, std, median


def pitch_praat(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    f0_max: int = F0_MAX,
) -> torch.Tensor:
    device = waveform.device

    def _single_pitch(waveform):
        sound = torch_to_praat_sound(waveform, sample_rate)
        pitch = sound.to_pitch(
            time_step=hop_time, pitch_floor=f0_min, pitch_ceiling=f0_max
        )
        return torch.from_numpy(pitch.selected_array["frequency"]).float().to(device)

    if waveform.ndim == 1:
        f0 = _single_pitch(waveform)
    elif waveform.ndim == 2:
        f0s = []
        for n in range(waveform.shape[0]):
            f0s.append(_single_pitch(waveform[n]))
        f0 = torch.stack(f0s).to(device)
    else:
        f0s = []
        for n in range(waveform.shape[0]):
            chs = []
            for ch in range(waveform.shape[1]):
                chs.append(_single_pitch(waveform[n, ch]))
            f0s.append(torch.stack(chs))
        f0 = torch.stack(f0s).to(device)
    return f0


def intensity(
    x: Tensor,
    frame_time: float = 0.05,
    hop_time: float = HOP_TIME,
    top_db: float = 80,
    sample_rate: int = SAMPLE_RATE,
) -> Tensor:
    frame_length = int(frame_time * sample_rate)
    hop_length = int(hop_time * sample_rate)
    xx = F.pad(x, (0, frame_length - 1))
    # h = x.unfold(dimension=-1, size=frame_length, step=hop_length)
    h = xx.unfold(dimension=-1, size=frame_length, step=hop_length)
    hm, _ = h.abs().max(-1)
    return AT.AmplitudeToDB(stype="amplitude", top_db=top_db)(hm) + top_db


def intensity_praat(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    subtract_mean: bool = False,
) -> torch.Tensor:
    device = waveform.device

    def _single_intensity(waveform):
        sound = torch_to_praat_sound(waveform, sample_rate=sample_rate)
        intensity = sound.to_intensity(
            minimum_pitch=f0_min, time_step=hop_time, subtract_mean=subtract_mean
        )
        return praat_to_torch(intensity)[0]

    if waveform.ndim == 1:
        intensity = _single_intensity(waveform)
    elif waveform.ndim == 2:
        intensities = []
        for n in range(waveform.shape[0]):
            intensities.append(_single_intensity(waveform[n]))
        intensity = torch.stack(intensities).to(device)
    else:
        intensities = []
        for n in range(waveform.shape[0]):
            chs = []
            for ch in range(waveform.shape[1]):
                chs.append(_single_intensity(waveform[n, ch]))
            intensities.append(torch.stack(chs))
        intensity = torch.stack(intensities).to(device)
    return intensity


if __name__ == "__main__":

    p = pitch_praat(x, sample_rate=16_000)
