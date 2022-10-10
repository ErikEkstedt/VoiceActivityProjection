import torch
import torchaudio.functional as AF
from typing import Union, Optional, Tuple

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    raise ImportError(
        "Missing dependency 'praat-parselmouth'. Please install ('pip install praat-parselmouth') if you require praat-based augmentations."
    )

from vap.encoder import CConv1d

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


def f0_statistics(f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def compute_kaldi_pitch(
    y: torch.Tensor,
    fmin: int = F0_MIN,
    fmax: int = F0_MAX,
    hop_time: float = HOP_TIME,
    frame_time: float = 0.025,
    sample_rate: int = SAMPLE_RATE,
    **kwargs,
) -> torch.Tensor:
    f0 = AF.compute_kaldi_pitch(
        y,
        sample_rate=sample_rate,
        frame_length=int(1000 * frame_time),
        frame_shift=int(1000 * hop_time),
        min_f0=fmin,
        max_f0=fmax,
        **kwargs,
    )
    return f0[..., 1]


def pitch_praat(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    f0_max: int = F0_MAX,
) -> torch.Tensor:
    device = waveform.device

    if waveform.ndim == 1:
        sound = torch_to_praat_sound(waveform, sample_rate)
        pitch = sound.to_pitch(
            time_step=hop_time, pitch_floor=f0_min, pitch_ceiling=f0_max
        )
        f0 = torch.from_numpy(pitch.selected_array["frequency"]).float().to(device)
    elif waveform.ndim == 2:
        f0s = []
        for n in range(waveform.shape[0]):
            sound = torch_to_praat_sound(waveform[n], sample_rate)
            pitch = sound.to_pitch(
                time_step=hop_time, pitch_floor=f0_min, pitch_ceiling=f0_max
            )
            f0s.append(torch.from_numpy(pitch.selected_array["frequency"]).float())
        f0 = torch.stack(f0s).to(device)
    else:
        f0s = []
        for n in range(waveform.shape[0]):
            chs = []
            for ch in range(waveform.shape[1]):
                sound = torch_to_praat_sound(waveform[n, ch], sample_rate)
                pitch = sound.to_pitch(
                    time_step=hop_time, pitch_floor=f0_min, pitch_ceiling=f0_max
                )
                chs.append(torch.from_numpy(pitch.selected_array["frequency"]).float())
            f0s.append(torch.stack(chs))
        f0 = torch.stack(f0s).to(device)
    return f0


def intensity_praat(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    subtract_mean: bool = False,
) -> torch.Tensor:
    sound = torch_to_praat_sound(waveform, sample_rate=sample_rate)
    intensity = sound.to_intensity(
        minimum_pitch=f0_min, time_step=hop_time, subtract_mean=subtract_mean
    )
    return praat_to_torch(intensity)


def pitch_praat_flatten(
    waveform: torch.Tensor,
    target_f0: Union[float, torch.Tensor] = 200.0,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    f0_max: int = F0_MAX,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """

    Inspiration:
        http://phonetics.linguistics.ucla.edu/facilities/acoustic/FlatIntonationSynthesizer.txt
    """
    device = waveform.device
    if isinstance(target_f0, torch.Tensor):
        target_f0 = target_f0.item()

    # convert to sound (Only on cpu)
    sound = torch_to_praat_sound(waveform, sample_rate)
    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)
    # Create flat pitch tier
    pitch_tier = call(
        manipulation, "Create PitchTier", "flat", sound.start_time, sound.end_time
    )
    # Add points att start and end time (covering entire waveform)
    call(pitch_tier, "Add point", sound.start_time, target_f0)
    call(pitch_tier, "Add point", sound.end_time, target_f0)
    # Select the original and the replacement tier -> replace pitch
    call([pitch_tier, manipulation], "Replace pitch tier")
    # Extract the new sound
    sound_flat = call(manipulation, "Get resynthesis (overlap-add)")

    # From sound -> torch.Tensor
    x = praat_to_torch(sound_flat)
    x = x.to(device)  # reset to original device
    return x


def pitch_praat_shift(
    waveform: torch.Tensor,
    factor: float = 0.95,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
    f0_max: int = F0_MAX,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    device = waveform.device
    sound = torch_to_praat_sound(waveform, sample_rate)

    # Source: https://parselmouth.readthedocs.io/en/latest/examples/pitch_manipulation.html
    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)
    pitch_tier = call(manipulation, "Extract pitch tier")

    # Should not Multiply but simply shift NOT scale
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_shifted = call(manipulation, "Get resynthesis (overlap-add)")

    # From sound -> torch.Tensor
    x = praat_to_torch(sound_shifted)
    x = x.to(device)  # reset to original device
    return x


def low_pass_filter_resample(
    waveform: torch.Tensor, cutoff_freq: int, sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    new_freq = int(cutoff_freq * 2)  # Nyquist
    x = AF.resample(waveform, orig_freq=sample_rate, new_freq=new_freq)
    x = AF.resample(x, orig_freq=new_freq, new_freq=sample_rate)
    return x


if __name__ == "__main__":
    import sounddevice as sd
    import matplotlib.pyplot as plt
    from vap.audio import load_waveform, get_audio_info
    from vap.plot_utils import plot_stereo_mel_spec
    from vap.utils import load_vad_list

    wavpath = "example/student_long_female_en-US-Wavenet-G.wav"
    info = get_audio_info(wavpath)

    waveform, sample_rate = load_waveform(wavpath)
    vad = load_vad_list(
        "example/student_long_female_en-US-Wavenet-G_vad_list.json",
        duration=info["duration"],
    )
    print("waveform: ", tuple(waveform.shape))

    # x = intensity_praat_flatten(waveform)
    # x = pitch_praat_flatten(waveform)
    # x = pitch_praat_shift(waveform)
    x = FlatIntensity(vad_hz=50)(waveform, vad=vad.unsqueeze(0))
    print("x: ", tuple(x.shape))

    wi = intensity_praat(waveform, sample_rate, subtract_mean=True)
    # wi = 79*wi/wi.max()
    wi[wi < 0] = 0
    wx = intensity_praat(x, sample_rate, subtract_mean=True)
    # wx = 79*wx/wx.max()
    wx[wx < 0] = 0

    fig, ax = plt.subplots(2, 1)
    _ = plot_stereo_mel_spec(torch.cat((waveform, x)), ax=ax, plot=True)
    plt.pause(0.1)

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(waveform[0])
    ax[1].plot(wi[0].log())
    ax[2].plot(wx[0].log())
    ax[3].plot(x[0])
    plt.pause(0.1)
