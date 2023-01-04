import torch
from torch import Tensor
import torchaudio.functional as AF
from typing import Union, Tuple, Optional

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
    assert (
        waveform.ndim == 1
    ), "praat functions only takes single samples -> waveform.ndim == 1."
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
    assert (
        waveform.ndim == 1
    ), "praat functions only takes single samples -> waveform.ndim == 1."
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


def intensity_praat_flatten(
    waveform: torch.Tensor,
    target_intensity: float,
    output_intensity: float = 70.0,
    min_intensity: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
    hop_time: float = HOP_TIME,
    f0_min: int = F0_MIN,
):
    assert (
        waveform.ndim == 1
    ), "praat functions only takes single samples -> waveform.ndim == 1."
    sound = torch_to_praat_sound(waveform, sample_rate=sample_rate)
    snd_intensity = sound.to_intensity(
        minimum_pitch=f0_min, time_step=hop_time, subtract_mean=False
    )
    intensities = praat_to_torch(
        sound.to_intensity(
            minimum_pitch=f0_min, time_step=hop_time, subtract_mean=False
        )
    ).squeeze()
    times = [
        snd_intensity.get_time_from_frame_number(f + 1)
        for f in range(snd_intensity.get_number_of_frames())
    ]

    int_tier = call(
        snd_intensity, "Create IntensityTier", "intensity_tier", 0, sound.total_duration
    )

    t = 0
    for t, i in zip(times, intensities):
        if i < min_intensity:
            continue
        scale = target_intensity - i.item()
        call(int_tier, "Add point", t, scale)
    call(int_tier, "Add point", t + 0.001, 0)
    new_sound = call([int_tier, sound], "Multiply", 1)
    call(new_sound, "Scale intensity", output_intensity)
    return praat_to_torch(new_sound)


def load_vad_list(
    path: str, frame_hz: int = 50, duration: Optional[float] = None
) -> Tensor:
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
        frame_hz=frame_hz,
        duration=duration if duration is not None else last_vad,
    )

    return all_vad_frames


if __name__ == "__main__":
    import sounddevice as sd
    import matplotlib.pyplot as plt
    from vap.audio import load_waveform, get_audio_info
    from vap.plot_utils import plot_stereo_mel_spec

    wavpath = "example/student_long_female_en-US-Wavenet-G.wav"
    info = get_audio_info(wavpath)
    waveform, sample_rate = load_waveform(wavpath)
    vad = load_vad_list(
        "example/student_long_female_en-US-Wavenet-G_vad_list.json",
        duration=info["duration"],
    )
    print("waveform: ", tuple(waveform.shape))

    batch_wav = torch.stack([waveform] * 4)
    print("batch_wav: ", tuple(batch_wav.shape))
    f0 = pitch_praat(batch_wav[:, 0], sample_rate)
    print("f0: ", tuple(f0.shape))
    intens = intensity_praat(batch_wav[:, 0], sample_rate)
    print("intens: ", tuple(intens.shape))

    hop_time = 0.01
    f0_min = 60
    f0_max = 600
    sound = torch_to_praat_sound(waveform)
    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)
    pitch_tier = call(manipulation, "Extract pitch tier")

    x = intensity_praat_flatten(waveform, target_intensity=50)

    sd.play(x[0], samplerate=SAMPLE_RATE)

    sd.play(waveform[0], samplerate=SAMPLE_RATE)

    # x = intensity_praat_flatten(waveform)
    # x = pitch_praat_flatten(waveform)
    # x = pitch_praat_shift(waveform)
    # x = FlatIntensity(vad_hz=50)(waveform, vad=vad.unsqueeze(0))
    # x = IntensityNeutralizer(vad_hz=50)(waveform)
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
    ax[0].set_ylim([-1, 1])
    ax[1].plot(wi[0].log())
    ax[2].plot(wx[0].log())
    ax[3].plot(x[0])
    ax[3].set_ylim([-1, 1])
    plt.pause(0.1)
