import pytest
import torch

from vap.audio import load_waveform
import vap.transforms as VT
import vap.functional as VF

BATCH_SIZE = 4
SAMPLE_RATE = 16_000


@pytest.fixture
def waveform():
    x, _ = load_waveform(
        "example/student_long_female_en-US-Wavenet-G.wav", sample_rate=SAMPLE_RATE
    )
    x = torch.stack([x] * BATCH_SIZE)
    return x


@pytest.mark.transforms
@pytest.mark.pitch
def test_Flat_Pitch(waveform):
    orig_f0 = VF.pitch_praat(waveform, sample_rate=SAMPLE_RATE)
    orig_means, _, _ = VF.f0_statistics(orig_f0)

    B, N, N_SAMPLES = waveform.shape
    transform = VT.FlatPitch(sample_rate=SAMPLE_RATE)
    x = transform(waveform)
    assert x.shape[0] == B, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[1] == N, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[2] == N_SAMPLES, f"Wrong sample size {tuple(x.shape)} != {(B, N)}"

    x_f0 = VF.pitch_praat(x, sample_rate=SAMPLE_RATE)
    x_means, x_std, _ = VF.f0_statistics(x_f0)

    max_mean_diff = 2
    max_std = 2
    assert (
        x_means - orig_means
    ).abs().max() < max_mean_diff, (
        f"Transformed pitch means differ by more than {max_mean_diff}Hz"
    )
    assert (
        x_std
    ).abs().max() < max_std, f"Transformed pitch std are greater than {max_std}Hz"


@pytest.mark.transforms
@pytest.mark.pitch
def test_Shift_Pitch(waveform):
    hz_max_deviation = 2
    hz_max_std_deviation = 5
    factor = 0.9

    orig_f0 = VF.pitch_praat(waveform, sample_rate=SAMPLE_RATE)
    orig_means, orig_std, _ = VF.f0_statistics(orig_f0)
    calculated_new_mean = factor * orig_means

    B, N, N_SAMPLES = waveform.shape
    transform = VT.ShiftPitch(factor=0.9, sample_rate=SAMPLE_RATE)
    x = transform(waveform)
    assert x.shape[0] == B, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[1] == N, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[2] == N_SAMPLES, f"Wrong sample size {tuple(x.shape)} != {(B, N)}"

    x_f0 = VF.pitch_praat(x, sample_rate=SAMPLE_RATE)
    x_mean, x_std, _ = VF.f0_statistics(x_f0)

    assert (
        x_mean - calculated_new_mean
    ).abs().max() < hz_max_deviation, f"Pitch shift mean {x_mean.item()} differed more than {hz_max_deviation}Hz from calculated mean {calculated_new_mean}"
    assert (
        x_std - orig_std
    ).abs().max() < hz_max_std_deviation, f"The shifted std {x_std.item():.2f} differs more than {hz_max_std_deviation}Hz from original std {orig_std.item():.2f}"


@pytest.mark.transforms
def test_Flat_Intensity(waveform):
    B, N, N_SAMPLES = waveform.shape
    transform = VT.FlatIntensity(sample_rate=SAMPLE_RATE)
    x = transform(waveform)
    assert x.shape[0] == B, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[1] == N, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[2] == N_SAMPLES, f"Wrong sample size {tuple(x.shape)} != {(B, N)}"
    assert (
        x.std().max() < waveform.std().max()
    ), f"Standard deviation not lower after intensity flattening!"


@pytest.mark.transforms
def test_low_pass(waveform):
    B, N, N_SAMPLES = waveform.shape
    transform = VT.LowPass(cutoff_freq=400, sample_rate=SAMPLE_RATE)
    x = transform(waveform)
    assert x.shape[0] == B, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[1] == N, f"Wrong batch size {tuple(x.shape)} != {(B, N)}"
    assert x.shape[2] == N_SAMPLES, f"Wrong sample size {tuple(x.shape)} != {(B, N)}"
