import pytest

from vap.audio import load_waveform
import vap.functional as VF

SAMPLE_RATE = 16_000


@pytest.fixture
def waveform():
    x, _ = load_waveform(
        "example/student_long_female_en-US-Wavenet-G.wav", sample_rate=SAMPLE_RATE
    )
    return x


@pytest.mark.functional
@pytest.mark.pitch
def test_pitch_praat_flat(waveform):
    f0 = VF.pitch_praat(waveform, sample_rate=SAMPLE_RATE)
    mean, _, _ = VF.f0_statistics(f0)

    waveform_flat = VF.pitch_praat_flatten(
        waveform, target_f0=mean, sample_rate=SAMPLE_RATE
    )

    f0_flat = VF.pitch_praat(waveform_flat, sample_rate=SAMPLE_RATE)
    flat_mean, flat_std, _ = VF.f0_statistics(f0_flat)

    hz_max_deviation = 2
    assert (
        flat_mean - mean
    ).abs() < hz_max_deviation, (
        f"Pitch flat average differed more than {hz_max_deviation}Hz"
    )
    assert (
        flat_std < hz_max_deviation
    ), f"The standard deviation of flat pitch exceeded {hz_max_deviation}Hz"


@pytest.mark.functional
@pytest.mark.pitch
def test_pitch_shift(waveform):
    hz_max_deviation = 2
    hz_max_std_deviation = 5
    factor = 0.9

    f0 = VF.pitch_praat(waveform, sample_rate=SAMPLE_RATE)
    mean, std, _ = VF.f0_statistics(f0)
    calculated_new_mean = factor * mean
    waveform_shift = VF.pitch_praat_shift(
        waveform, factor=factor, sample_rate=SAMPLE_RATE
    )
    f0_shift = VF.pitch_praat(waveform_shift, sample_rate=SAMPLE_RATE)
    shift_mean, shift_std, _ = VF.f0_statistics(f0_shift)

    assert (
        shift_mean - calculated_new_mean
    ).abs() < hz_max_deviation, f"Pitch shift mean {shift_mean.item()} differed more than {hz_max_deviation}Hz from calculated mean {calculated_new_mean}"
    assert (
        shift_std - std
    ).abs() < hz_max_std_deviation, f"The shifted std {shift_std.item():.2f} differs more than {hz_max_std_deviation}Hz from original std {std.item():.2f}"


@pytest.mark.functional
@pytest.mark.intensity
def test_intensity_praat(waveform):
    _ = VF.intensity_praat(waveform, sample_rate=SAMPLE_RATE)
