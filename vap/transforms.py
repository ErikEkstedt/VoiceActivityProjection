import torch
import torch.nn as nn
import einops
from typing import Optional

import vap.functional as VF
from vap.encoder import CConv1d


def _check_waveform_and_vad_shape(
    waveform: torch.Tensor, vad: Optional[torch.Tensor] = None
) -> None:
    assert (
        waveform.ndim == 3
    ), f"Must provide input waveform of shape (B, 1, n_samples) or stereo (B, 2, n_samples). Got {tuple(waveform.shape)}"
    assert (
        waveform.shape[1] == 1 or waveform.shape[1] == 2
    ), f"Must provide input waveform of shape (B, 1, n_samples) or stereo (B, 2, n_samples). Got {tuple(waveform.shape)}"

    if vad is not None:
        assert (
            vad.ndim == 3
        ), f"Vad must be of shape (B, n_frames, 2). Got {tuple(vad.shape)}"
        assert (
            vad.shape[-1] == 2
        ), f"Vad must be of shape (B, n_frames, 2). Got {tuple(vad.shape)}"
    return None


class LowPass(nn.Module):
    def __init__(
        self,
        cutoff_freq: int = 400,
        sample_rate: int = VF.SAMPLE_RATE,
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate

    def forward(
        self, waveform: torch.Tensor, vad: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _check_waveform_and_vad_shape(waveform, vad)
        B, NC, N_SAMPLES = waveform.shape
        x = VF.low_pass_filter_resample(waveform, self.cutoff_freq, self.sample_rate)
        if x.shape[-1] > N_SAMPLES:
            x = x[..., :N_SAMPLES]
        elif x.shape[-1] < N_SAMPLES:
            diff = N_SAMPLES - x.shape[-1]
            x = torch.cat((x, torch.ones((B, NC, diff))), dim=-1)
        return x


# TODO: add check for empty vad -> omit
class FlatPitch(nn.Module):
    def __init__(
        self,
        target_f0: int = -1,
        sample_rate: int = 16000,
        hop_time: float = 0.01,
    ):
        super().__init__()
        self.target_f0 = target_f0
        self.sample_rate = sample_rate
        self.hop_time = hop_time

    def forward(
        self, waveform: torch.Tensor, vad: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        waveform: mono (B, 1, n_samples) or stereo (B, 2, n_samples)
        """
        _check_waveform_and_vad_shape(waveform, vad)

        B, NC, _ = waveform.shape
        x = torch.zeros_like(waveform)
        # Extract mean over each sample
        for b in range(B):
            for channel in range(NC):
                f0 = VF.pitch_praat(waveform[b, channel], sample_rate=self.sample_rate)
                m, _, _ = VF.f0_statistics(f0)
                x[b, channel] = VF.pitch_praat_flatten(
                    waveform[b, channel],
                    target_f0=m,
                    hop_time=self.hop_time,
                    sample_rate=self.sample_rate,
                )
        return x


class ShiftPitch(nn.Module):
    def __init__(
        self,
        factor: float = 0.9,
        sample_rate: int = VF.SAMPLE_RATE,
        hop_time: float = VF.HOP_TIME,
        f0_min: int = VF.F0_MIN,
        f0_max: int = VF.F0_MAX,
    ):
        super().__init__()
        self.factor = factor
        self.sample_rate = sample_rate
        self.hop_time = hop_time
        self.f0_min = f0_min
        self.f0_max = f0_max

    def forward(
        self, waveform: torch.Tensor, vad: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _check_waveform_and_vad_shape(waveform, vad)

        B, NC, _ = waveform.shape
        x = torch.zeros_like(waveform)

        # Extract mean over each sample
        for b in range(B):
            for channel in range(NC):
                x[b, channel] = VF.pitch_praat_shift(
                    waveform[b, channel],
                    factor=self.factor,
                    hop_time=self.hop_time,
                    sample_rate=self.sample_rate,
                    f0_min=self.f0_min,
                    f0_max=self.f0_max,
                )
        return x
