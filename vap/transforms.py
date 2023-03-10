import torch
import torchaudio.transforms as AT
import torchaudio.functional as AF
import random

"""
* PitchShift sound ok for [-2, 2] steps
"""


class Augmentation(torch.nn.Module):
    def __init__(
        self,
        probability: float = 0.5,
        noise_amplitude: float = 0.01,
        pitch_steps: list[int] = [-2, -1, 1, 2],
        freq_mask_param: int = 100,
        iid_masks: bool = True,
        sample_rate: int = 16_000,
        device: str = "cpu",
    ):
        super().__init__()
        self.probability = probability
        self.sample_rate = sample_rate
        self.pitch_steps = pitch_steps
        self.noise_amplitude = noise_amplitude
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks

        self.shift_pitch = PitchShift(
            pitch_steps=self.pitch_steps, sample_rate=sample_rate
        )
        self.frequency_masking = WaveformFrequencyMasking(
            freq_mask_param=freq_mask_param,
            iid_masks=iid_masks,
            sample_rate=sample_rate,
        )
        self.noise = AddGaussianNoise(max_amplitude=noise_amplitude)

        if device != "cpu":
            self.to(device)

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += f"\tnoise_amplitude={self.noise_amplitude},\n"
        s += f"\tpitch_steps={self.pitch_steps},\n"
        s += f"\tfreq_mask_param={self.freq_mask_param},\n"
        s += f"\tiid_masks={self.iid_masks},\n"
        s += f"\tsample_rate={self.sample_rate},\n"
        s += ")\n"
        return s

    def apply_all(self, x):
        x = self.shift_pitch(x)
        x = self.frequency_masking(x)
        return self.noise(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > self.probability:
            return x

        r = torch.rand(1)
        if r < 0.25:
            x = self.shift_pitch(x)
        elif 0.25 < r < 0.50:
            x = self.noise(x)
        elif 0.5 < r < 0.75:
            x = self.frequency_masking(x)
        else:
            x = self.apply_all(x)
        return x


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, max_amplitude=0.01):
        """
        :param min_amplitude: Minimum noise amplification factor
        :param max_amplitude: Maximum noise amplification factor
        :param p:
        """
        super().__init__()
        assert max_amplitude > 0.0
        self.max_amplitude = max_amplitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        noise -= noise.min()
        noise = 2 * self.max_amplitude * noise / noise.max()
        noise -= noise.max() / 2
        return x + noise


class PitchShift(torch.nn.Module):
    def __init__(
        self, pitch_steps: list[int] = [-2, -1, 1, 2], sample_rate: int = 16_000
    ):
        super().__init__()
        self.pitch_steps = pitch_steps
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cuda don't allow for use_deterministic_algorithms(True) so we untoggle it here"""
        p = random.choice(self.pitch_steps)
        torch.use_deterministic_algorithms(False)
        x = AF.pitch_shift(x, sample_rate=self.sample_rate, n_steps=p)
        torch.use_deterministic_algorithms(True)
        return x


class WaveformFrequencyMasking(torch.nn.Module):
    def __init__(
        self,
        window_time: float = 0.05,
        hop_time: float = 0.02,
        freq_mask_param: int = 100,
        iid_masks: bool = True,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = int(window_time * sample_rate)
        self.hop_length = int(hop_time * sample_rate)
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks

        self.to_spectrogram = AT.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None
        )
        self.to_waveform = AT.InverseSpectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.frequency_masking = AT.FrequencyMasking(freq_mask_param, iid_masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.to_spectrogram(x)  # (B, C, Freq, Time)
        spec_fm = self.frequency_masking(spec.real)
        spec.real = spec_fm
        return self.to_waveform(spec)


if __name__ == "__main__":

    import torch
    import matplotlib.pyplot as plt
    import sounddevice as sd
    from vap.plot_utils import plot_waveform, plot_stereo_mel_spec
    from vap.train import DataConfig
    from vap_dataset.datamodule import VapDataModule

    sample_rate = 16_000
    dconf = DataConfig()
    dm = VapDataModule(
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        horizon=2,
        batch_size=4,
        num_workers=1,
    )
    dm.prepare_data()
    dm.setup()
    print(dm)

    batch = next(iter(dm.train_dataloader()))

    aug = Augmentation(device="cuda")

    aug.to("cuda")

    # power=None returns complex spectrogram of same shape but dtype=torch.complex64
    b = 1
    hop_length = int(0.01 * sample_rate)
    n_fft = int(0.02 * sample_rate)
    WM = WaveformFrequencyMasking()
    N = AddGaussianNoise()
    aug = Augmentation().to("cuda")

    x = batch["waveform"].to("cuda")

    with torch.no_grad():
        x_aug = aug.apply_all(x)

    fig, ax = plt.subplots(4, 1, sharex=True)
    plot_stereo_mel_spec(x[b].cpu(), ax=[ax[0], ax[2]])
    plot_stereo_mel_spec(x_aug[b].cpu(), ax=[ax[1], ax[3]])
    plt.show()

    plot_mel
    plot_f0(x[b, 0], ax=ax[1])
    plot_f0(x[b, 1], ax=ax[1], color="orange")
    plot_waveform(x_shift[b, 1], ax=ax[2], color="orange")
    plot_f0(x_shift[b, 0], ax=ax[3], color="b")
    plot_f0(x_shift[b, 1], ax=ax[3], color="orange")
    plt.show()

    sd.play(x[b].t(), samplerate=sample_rate)

    sd.play(x_noise[b].t(), samplerate=sample_rate)
