import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Union, List, Tuple, Iterable


from vap.utils.audio import log_mel_spectrogram


AX = Union[Axes, Iterable[Axes]]


def plot_melspectrogram(
    waveform: torch.Tensor,
    ax: AX,
    sample_rate: int = 16_000,
    hop_time: float = 0.02,
    frame_time: float = 0.05,
    n_mels: int = 80,
    fontsize: int = 12,
    plot: bool = False,
    return_spec: bool = False,
) -> Union[AX, Tuple[AX, torch.Tensor]]:
    assert waveform.ndim == 2, f"Expected (N_CHANNELS, N_SAMPLES) got {waveform.shape}"

    if type(ax) == Axes:
        ax = [ax]

    duration = waveform.shape[-1] / sample_rate
    xmin, xmax = 0, duration
    ymin, ymax = 0, 80

    hop_length = round(sample_rate * hop_time)
    frame_length = round(sample_rate * frame_time)
    mel_spec = log_mel_spectrogram(
        waveform,
        n_mels=n_mels,
        n_fft=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )

    ax[0].imshow(
        mel_spec[0],
        interpolation="none",
        aspect="auto",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
    )
    ax[0].set_yticks([])
    ax[0].set_yticks([])
    ax[0].set_ylabel("A", fontsize=fontsize)

    if mel_spec.shape[0] > 1 and len(ax) > 1:
        ax[1].imshow(
            mel_spec[1],
            interpolation="none",
            aspect="auto",
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
        )
        ax[1].set_yticks([])
        ax[1].set_yticks([])
        ax[1].set_ylabel("B", fontsize=fontsize)
    if plot:
        plt.pause(0.01)
    if return_spec:
        return ax, mel_spec
    return ax


def plot_waveform(
    waveform,
    ax: AX,
    color: str = "lightblue",
    alpha: float = 0.6,
    label: Optional[str] = None,
    downsample: int = 10,
    sample_rate: int = 16000,
) -> Axes:
    assert (
        waveform.ndim == 1
    ), f"Expects a single channel waveform (n_samples, ) got {waveform.shape}"
    x = waveform[..., ::downsample]

    new_rate = sample_rate / downsample
    x_time = torch.arange(x.shape[-1]) / new_rate

    ax.plot(x_time, x, color=color, zorder=0, alpha=alpha, label=label)  # , alpha=0.2)
    ax.set_xlim([0, x_time[-1]])

    # ax.set_xticks(ax.get_xticks()/sample_rate/downsample)
    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    return ax


def plot_vap_probs(
    p: torch.Tensor,
    ax: Axes,
    color: List[str] = ["b", "orange"],
    label: List[str] = ["A", "B"],
    prob_label: str = "P now",
    yticks: list[str] = ["B", "A"],
    alpha_ns: float = 0.6,
    fontsize: int = 12,
    no_xticks: bool = True,
    legend: bool = True,
    frame_hz: int = 50,
) -> Axes:
    assert p.ndim == 1, f"Expected p shape (N_FRAMES) got {p.shape}"
    p = p.cpu()
    x = torch.arange(len(p)) / frame_hz
    ax.fill_between(
        x,
        y1=0.5,
        y2=p,
        where=p > 0.5,
        alpha=alpha_ns,
        color=color[0],
        label=label[0],
    )
    ax.fill_between(
        x,
        y1=p,
        y2=0.5,
        where=p < 0.5,
        alpha=alpha_ns,
        color=color[1],
        label=label[1],
    )
    ax.plot(x, p, color="k", linewidth=1, label=prob_label, zorder=4)
    ax.set_yticks([0.25, 0.75], yticks, fontsize=fontsize)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, x[-1]])

    if legend:
        ax.legend(loc="lower left")
    ax.axhline(y=0.5, linestyle="dashed", linewidth=2, color="k")

    if no_xticks:
        ax.set_xticks([])

    return ax


def plot_vad(
    x,
    vad,
    ax: Axes,
    ypad: float = 0,
    color: str = "w",
    label: Optional[str] = None,
    **kwargs,
):
    assert vad.ndim == 1, f"Expects (N_FRAMES, ) got {vad.shape}"
    ymin, ymax = ax.get_ylim()
    scale = ymax - ymin - ypad
    ax.plot(x, ymin + vad.cpu() * scale, color=color, label=label, **kwargs)


def plot_stereo(waveform, p_now, p_fut, vad=None, plot=False, figsize=(10, 6)):
    assert waveform.ndim == 2, f"Expects (2, N_SAMPLES) got {waveform.shape}"
    assert p_now.ndim == 1, f"Expects (N_FRAMES) got {p_now.shape}"
    assert p_now.ndim == 1, f"Expects (N_FRAMES) got {p_now.shape}"

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize)
    plot_melspectrogram(waveform, ax=ax[:2])
    if vad is not None:
        x = torch.arange(len(p_now)) / 50
        plot_vad(x, vad[:, 0], ax=ax[0])
        plot_vad(x, vad[:, 1], ax=ax[1])
    plot_vap_probs(p_now, ax=ax[2], prob_label="P now")
    plot_vap_probs(p_fut, ax=ax[3], prob_label="P future")
    plt.tight_layout()

    if plot:
        plt.pause(0.001)
    return fig, ax


if __name__ == "__main__":

    from vap.data.datamodule import VAPDataset
    from vap.utils.audio import mono_to_stereo

    dset = VAPDataset(path="example/data/sliding_dev.csv")

    ii = 0
    d = dset[ii]
    mono = d["waveform"].mean(0).unsqueeze(0)
    vad_list = dset.df.iloc[ii]["vad_list"]
    stereo = mono_to_stereo(mono, vad_list, sample_rate=16_000)

    fig, ax = plt.subplots(3, 1)
    plot_melspectrogram(mono, ax[0])
    plot_melspectrogram(stereo, ax=[ax[1], ax[2]])
    plt.show()

    fig, ax = plt.subplots(3, 1)
    plot_waveform(mono[0], ax[0])
    plot_waveform(stereo[0], ax[1])
    plot_waveform(stereo[1], ax[2])
    plt.show()
