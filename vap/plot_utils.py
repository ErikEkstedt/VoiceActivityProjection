import torch
import torchaudio.transforms as AT
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple

from vap.audio import log_mel_spectrogram
from vap.utils import read_json
import vap.functional as VF


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 3:
        return waveform.mean(-2, keepdim=True)
    elif waveform.ndim == 2 and waveform.shape[0] == 2:
        return waveform.mean(0, keepdim=True)
    else:
        raise NotImplementedError(
            f"{waveform.shape} must be (N, 2, n_samples) or (2, n_samples)"
        )


def plot_stereo(
    waveform: torch.Tensor,
    p_ns: torch.Tensor,
    vad: torch.Tensor,
    plot: bool = True,
    figsize=(9, 6),
):
    assert (
        waveform.ndim == 2
    ), f"Expected waveform of shape (2, n_samples) got {waveform.shape}"

    assert (
        waveform.shape[0] == 2
    ), f"Expected waveform of shape (2, n_samples) got {waveform.shape}"

    assert vad.ndim == 2, f"Expected vad of shape (n_frames, 2) got {vad.shape}"
    assert vad.shape[-1] == 2, f"Expected vad of shape (n_frames, 2) got {vad.shape}"

    fig, ax = plt.subplots(4, 1, figsize=figsize)
    _ = plot_waveform(waveform=waveform[0], ax=ax[0])
    _ = plot_waveform(waveform=waveform[1], ax=ax[0], color="orange")
    ax[0].set_xticks([])

    plot_stereo_mel_spec(waveform, ax=[ax[1], ax[2]], vad=vad)
    plot_next_speaker_probs(p_ns=p_ns, ax=ax[3])
    plt.subplots_adjust(
        left=0.08, bottom=None, right=None, top=None, wspace=None, hspace=0.04
    )
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_waveform(
    waveform,
    ax: mpl.axes.Axes,
    color: str = "lightblue",
    alpha: float = 0.6,
    downsample: int = 10,
    sample_rate: int = 16000,
) -> mpl.axes.Axes:
    assert (
        waveform.ndim == 1
    ), f"Expects a single channel waveform (n_samples, ) got {waveform.shape}"
    x = waveform[..., ::downsample]
    ax.plot(x, color=color, zorder=0, alpha=alpha)  # , alpha=0.2)
    ax.set_xlim([0, len(x)])
    # ax.set_xticks(ax.get_xticks()/sample_rate/downsample)
    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    ax.set_ylabel("waveform", fontsize=14)
    return ax


def plot_stereo_mel_spec(
    waveform: torch.Tensor,
    ax: List[mpl.axes.Axes],
    vad: Optional[torch.Tensor] = None,
    mel_spec: Optional[torch.Tensor] = None,
    fontsize: int = 12,
    plot: bool = False,
) -> List[mpl.axes.Axes]:
    if mel_spec is None:
        mel_spec = log_mel_spectrogram(waveform)

    colors = ["b", "orange"]
    n_channels, n_mels, n_frames = mel_spec.shape
    for ch in range(n_channels):
        # print(mel_spec[ch].max(), mel_spec[ch].min())
        ax[ch].imshow(mel_spec[ch], aspect="auto", origin="lower", vmin=-1.5, vmax=1.5)
        if vad is not None:
            ax[ch].plot(
                vad[:n_frames, ch] * (n_mels - 1),
                alpha=0.9,
                linewidth=2,
                color=colors[ch],
            )

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[0].set_ylabel("A", fontsize=fontsize)
    ax[1].set_ylabel("B", fontsize=fontsize)
    plt.subplots_adjust(
        left=0.05, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0
    )
    if plot:
        plt.pause(0.1)
    return ax


def plot_mel_spec(
    waveform: torch.Tensor,
    ax: mpl.axes.Axes,
    vad: Optional[torch.Tensor] = None,
    mel_spec: Optional[torch.Tensor] = None,
    no_ticks: bool = False,
    cmap: str = "inferno",
    interpolation: bool = True,
    frame_hz: int = 50,
    sample_rate: int = 16000,
    plot: bool = False,
) -> List[mpl.axes.Axes]:
    if mel_spec is None:
        hop_length = int(sample_rate / frame_hz)
        mel_spec = log_mel_spectrogram(waveform, hop_length=hop_length)

    if mel_spec.ndim == 2:
        pass
    elif mel_spec.ndim == 3 and mel_spec.shape[0] == 1:
        mel_spec = mel_spec.squeeze(0)
    else:
        raise NotImplementedError(
            f'Trying to plot multiple channels with. Use "plot_stereo_mel_spec" instead. {waveform.shape}'
        )

    n_mels, n_frames = mel_spec.shape

    interp = None
    if not interpolation:
        interp = "none"
    ax.imshow(mel_spec, aspect="auto", origin="lower", interpolation=interp, cmap=cmap)
    if vad is not None:
        ax.plot(vad[:n_frames] * (n_mels - 1), alpha=0.9, linewidth=5, color="b")
    if no_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if plot:
        plt.pause(0.1)
    return ax


def plot_next_speaker_probs(
    p_ns: Union[np.ndarray, torch.Tensor],
    ax: mpl.axes.Axes,
    p_bc: Optional[Union[np.ndarray, torch.Tensor]] = None,
    vad: Optional[Union[np.ndarray, torch.Tensor]] = None,
    color: Union[List[str], str] = ["b", "orange"],
    alpha_ns: float = 0.6,
    alpha_bc: float = 0.3,
    legend: bool = False,
    fontsize: int = 12,
) -> mpl.axes.Axes:
    if isinstance(p_ns, np.ndarray):
        p_ns = torch.from_numpy(p_ns)

    if p_ns.ndim == 2:
        p_ns = p_ns[:, 0]  # choose first speaker (they always sum to 1)

    x = torch.arange(len(p_ns))
    ax.fill_between(
        x,
        y1=0.5,
        y2=p_ns,
        where=p_ns > 0.5,
        alpha=alpha_ns,
        color=color[0],
        label="A",
    )
    ax.fill_between(
        x,
        y1=p_ns,
        y2=0.5,
        where=p_ns < 0.5,
        alpha=alpha_ns,
        color=color[1],
        label="B",
    )
    ax.plot(p_ns, color="k", linewidth=0.8)
    ax.set_xlim([0, len(p_ns)])
    ax.set_xticks([])
    ax.set_yticks([0.25, 0.75], ["SHIFT", "HOLD"], fontsize=fontsize)
    ax.set_ylim([0, 1])

    if p_bc is not None:
        if isinstance(p_bc, np.ndarray):
            p_bc = torch.from_numpy(p_bc)

        n_frames, _ = p_bc.shape
        x = torch.arange(n_frames)

        mid_line = torch.ones(n_frames) * 0.5

        not_vad = 1.0
        if vad is not None:
            if isinstance(vad, np.ndarray):
                vad = torch.from_numpy(vad)
            not_vad = torch.logical_not(vad[:n_frames]).float()
            p_bc = p_bc * not_vad

        ax.plot(0.5 + p_bc[:, 0] / 2, color="darkgreen")
        ax.plot(0.5 - p_bc[:, 1] / 2, color="darkgreen")
        ax.fill_between(
            x, 0.5 + p_bc[:, 0] / 2, mid_line, color="g", alpha=alpha_bc, label="BC"
        )
        ax.fill_between(x, mid_line, 0.5 - p_bc[:, 1] / 2, color="g", alpha=alpha_bc)
        # ax.hlines(y=0.5, xmin=0, xmax=n_frames, color="k")

    if legend:
        ax.legend(loc="lower left")
    ax.axhline(y=0.5, linestyle="dashed", linewidth=2, color="k")
    return ax


def plot_evaluation_scores(
    scores: Union[str, dict],
    figsize: Union[List[int], Tuple[int, int]] = (6, 4),
    plot: bool = False,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes, dict]:
    """

    {
      "f1_hold_shift": 0.8752257227897644,
      "f1_predict_shift": 0.7964005470275879,
      "f1_short_long": 0.7920717000961304,
      "f1_bc_prediction": 0.7287405729293823,
      "shift": {
        "f1": 0.6144151091575623,
        "precision": 0.6098514199256897,
        "recall": 0.6190476417541504,
        "support": 15120
      },
      "hold": {
        "f1": 0.9253013730049133,
        "precision": 0.9266447424888611,
        "recall": 0.923961877822876,
        "support": 78750
      },
      "loss": 1.7417683601379395,
      "threshold_pred_shift": 0.08999999612569809,
      "threshold_pred_bc": 0.04999999701976776,
      "threshold_short_long": 0.3100000023841858
    }
    """
    if isinstance(scores, str):
        scores = read_json(scores)

    heights = [
        scores["f1_hold_shift"],
        scores["f1_predict_shift"],
        scores["f1_short_long"],
        scores["f1_bc_prediction"],
    ]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(x=list(range(4)), height=heights)
    for xx, metric in enumerate(
        ["f1_hold_shift", "f1_predict_shift", "f1_short_long", "f1_bc_prediction"]
    ):
        ax.text(
            x=xx,
            y=scores[metric],
            s=f"{scores[metric]:.3f}",
            fontsize=12,
            # fontweight='bold',
            horizontalalignment="center",
        )
    # Shift/Hold specific
    ax.text(
        x=0,
        y=max(scores["f1_hold_shift"] - 0.1, 0),
        s=f'shift: {scores["shift"]["f1"]:.3f}\nhold: {scores["hold"]["f1"]:.3f}',
        fontsize=10,
        horizontalalignment="center",
    )

    # Thresholds
    ax.text(
        x=3.4,
        y=0.85,
        s=f'Thresholds\nSL: {scores["threshold_short_long"]:.3f}\nPred-S: {scores["threshold_pred_shift"]:.3f}\nPred-BC: {scores["threshold_pred_bc"]:.3f}',
        horizontalalignment="right",
        fontsize=10,
    )
    ax.set_title(f"Turn-taking Events: loss={scores['loss']:.3f}")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["SH", "Pred-S", "SL", "Pred-BC"], fontsize=14)
    ax.set_ylim([0.5, 1])
    ax.set_ylabel("F1 (weighted)", fontsize=14)

    if plot:
        plt.pause(0.1)
    return fig, ax, scores


def plot_words(
    words,
    word_starts: List[float],
    ax: mpl.axes.Axes,
    word_ends: Optional[List[float]] = None,
    rows: int = 4,
    frame_hz: int = 50,
    fontsize: int = 12,
    color: str = "k",
    linewidth: int = 2,
):
    if word_ends is None:
        word_ends = [None for a in word_starts]
    y_min, y_max = ax.get_ylim()
    diff = y_max - y_min
    pad = diff * 0.05
    # Plot text on top of waveform
    for ii, (word, start_time, end_time) in enumerate(
        zip(words, word_starts, word_ends)
    ):
        yy = pad + y_min + diff * (ii % rows) / rows
        start_text = start_time * frame_hz

        alignment = "left"
        if end_time is not None:
            alignment = "center"
            x_text = start_text + 0.5 * frame_hz * (end_time - start_time)
        else:
            x_text = start_text

        ax.vlines(
            start_text,
            ymin=y_min + pad,
            ymax=y_max - pad,
            linestyle="dashed",
            linewidth=linewidth,
            color=color,
            alpha=0.8,
        )
        ax.text(
            x=x_text,
            y=yy,
            s=word,
            fontsize=fontsize,
            horizontalalignment=alignment,
            color=color,
        )

    # final end of word
    if word_ends[0] is not None:
        ax.vlines(
            word_ends[-1] * frame_hz,
            ymin=y_min + pad,
            ymax=y_max - pad,
            linewidth=3,
            color="r",
            alpha=0.8,
        )

    return ax


def plot_sample_waveform(
    sample, ax: mpl.axes.Axes, downsample: int = 10, sample_rate: int = 16000
) -> mpl.axes.Axes:

    x = sample["waveform"].squeeze()[..., ::downsample]
    ax.plot(x, color="lightblue", zorder=0)  # , alpha=0.2)
    ax.set_xlim([0, len(x)])
    ax.set_xticks([])
    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    ax.set_ylabel("waveform", fontsize=14)
    if "words" in sample:
        ax = plot_words(
            sample["words"],
            word_starts=sample["starts"],
            word_ends=sample.get("ends", None),
            ax=ax,
            fontsize=14,
            linewidth=2,
            frame_hz=int(sample_rate / downsample),
        )
    return ax


def plot_sample_mel_spec(
    sample, ax: mpl.axes.Axes, frame_hz: int = 50
) -> mpl.axes.Axes:
    ax = plot_mel_spec(sample["waveform"].squeeze(), ax=ax, cmap="magma", no_ticks=True)
    ax.yaxis.tick_right()
    ax.set_ylabel("Mel (Hz)", fontsize=14)
    if "words" in sample:
        ax = plot_words(
            sample["words"],
            word_starts=sample["starts"],
            word_ends=sample.get("ends", None),
            ax=ax,
            fontsize=14,
            frame_hz=frame_hz,
            color="w",
        )
    return ax


def plot_sample_f0(
    sample, ax: mpl.axes.Axes, sample_rate: int = 16000
) -> mpl.axes.Axes:
    f0 = VF.pitch_praat(sample["waveform"].squeeze(), sample_rate=sample_rate)
    f0[f0 == 0] = torch.nan
    ax.plot(f0, "o", markersize=3, color="b")
    ymin, ymax = ax.get_ylim()
    diff = ymax - ymin
    if diff < 10:
        ymin -= 5
        ymax += 5
        ax.set_ylim([ymin, ymax])
    ax.set_xlim([0, len(f0)])
    ax.set_xticks([])
    ax.set_ylabel("F0 (Hz)", fontsize=14)
    ax.yaxis.tick_right()
    return ax


def plot_phrases_sample(sample, probs, frame_hz: int = 50, sample_rate: int = 16000):
    # Calculate axs
    fig, ax = plt.subplots(4, 1, figsize=(9, 6))
    ax[0] = plot_sample_waveform(sample, ax=ax[0], sample_rate=sample_rate)
    ax[1] = plot_sample_mel_spec(sample, ax=ax[1], frame_hz=frame_hz)
    ax[2] = plot_sample_f0(sample, ax=ax[2])
    ax[3] = plot_next_speaker_probs(p_ns=probs["p"][0], ax=ax[3])  # , p_bc=p_bc)
    if sample.get("ends", None) is not None:
        end_frame = sample["ends"][-1] * frame_hz
        ax[3].vlines(x=end_frame, ymin=-1, ymax=1, color="r", linewidth=2)
    plt.subplots_adjust(left=0.08, bottom=0.01, right=0.95, top=0.99, hspace=0.04)
    return fig, ax


if __name__ == "__main__":

    from datasets_turntaking import DialogAudioDM
    from vap.model import VAPModel
    import matplotlib.pyplot as plt

    dm = DialogAudioDM(
        datasets=["fisher"],
        audio_mono=False,
        audio_duration=10,
        audio_overlap=5,
        flip_channels=False,
        vad_hz=50,
        vad_history=True,
    )
    dm.prepare_data()
    dm.setup()

    # Load Model
    checkpoint = "example/50hz_48_10s-epoch20-val_1.85.ckpt"
    model = VAPModel.load_from_checkpoint(checkpoint)

    # d = dm.train_dset[10]
    d = dm.train_dset[10]
    waveform = d["waveform"][0]  # (1, 2, n_samples) -> (n_samples,)
    vad = d["vad"][0]  # (1, n_frames, 2) -> (n_frames, 2)
    mel_spec = log_mel_spectrogram(waveform, hop_length=320)

    # Forward (stereo->mono)
    d["waveform"] = d["waveform"].mean(-2, keepdim=True)
    loss, out, probs, batch = model.output(d)

    ###################################################
    # Figure
    ###################################################
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    _, ax_mels = plot_stereo_mel_spec(
        waveform, mel_spec=mel_spec, vad=vad, ax=[ax[0], ax[2]], plot=False
    )
    ax[1] = plot_next_speaker_probs(
        p_ns=probs["p"][0],
        ax=ax[1],
        p_bc=probs["bc_prediction"][0],
        vad=d["vad"][0],
        alpha_ns=0.8,
        legend=True,
    )
    plt.show()

    fig, ax, score = plot_evaluation_scores(
        "runs/runs_evaluation/50hz_48_10s-epoch20-val_1.85_switchboard_fisher/metric.json",
        plot=True,
    )
