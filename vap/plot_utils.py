import torch
import torchaudio.transforms as AT
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple

from vap.utils import read_json

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 320


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 3:
        return waveform.mean(-2, keepdim=True)
    elif waveform.ndim == 2 and waveform.shape[0] == 2:
        return waveform.mean(0, keepdim=True)
    else:
        raise NotImplementedError(
            f"{waveform.shape} must be (N, 2, n_samples) or (2, n_samples)"
        )


def log_mel_spectrogram(
    waveform: torch.Tensor,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    mel_spec = AT.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        normalized=True,
    )(waveform)
    log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
    log_mel_spec = (log_mel_spec + 4.0) / 4.0
    return log_mel_spec


def plot_stereo_mel_spec(
    waveform: torch.Tensor,
    ax: List[mpl.axes.Axes],
    vad: Optional[torch.Tensor] = None,
    mel_spec: Optional[torch.Tensor] = None,
    plot: bool = False,
) -> List[mpl.axes.Axes]:
    if mel_spec is None:
        mel_spec = log_mel_spectrogram(waveform)

    colors = ["b", "orange"]
    n_channels, n_mels, n_frames = mel_spec.shape
    for ch in range(n_channels):
        ax[ch].imshow(mel_spec[ch], aspect="auto", origin="lower")
        if vad is not None:
            ax[ch].plot(
                vad[:n_frames, ch] * (n_mels - 1),
                alpha=0.9,
                linewidth=5,
                color=colors[ch],
            )

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    plt.subplots_adjust(
        left=0.05, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0
    )
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

        ax.plot(0.5 + p_bc[:, 0], color="darkgreen")
        ax.plot(0.5 - p_bc[:, 1], color="darkgreen")
        ax.fill_between(
            x, 0.5 + p_bc[:, 0], mid_line, color="g", alpha=alpha_bc, label="BC"
        )
        ax.fill_between(x, mid_line, 0.5 - p_bc[:, 1], color="g", alpha=alpha_bc)
        # ax.hlines(y=0.5, xmin=0, xmax=n_frames, color="k")

    if legend:
        ax.legend(loc="lower left")
    ax.hlines(y=0.5, xmin=0, xmax=len(p_ns), linewidth=2, color="k")
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
