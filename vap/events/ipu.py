import torch
from torch import Tensor
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torchaudio.functional as AF


from vap.utils.audio import load_waveform
from vap.utils.utils import (
    read_json,
    vad_list_to_onehot,
    vad_onehot_to_vad_list,
    get_dialog_states,
    get_vad_list_subset,
    find_island_idx_len,
)
from vap.utils.plot import plot_melspectrogram
import vap.functional as VF

from typing import Optional

TRIAD_HOLD: Tensor = torch.tensor([[0, 1, 0], [3, 1, 3]])  # on silence

# TODO: move to vap.utils.utils
def load_vad(path: str, duration=None, frame_hz=50):
    vad_list = read_json(path)
    return vad_list_to_onehot(vad_list, duration=duration, frame_hz=frame_hz)


# TODO: move to vap.utils.utils
def load_sample(
    audio_path: Optional[str] = None,
    vad_path: Optional[str] = None,
    word_path: Optional[str] = None,
    ipu: bool = True,
    turn: bool = True,
    sample_rate: int = 16_000,
    mono: bool = False,
    end_time=None,
    start_time=None,
    frame_hz=50,
):
    """
    Load a sample from the dataset given the audio path and the vad path
    :param audio_path: path to the audio file
    :param vad_path: path to the vad file
    :param sample_rate: sample rate of the audio
    :param mono: mono or stereo
    :param end_time: end time of the audio
    :param start_time: start time of the audio
    :return: audio, vap, vad
    """

    ret = {}

    if audio_path:
        ret["waveform"], _ = load_waveform(
            audio_path,
            start_time=start_time,
            end_time=end_time,
            sample_rate=sample_rate,
            mono=mono,
        )
        ret["duration"] = ret["waveform"].shape[-1] / sample_rate
        ret["session"] = audio_path.split("/")[-1].split(".")[0]

    if vad_path:
        ret["vad_list"] = read_json(vad_path)

        if "session" not in ret:
            ret["session"] = vad_path.split("/")[-1].split(".")[0]

        if "duration" not in ret:
            ret["duration"] = max(
                ret["vad_list"][0][-1][-1], ret["vad_list"][1][-1][-1]
            )
        ret["vad"] = vad_list_to_onehot(
            ret["vad_list"], duration=ret["duration"], frame_hz=frame_hz
        )

    if word_path:
        ret["words"] = pd.read_csv(word_path)

    if ipu:
        ret["ipu"] = to_ipu(ret["vad"])

    if turn:
        if "ipu" not in ret:
            ret["ipu"] = to_ipu(ret["vad"])
        # A turn merges consecutive IPUs separated by holds
        ret["turns"] = fill_holds(ret["ipu"])
        ret["turn_list"] = vad_onehot_to_vad_list(ret["turns"].unsqueeze(0))[0]

        # add speaker and sort
        tl = [(s, e, 0) for s, e in ret["turn_list"][0]] + [
            (s, e, 1) for s, e in ret["turn_list"][1]
        ]
        tl.sort()
        ret["turn_list_all"] = tl

    return ret


# TODO: vap.utils.words ?
def df_to_utterances_list(df: pd.DataFrame) -> list:
    """
    Get utterances (by speaker) from words
    """
    df.sort_values(by="start", inplace=True)
    utts = []
    spkrs = []
    curr_utt = {"text": []}
    curr_speaker = None
    for i in range(len(df)):
        d = df.iloc[i]
        sp = d.speaker
        if sp == curr_speaker:
            curr_utt["text"].append(str(d.word))
            curr_utt["starts"].append(d.start)
            curr_utt["ends"].append(d.end)
            curr_utt["end"] = d.end
        else:
            if len(curr_utt["text"]) > 0:
                curr_utt["text"] = " ".join(curr_utt["text"])
                utts.append(curr_utt)
                spkrs.append(curr_speaker)
            curr_utt = {
                "text": [d.word],
                "start": d.start,
                "end": d.end,
                "speaker": sp,
                "starts": [d.start],
                "ends": [d.end],
                "channel": 0 if d.speaker == "A" else 1,
            }
            curr_speaker = sp
    if len(curr_utt["text"]) > 0:
        curr_utt["text"] = " ".join(curr_utt["text"])
        utts.append(curr_utt)
        spkrs.append(curr_speaker)

    return utts


# TODO: vap.utils.words ?
def df_to_utterances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get utterances (by speaker) from words
    """
    utts = df_to_utterances_list(df)
    uf = pd.DataFrame(utts)
    return uf


# TODO: move to vap.utils.utils
@torch.no_grad()
def fill_holds(
    vad: torch.Tensor,
    ds: Optional[torch.Tensor] = None,
    islands: Optional[tuple[Tensor, Tensor, Tensor]] = None,
) -> torch.Tensor:
    assert vad.ndim == 2, "fill_pauses require vad=(n_frames, 2)"

    filled_vad = vad.clone()

    if islands is None:
        if ds is None:
            ds = get_dialog_states(vad)

        assert ds.ndim == 1, "fill_pauses require ds=(n_frames,)"
        s, d, v = find_island_idx_len(ds)
    else:
        s, d, v = islands

    # less than three entries means that there are no pauses
    # requires at least: activity-from-speaker  ->  Silence   --> activity-from-speaker
    if len(v) < 3:
        return vad

    # Find all holds/pauses and fill them
    # A hold is defined to not contain activity by the listener
    # inside the pause
    triads = v.unfold(0, size=3, step=1)
    next_speaker, steps = torch.where(
        (triads == TRIAD_HOLD.unsqueeze(1).to(triads.device)).sum(-1) == 3
    )
    for ns, pre in zip(next_speaker, steps):
        cur = pre + 1
        # Fill the matching template
        filled_vad[s[cur] : s[cur] + d[cur], ns] = 1.0
    return filled_vad


# TODO: move to vap.utils.utils
def to_ipu_tensor(
    vad: torch.Tensor, ipu_time: float = 0.2, frame_hz: int = 50
) -> torch.Tensor:
    """
    Fill all silence shorter than `ipu_time` with activity
    Convert a vad to ipu
    :param vad: vad tensor
    :param frame_hz: frame hz of the vad
    :return: ipu tensor
    """
    ipu = vad.clone()
    n_frames = int(ipu_time * frame_hz)
    for channel in range(vad.shape[-1]):
        start, duration, value = find_island_idx_len(vad[:, channel])
        duration = duration[value == 0]
        start = start[value == 0]
        w = torch.where(duration <= n_frames)[0]
        for s, d in zip(start[w], duration[w]):
            ipu[s : s + d, channel] = 1.0
    return ipu


def to_ipu(vad, ipu_time=0.2, frame_hz=50):
    if isinstance(vad, list):
        ipu_frames = int(ipu_time * frame_hz)
        ipu_list = []
        for vad_channel in vad:
            ch_ipu = []
            last_end = vad_channel[0][1]
            ch_ipu = [vad_channel[0]]
            for s, e in vad_channel[1:]:
                if s - last_end <= ipu_frames:
                    ch_ipu[-1][-1] = e
                    last_end = e
                else:
                    ch_ipu.append([s, e])
            ipu_list.append(ch_ipu)
        return ipu_list
    else:
        return to_ipu_tensor(vad, ipu_time, frame_hz)


def stats_ipu(ipu: Tensor, ymax=10, bins=20, frame_hz: int = 50):
    start, duration, values = find_island_idx_len(ipu[:, 0])
    ipu_duration = duration[values == 1]
    # silence_duration = duration[values == 0]
    start2, duration2, values2 = find_island_idx_len(ipu[:, 1])
    ipu_duration2 = duration2[values2 == 1]
    # silence_duration2 = duration2[values == 0]

    asum = ipu_duration.sum() / frame_hz
    bsum = ipu_duration2.sum() / frame_hz
    ra = (asum / (asum + bsum)).item()
    rb = (bsum / (asum + bsum)).item()

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].hist(
        ipu_duration / frame_hz,
        bins=bins,
        range=(0, ymax),
        color="blue",
        alpha=0.6,
        label=f"RA: {ra*100:.2f}%",
    )
    ax[1].hist(
        ipu_duration2 / frame_hz,
        bins=bins,
        range=(0, ymax),
        color="orange",
        alpha=0.6,
        label=f"RB: {rb*100:.2f}%",
    )
    ax[1].set_xlabel("IPU duration (s)")
    ax[1].set_ylabel("N")
    for a in ax:
        a.legend()
    ax[1].set_xlim([0, ymax])
    plt.tight_layout()
    plt.show()


# TODO: move to vap.utils.plot
def plot_words_time(
    words,
    starts: list[float],
    ax,
    ends=None,
    rows: int = 4,
    fontsize: int = 14,
    color: str = "w",
    linewidth: int = 1,
    linealpha: float = 0.6,
    **kwargs,
):
    if ends is None:
        ends = [None for _ in starts]

    y_min, y_max = ax.get_ylim()
    diff = y_max - y_min
    pad = diff * 0.1

    # Plot text on top of waveform
    for ii, (word, start_time, end_time) in enumerate(zip(words, starts, ends)):
        yy = pad + y_min + diff * (ii % rows) / rows

        alignment = "left"
        if end_time is not None:
            alignment = "center"
            x_text = start_time + 0.5 * (end_time - start_time)
        else:
            x_text = start_time

        ax.vlines(
            start_time,
            ymin=y_min + pad,
            ymax=y_max - pad,
            linestyle="dashed",
            linewidth=linewidth,
            color=color,
            alpha=linealpha,
        )
        ax.text(
            x=x_text,
            y=yy,
            s=word,
            fontsize=fontsize,
            # fontweight="bold",
            horizontalalignment=alignment,
            color=color,
            **kwargs,
        )

        if end_time is not None:
            ax.vlines(
                end_time,
                ymin=y_min + pad,
                ymax=y_max - pad,
                linestyle="dashed",
                linewidth=linewidth,
                color=color,
                alpha=linealpha,
            )

    return ax


def get_waveform_subset(waveform, start, end, sample_rate=16_000):
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    return waveform[..., start_idx:end_idx]


def plot_ipu(
    d, start, end, channel, pad=1.0, min_duration=5, f0_max=400, figsize=(16, 4)
):
    """
    plot ipu center
    """
    # Select the words in the turn
    # using start/end time
    speaker = "A" if channel == 0 else "B"
    turn_words = d["words"][
        (d["words"]["speaker"] == speaker)
        & (d["words"]["end"] >= start)
        & (d["words"]["start"] <= end)
    ]
    # print(turn_words)
    print(" ".join(turn_words.word.values.tolist()))

    fig_end = end + pad
    fig_start = start - pad
    if fig_start < 0:
        pad = start
        fig_start = 0

    audio_duration = d["waveform"].shape[-1] / 16_000
    if fig_end > audio_duration:
        fig_end = audio_duration

    dur = fig_end - fig_start
    if dur < min_duration:
        diff = min_duration - dur
        h = diff / 2
        if fig_start - h > 0:
            fig_start = fig_start - h
            fig_end = fig_end + h
        else:
            fig_end = fig_end + diff

    w = get_waveform_subset(d["waveform"], fig_start, fig_end)
    vad_list = get_vad_list_subset(d["vad_list"], fig_start, fig_end)
    vad = vad_list_to_onehot(vad_list, duration=fig_end - fig_start, frame_hz=50)

    # Rephrase the abouve code to be more efficient
    wa = d["words"][
        (d["words"]["speaker"] == "A")
        & (d["words"]["end"] >= fig_start)
        & (d["words"]["start"] <= fig_end)
    ]
    wb = d["words"][
        (d["words"]["speaker"] == "B")
        & (d["words"]["end"] >= fig_start)
        & (d["words"]["start"] <= fig_end)
    ]

    #######################################################################
    fig, ax = plt.subplots(
        3, 1, figsize=figsize, sharex=True  # , height_ratios=[1, 1, 0.5]
    )
    plot_melspectrogram(w, ax=ax[:2])
    if len(wa) > 0:
        # color = 'w' if channel == 0 else 'gray'
        fw = "bold" if channel == 0 else None
        plot_words_time(
            words=wa.word.values,
            starts=wa.start.values - fig_start,
            ends=wa.end.values - fig_start,
            ax=ax[0],
            rows=max(5, len(wb) // 3),
            fontweight=fw,
        )
    if len(wb) > 0:
        # color = 'w' if channel == 1 else 'gray'
        fw = "bold" if channel == 1 else None
        plot_words_time(
            words=wb.word.values,
            starts=wb.start.values - fig_start,
            ends=wb.end.values - fig_start,
            ax=ax[1],
            rows=max(5, len(wb) // 3),
            fontweight=fw,
        )

    # PROSODY
    f0 = VF.pitch_praat(w, hop_time=0.02, f0_max=f0_max)
    if f0.shape[-1] != vad.shape[0]:
        diff = vad.shape[0] - f0.shape[-1]
        if diff > 0:
            print("f0: ", tuple(f0.shape))
            f0 = F.pad(f0, pad=(0, diff), mode="constant", value=0)
            print("f0: ", tuple(f0.shape))

    f0 = (f0 * vad[: f0.shape[-1]].t()).numpy()
    f0[f0 == 0] = np.nan
    x = torch.arange(f0.shape[-1]) / 50
    ax[-1].plot(x, f0[0], color="blue", linewidth=3)
    ax[-1].plot(x, f0[1], color="orange", linewidth=3)
    # tax = ax[-1].twinx()
    # tax.plot(x, f0[1], color="orange", linewidth=3)

    ax[-1].set_xlim([0, fig_end - fig_start])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    return fig, ax


# SCRIPTS
def example_ipu_turn_hist():
    bins = 50
    ipu_max = 10
    frame_hz = 50
    ipus = []
    turns = []
    for ii in tqdm(range(len(df))):
        audio_path, vad_path = df["audio_path"].iloc[ii], df["vad_path"].iloc[ii]
        name = audio_path.split("/")[-1].split(".")[0]
        word_path = f"/home/erik/projects/CCConv/vap_switchboard/data/words/{name}.csv"
        # d = load_sample(audio_path, vad_path, word_path)
        d = load_sample(vad_path=vad_path)
        # IPUs are VA segments separated by less than 200ms
        # Indeto_ipu_tensort of activty in the 'other' channel
        # Separated by holds
        ipu = to_ipu(d["vad"])
        ipus.append(
            get_duration(ipu, value=1, frame_hz=frame_hz).histc(
                bins=bins, min=0, max=ipu_max
            )
        )
        # A turn merges consecutive IPUs separated by holds
        turn = fill_holds(ipu)
        turns.append(
            get_duration(turn, value=1, frame_hz=frame_hz).histc(
                bins=bins, min=0, max=ipu_max
            )
        )
    ipus = torch.stack(ipus)
    turns = torch.stack(turns)

    fig, ax = plt.subplots(2, 1, sharex=True)
    # ax.imshow(turns, aspect='auto', origin='lower')
    # plt.show()
    x = torch.arange(len(p)) * ipu_max / bins
    p_ipu = ipus.sum(0)
    p_ipu /= p_ipu.sum()
    p_turn = turns.sum(0)
    p_turn /= p_turn.sum()
    ax[0].bar(
        x,
        height=100 * p_ipu,
        align="edge",
        width=x[1] - x[0],
        color="b",
        alpha=0.5,
        label="IPU",
    )
    ax[0].bar(
        x,
        height=100 * p_turn,
        align="edge",
        width=x[1] - x[0],
        color="g",
        alpha=0.5,
        label="TURN",
    )
    ax[1].bar(
        x,
        height=100 * p_ipu.cumsum(0),
        align="edge",
        width=x[1] - x[0],
        color="b",
        alpha=0.5,
        label="IPU",
    )
    ax[1].bar(
        x,
        height=100 * p_turn.cumsum(0),
        align="edge",
        width=x[1] - x[0],
        color="g",
        alpha=0.5,
        label="TURN",
    )
    ax[1].axhline(0.5 * 100, linestyle="--", color="k", linewidth=0.5)
    ax[1].axhline(0.9 * 100, linestyle="--", color="k", linewidth=0.5)
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlim([0, ipu_max])
    ax[1].set_xlabel("Duration (s)")
    ax[0].set_ylabel("%")
    ax[1].set_ylabel("%")
    plt.show()


def get_figure_lims(d, start, end, pad=0.5, min_duration=10):
    fig_end = end + pad
    fig_start = start - pad
    if fig_start < 0:
        pad = start
        fig_start = 0

    audio_duration = d["waveform"].shape[-1] / 16_000
    if fig_end > audio_duration:
        fig_end = audio_duration

    dur = fig_end - fig_start
    if dur < min_duration:
        diff = min_duration - dur
        h = diff / 2
        if fig_start > h:
            fig_start = fig_start - h
            fig_end = fig_end + h
            pad += h
        else:
            fig_end = fig_end + diff
    return fig_start, fig_end, pad


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    p = "/home/erik/projects/CCConv/VoiceActivityProjection/data/swb_all_vad.csv"
    df = pd.read_csv(p)
    ii = 2

    def get_duration(x, value=1, frame_hz=50):
        assert x.ndim == 2, f"x must be 2D got {x.ndim}"
        assert x.shape[-1] == 2, f"x must have 2 columns got {x.shape[-1]}"

        _, d, v = find_island_idx_len(x[:, 0])
        durs = d[v == value] / frame_hz

        _, d, v = find_island_idx_len(x[:, 1])
        durs = torch.cat((durs, d[v == value] / frame_hz))

        return durs

    dd = pd.DataFrame(turns.numpy())

    ii = 3
    frame_hz = 50
    audio_path, vad_path = df["audio_path"].iloc[ii], df["vad_path"].iloc[ii]
    name = audio_path.split("/")[-1].split(".")[0]
    word_path = f"/home/erik/projects/CCConv/vap_switchboard/data/words/{name}.csv"
    d = load_sample(audio_path, vad_path, word_path, frame_hz=frame_hz)
    d_ipu = get_duration(
        d["ipu"], value=1, frame_hz=frame_hz
    )  # .histc(bins=bins, min=0, max=ipu_max)
    d_turn = get_duration(
        d["turns"], value=1, frame_hz=frame_hz
    )  # .histc(bins=bins, min=0, max=ipu_max))

    def plot_waveform(
        waveform,
        ax,
        color: str = "blue",
        alpha: float = 0.6,
        to_sample_rate: int = 400,
        sample_rate: int = 16000,
        **kwargs,
    ):
        assert waveform.ndim == 1, f"waveform must be 1D got {waveform.ndim}"
        downsample_factor = sample_rate // to_sample_rate
        ww = waveform.unfold(0, step=downsample_factor, size=downsample_factor).max(-1)[
            0
        ]
        x = torch.arange(len(ww)) / to_sample_rate
        ax.fill_between(
            x, y1=ww.abs(), y2=-ww.abs(), color=color, alpha=alpha, **kwargs
        )
        return ax

    def bin_is_active(region, vad_list, channel):
        is_on = False
        for s, e in vad_list[channel]:
            if s > region[1]:
                break
            if e < region[0]:
                continue
            # completely inside region
            if s <= region[0] and region[1] <= e:
                # print('region surrounds')
                is_on = True
            elif region[0] <= s <= region[1]:
                # print('start inside region')
                if e >= region[1]:
                    p = (region[1] - s) / (region[1] - region[0])
                else:
                    # print('end inside region')
                    p = e - s / (region[1] - region[0])
                if p >= 0.5:
                    is_on = True
            elif region[0] <= e <= region[1]:
                # print('end inside region')
                if s <= region[0]:
                    p = (e - region[0]) / (region[1] - region[0])
                else:
                    # print('start inside region')
                    p = e - s / (region[1] - region[0])
                if p >= 0.5:
                    is_on = True
        return is_on

    def draw_projection_window_question(T, ax, box_alpha=0.6, horizon_time=2):
        ax[0].axvline(T, color="r", linewidth=2, label="Current Timestep")
        ax[1].axvline(T, color="r", linewidth=2)
        ax[0].axvspan(
            T,
            T + horizon_time,
            ymax=0.9,
            facecolor="white",
            alpha=box_alpha,
            edgecolor="r",
            linewidth=1,
            label="Prediction Label",
        )
        ax[1].axvspan(
            T,
            T + horizon_time,
            ymin=0.1,
            facecolor="white",
            alpha=box_alpha,
            edgecolor="r",
            linewidth=1,
        )
        # ax[0].text(0.5, -.2, s="?", transform=ax[0].transAxes, fontsize=45, color='r', fontweight='bold')
        plt.text(
            T + 1,
            0.75,
            s="?",
            fontsize=45,
            color="r",
            fontweight="bold",
            ha="center",
        )

    def draw_projection_window_skantze(
        T, vad_list, ax, box_alpha=0.3, horizon_time=3, frame_width=0.2, channel=0
    ):
        start = 0
        color = "blue" if channel == 0 else "orange"
        for end in range(1, int(horizon_time / frame_width)):
            ts = start * frame_width
            te = end * frame_width
            region = [ts + T, te + T]
            is_on = bin_is_active(region, vad_list, channel)
            if is_on:
                ax[channel].axvspan(
                    region[0],
                    region[1],
                    ymin=0.1,
                    ymax=0.9,
                    alpha=box_alpha,
                    facecolor=color,
                    edgecolor="k",
                )
                ax[channel].text(
                    (region[0] + region[1]) / 2,
                    0,
                    s="1",
                    fontsize=12,
                    color="k",
                    ha="center",
                )
            else:
                ax[channel].axvspan(
                    region[0],
                    region[1],
                    ymin=0.1,
                    ymax=0.9,
                    alpha=box_alpha,
                    facecolor="white",
                    edgecolor="k",
                )
                ax[channel].text(
                    (region[0] + region[1]) / 2,
                    0,
                    s="0",
                    fontsize=12,
                    color="k",
                    ha="center",
                )
            start = end

    def draw_projection_window_vap(
        T, vad_list, ax, box_alpha=0.3, bin_times=[0.2, 0.4, 0.6, 0.8]
    ):
        text_kw = {
            "fontsize": 12,
            "color": "k",
            "ha": "center",
        }
        for channel in range(2):
            ymax = 0.9 if channel == 0 else 1.0
            ymin = 0 if channel == 0 else 0.1
            color = "blue" if channel == 0 else "orange"
            start = 0
            span_kw = {
                "ymin": ymin,
                "ymax": ymax,
                "alpha": box_alpha,
                "edgecolor": "k",
                "linewidth": 1,
            }
            for bin_time in bin_times:
                end = round(start + bin_time, 1)
                region = [T + start, T + end]
                is_on = bin_is_active(region, vad_list, channel)
                if is_on:
                    ax[channel].axvspan(
                        region[0], region[1], facecolor=color, **span_kw
                    )
                    ax[channel].text((region[0] + region[1]) / 2, 0, s="1", **text_kw)
                else:
                    ax[channel].axvspan(
                        region[0], region[1], facecolor="white", **span_kw
                    )
                    ax[channel].text((region[0] + region[1]) / 2, 0, s="0", **text_kw)
                start = end

    def draw_vad(vad_list, ax):
        for vad_channel, vc in enumerate(vad_list):
            color = "blue" if vad_channel == 0 else "orange"
            for ii, (s, e) in enumerate(vc):
                if ii == 0:
                    ax[vad_channel].axvspan(s, e, color=color, alpha=0.3, label="VAD")
                else:
                    ax[vad_channel].axvspan(s, e, color=color, alpha=0.3)

    def plot_projection_label(
        waveform, vad_list, T=2.2, label="question", label_channel=0
    ):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 4))
        plot_waveform(waveform[0], ax[0], to_sample_rate=400, label="Waveform")
        plot_waveform(
            waveform[1], ax[1], to_sample_rate=400, color="orange", label="Waveform"
        )
        ax[0].set_ylabel("Speaker A", fontweight="bold", color="b", fontsize=14)
        ax[1].set_ylabel("Speaker B", fontweight="bold", color="orange", fontsize=14)
        draw_vad(vad_list, ax)
        for a in ax:
            a.set_ylim([-1, 1])
            a.set_xlim([0, fig_end - fig_start])
            a.set_yticks([])
        #################################################
        # Draw current timestep
        #################################################
        for ii, a in enumerate(ax):
            a.axvline(
                T, color="r", linewidth=2, label="Current Timestep" if ii == 0 else None
            )
        if label == "question":
            draw_projection_window_question(T, ax, box_alpha=0.6, horizon_time=2)
        elif label == "skantze":
            draw_projection_window_skantze(
                T, vad_list, ax, channel=label_channel, horizon_time=2
            )
        else:
            draw_projection_window_vap(T, vad_list, ax)
        ax[-1].set_xlabel("Time (s)", fontsize=14)
        ax[0].legend(loc="upper left")
        ax[1].legend(loc="upper left")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        return fig, ax

    #

    for start, end, channel in d["turn_list_all"]:
        fig_start, fig_end, pad = get_figure_lims(d, start, end, min_duration=5)
        waveform = get_waveform_subset(d["waveform"], fig_start, fig_end)
        vad_list = get_vad_list_subset(d["vad_list"], fig_start, fig_end)
        vad = vad_list_to_onehot(vad_list, duration=fig_end - fig_start, frame_hz=50)

        ##########################################################################
        f, a = plot_projection_label(waveform, vad_list, T=2.12, label="question")
        f1, a1 = plot_projection_label(
            waveform, vad_list, T=2.12, label="skantze", label_channel=0
        )
        f1, a1 = plot_projection_label(
            waveform, vad_list, T=2.12, label="skantze", label_channel=1
        )
        f2, a2 = plot_projection_label(waveform, vad_list, T=2.12, label="vap")
        plt.show()

        # print(" ".join(w.word.values.tolist()))
        plot_ipu(d, start, end, channel, min_duration=10, f0_max=300, figsize=(16, 4))
        plt.show()
