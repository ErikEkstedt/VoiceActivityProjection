import streamlit as st
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from vap.data.dset_event import VAPClassificationDataset
from vap.utils.plot import plot_melspectrogram, plot_vap_probs, plot_vad
from vap.utils.utils import read_json, get_vad_list_subset
from vap.utils.audio import load_waveform


SAMPLE_RATE = 16_000
CSV_PATH = {
    "swb": "data/splits/swb/val_events.csv",
    "fisher": "data/splits/fisher/val_events.csv",
    "fisher_swb": "data/splits/fisher_swb/val_events.csv",
}
DSET = {
    "word": {
        "swb": "/home/erik/projects/CCConv/vap_switchboard",
        "fisher": "/home/erik/projects/CCConv/vap_fisher",
    },
    "audio": {
        "swb": "/home/erik/projects/data/switchboard/audio",
        "fisher": "/home/erik/projects/data/Fisher",
    },
}


def load_session(session, dset="fisher"):
    word_root = Path(DSET["word"][dset])
    df = pd.read_csv(word_root.joinpath("data", "words", f"{session}.csv"))
    vl = read_json(word_root.joinpath("data", "vad_list", f"{session}.json"))

    audio_root = Path(DSET["audio"][dset])
    audio_path = list(audio_root.rglob(f"{session}.wav"))
    if len(audio_path) > 0:
        audio_path = str(audio_path[0])
    else:
        print("Error could not find audio: ", dset, session)
        audio_path = None
    return {"df": df, "vad_list": vl, "audio_path": audio_path}


def get_utterances(df):
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
            }
            curr_speaker = sp
    if len(curr_utt["text"]) > 0:
        curr_utt["text"] = " ".join(curr_utt["text"])
        utts.append(curr_utt)
        spkrs.append(curr_speaker)
    return utts


def get_timed_words(u, offset: float = 0.0):
    words = {
        "A": {"word": [], "start": [], "end": []},
        "B": {"word": [], "start": [], "end": []},
    }
    for uu in u:
        ws = uu["text"].split()
        assert len(ws) == len(uu["starts"]), f"{ws} {uu['starts']}"
        for w, s, e in zip(ws, uu["starts"], uu["ends"]):
            words[uu["speaker"]]["word"].append(w)
            words[uu["speaker"]]["start"].append(s - offset)
            words[uu["speaker"]]["end"].append(e - offset)
    return words


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
):
    if ends is None:
        ends = [None for _ in starts]

    y_min, y_max = ax.get_ylim()
    diff = y_max - y_min
    pad = diff * 0.05

    # Plot text on top of waveform
    last_t = -1
    for ii, (word, start_time, end_time) in enumerate(zip(words, starts, ends)):
        yy = pad + y_min + diff * (ii % rows) / rows

        alignment = "left"
        if end_time is not None:
            alignment = "center"
            x_text = start_time + 0.5 * (end_time - start_time)
        else:
            x_text = start_time

        if start_time != last_t:
            ax.vlines(
                start_time,
                ymin=y_min + pad,
                ymax=y_max - pad,
                linestyle="dashed",
                linewidth=linewidth,
                color=color,
                alpha=linealpha,
            )

        last_t = end_time
        ax.text(
            x=x_text,
            y=yy,
            s=word,
            fontsize=fontsize,
            horizontalalignment=alignment,
            color=color,
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


if __name__ == "__main__":

    if "df" not in st.session_state:
        csv_path = "data/splits/fisher_swb/val_events.csv"
        st.session_state.split = "fisher_swb"
        st.session_state.df = pd.read_csv(csv_path)

    csv_path = st.selectbox("Dataset", ["fisher", "swb", "fisher_swb"], index=2)
    csv_path = CSV_PATH[csv_path]
    if csv_path != st.session_state.split:
        st.session_state.split = csv_path
        st.session_state.df = pd.read_csv(csv_path)

    c_ev, c_tfo, c_idx = st.columns(3)
    ev_type = c_ev.selectbox("Event type", ["shift", "hold", "overlap"])
    tfo = c_tfo.number_input("TFO>=", step=0.1)
    df = st.session_state.df[st.session_state.df["label"] == ev_type]
    df = df[df["tfo"] <= 3]

    df.sort_values(by="tfo", inplace=True, ascending=False)

    # df = df[df.tfo <= 5]

    if tfo > 0:
        df = df[df.tfo > tfo]

    n = len(df)
    i = c_idx.number_input(f"Index {n}", min_value=0, max_value=n - 1, value=0, step=1)

    d = df.iloc[i]

    session = Path(d["vad_path"]).stem
    dataset = "fisher" if "fisher" in d["audio_path"] else "swb"
    m = load_session(session, dataset)
    utts = get_utterances(m["df"])

    context = 5
    end_context = 5
    start = d.ipu_end - context
    if start < 0:
        context = d.ipu_end
        start = 0
    end = d.ipu_end + end_context

    x, _ = load_waveform(
        d["audio_path"], start_time=start, end_time=end, sample_rate=SAMPLE_RATE
    )
    df = m["df"]
    df = df[df.start >= start]
    df = df[df.end <= end]
    df["start"] -= start
    df["end"] -= start

    # st.write(df)
    # u = utts[:5]

    c1, c2, c3 = st.columns(3)
    c1.write(st.session_state.df)
    c2.write(d)
    c3.subheader(f"TFO: {d.tfo}s")
    c3.subheader(f"Event: {d.label.upper()}")

    # Figure
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    plot_melspectrogram(x, ax=axs[:2])
    for i, ax in enumerate(axs):
        spkr = "A" if i == 0 else "B"
        ff = df[df["speaker"] == spkr]
        c = c2 if i == 0 else c3
        plot_words_time(
            ff.word,
            starts=ff.start,
            ends=ff.end,
            ax=ax,
            rows=7,
            linealpha=0.5,
        )
        ax.axvline(context, color="r", label=d.label.upper())
        ax.axvline(context + d.tfo, color="r", linestyle="--", label="TFO")

        leg_channel = d.speaker
        leg_channel = leg_channel if d.label == "hold" else 1 - leg_channel
        if i == leg_channel:
            ax.legend(loc="upper left")
            pad = d.tfo / 10
            ax.hlines(y=11, xmin=context + pad, xmax=context + d.tfo - pad, color="r")
            ax.text(
                s="TFO",
                x=context + d.tfo / 2,
                y=13,
                color="w",
                ha="center",
                fontsize=18,
                fontweight="bold",
            )
            ax.text(
                s=d.label.upper(),
                x=context + d.tfo / 2,
                y=3,
                color="w",
                ha="center",
                fontsize=18,
            )
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    axs[-1].set_xlabel("Time (s)")
    st.pyplot(fig)
    plt.close("all")
