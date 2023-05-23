import streamlit as st
from argparse import ArgumentParser
import os

import torch
from vap.model import load_model
from vap.utils import everything_deterministic, batch_to_device
from vap.plot_utils import plot_melspectrogram, plot_vap_probs
from vap.phrases.dataset import PhraseDataset

import matplotlib.pyplot as plt


def plot_single_channel_vap(waveform, p, vad, words=None, starts=None, ends=None):
    assert (
        waveform.ndim == 2
    ), f"Expected waveform to be (n_samples,) but got {waveform.shape}"
    assert (
        waveform.shape[0] == 1
    ), f"Expected waveform to be (1, n_samples) but got {waveform.shape}"
    assert p.ndim == 2, f"Expected p to be (4, n_frames) but got {p.shape}"

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(12, 8))
    plot_melspectrogram(waveform, ax=ax[0])
    for i, name in enumerate(["P1", "P2", "P3", "P4"]):
        plot_vap_probs(p[i], ax=ax[i + 1], prob_label=name)
    return fig


everything_deterministic()
torch.manual_seed(0)

parser = ArgumentParser()
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
)
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)


@st.cache_resource
def load_model_cache(checkpoint):
    model = load_model(checkpoint)
    model = model.eval()
    return model


def get_figure(experiment, longshort, gender, idx):
    idx -= 1  # Offset 0
    model = st.session_state.model
    dset = st.session_state.dset

    # get sample
    sample = dset.get_sample(experiment, longshort, gender, idx)
    batch = dset.sample_to_output(sample)
    batch = batch_to_device(batch, model.device)

    name = os.path.basename(sample["audio_path"])

    # Zero pad
    waveform = batch["waveform"]
    with torch.inference_mode():
        out = model.probs(waveform=waveform)  # , max_time=30)
    fig = plot_single_channel_vap(
        waveform[0, :1].cpu(), p=out["p"][:, 0].cpu(), vad=out["vad"][0].cpu()
    )
    return fig, name


if __name__ == "__main__":

    if "model" not in st.session_state:
        model = load_model_cache(args.checkpoint)
        st.session_state.model = model

    if "dset" not in st.session_state:
        st.session_state.dset = PhraseDataset(
            csv_path="dataset_phrases/phrases.csv", audio_mono=False
        )

    st.title("VAP Phrases")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    experiment = c1.selectbox(
        "Pick one",
        [
            "psychology",
            "student",
            "first_year",
            "basketball",
            "experiment",
            "live",
            "work",
            "drive",
            "bike",
        ],
    )
    longshort = c2.selectbox("Pick one", ["short", "long"])
    gender = c3.selectbox("Pick one", ["female", "male"])
    idx = c4.number_input(f"sample idx (max: 5)", 1, 5)

    if "model" in st.session_state:
        fig, name = get_figure(experiment, longshort, gender, idx)
        st.subheader(name)
        st.pyplot(fig)
