import streamlit as st
from argparse import ArgumentParser
import os

import torch
from vap.model import VAPModel
from vap.utils import everything_deterministic, batch_to_device
from vap.plot_utils import plot_stereo
from vap.phrases.dataset import PhraseDataset

everything_deterministic()
torch.manual_seed(0)

parser = ArgumentParser()
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default="example/VAP_ges0x55b_50Hz_ad20s_134-epoch4-val_2.70.ckpt",
)
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)


@st.cache
def load_model(checkpoint):
    model = VAPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    return model


def get_figure(experiment, longshort, gender, idx):
    idx -= 1  # Offset 0
    model = st.session_state.model

    # get sample
    batch = st.session_state.dset.get_sample(experiment, longshort, gender, idx)
    batch = batch_to_device(batch, model.device)

    # Zero pad
    # zpad = torch.randn_like(batch["waveform"]) * 0.01
    zpad = torch.zeros_like(batch["waveform"])
    waveform = torch.stack((batch["waveform"], zpad), dim=1)
    out = model.output(waveform=waveform)  # , max_time=30)
    fig, _ = plot_stereo(
        waveform=waveform[0].cpu(),
        p_ns=out["p"][0, :, 0].cpu(),
        vad=out["vad"][0].cpu(),
        plot=False,
    )
    return fig


if __name__ == "__main__":
    if "model" not in st.session_state:
        model = load_model(args.checkpoint)
        if torch.cuda.is_available():
            model = model.to("cuda")
        st.session_state.model = model

    if "dset" not in st.session_state:
        st.session_state.dset = PhraseDataset(
            phrase_path="dataset_phrases/phrases.json"
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

    # st.write("experiment: ", experiment)
    # st.write("longshort: ", longshort)
    # st.write("gender: ", gender)
    # st.write("idx: ", idx)

    fig = get_figure(experiment, longshort, gender, idx)
    st.pyplot(fig)
