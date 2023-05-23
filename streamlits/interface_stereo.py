import streamlit as st
from argparse import ArgumentParser
import os

import torch
from vap.model import load_model
from vap.utils import everything_deterministic, batch_to_device
from vap.plot_utils import plot_stereo
from datasets_turntaking import DialogAudioDM

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


# Reproducability
everything_deterministic()
torch.manual_seed(0)


@st.cache_resource
def load_model_cache(checkpoint):
    model = load_model(checkpoint)
    model = model.eval()
    return model


@st.cache
def load_dset(conf):
    dm = DialogAudioDM(
        datasets=["switchboard"],
        audio_duration=20,
        audio_overlap=2,
        type="events",
        vad_history=conf["model"]["va_cond"]["history"],
        audio_mono=False,
        batch_size=1,
        num_workers=0,
        mask_vad=True,
        mask_vad_probability=1.0,
    )
    dm.prepare_data()
    dm.setup()
    return dm.val_dset


def get_figure(idx=0):
    model = st.session_state.model
    dset = st.session_state.dset

    # get sample
    batch = dset[idx]
    batch = batch_to_device(batch, model.device)

    out = model.output(waveform=batch["waveform"])  # , max_time=30)
    # print(out.keys())
    fig, ax = plot_stereo(
        waveform=batch["waveform"][0].cpu(),
        p_ns=out["p"][0, :, 0].cpu(),
        vad=batch["vad"][0].cpu(),
        plot=False,
    )

    # print("Shifts: ", out["shift"])
    # print("Shorts: ", out["short"])
    if len(out["shift"][0]) > 0:
        for start, end, speaker in out["shift"][0]:
            # color = "b" if speaker == 0 else "orange"
            # ax[-1].axvline(x=start, color=color, linewidth=4)
            ax[-1].axvline(x=start, color="r", linewidth=2)

    if len(out["short"][0]) > 0:
        for start, end, speaker in out["short"][0]:
            color = "orange" if speaker == 0 else "b"
            ax[-1].axvline(x=start, color=color, linewidth=4)
            # ax[-1].axvline(x=start, color="green", linewidth=4)

    return fig


if __name__ == "__main__":

    if "model" not in st.session_state:
        print("ARGUMENT: ", args)
        model = load_model(args.checkpoint)
        if torch.cuda.is_available():
            model = model.to("cuda")
        st.session_state.model = model

    if "dset" not in st.session_state:
        st.session_state.dset = load_dset(st.session_state.model.conf)

    st.title("VAP Stereo")

    idx = st.number_input(
        f"dataset idx (max: {len(st.session_state.dset)})",
        0,
        len(st.session_state.dset),
    )

    fig = get_figure(idx)
    st.pyplot(fig)
