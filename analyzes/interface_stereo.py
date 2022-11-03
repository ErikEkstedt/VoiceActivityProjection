import streamlit as st
from argparse import ArgumentParser

import torch
from vap.model import VAPModel
from vap.utils import everything_deterministic, batch_to_device
from vap.plot_utils import plot_stereo
from datasets_turntaking import DialogAudioDM

# CHECKPOINT = "example/VAP_50Hz_ad20s_134-epoch13-val_2.49.ckpt"
# CHECKPOINT = "example/VAP_50Hz_ad20s_118_old-epoch41-val_2.43.ckpt"
CHECKPOINT = "example/VAP_50Hz_ad20s_118-epoch19-val_2.47.ckpt"

# Reproducability
everything_deterministic()
torch.manual_seed(0)


@st.cache
def load_model(checkpoint=CHECKPOINT):
    model = VAPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    return model


@st.cache
def load_dset(conf):
    dm = DialogAudioDM(
        datasets=["switchboard"],
        audio_duration=10,
        audio_overlap=2,
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
    print(batch.keys())

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
        model = load_model()
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
