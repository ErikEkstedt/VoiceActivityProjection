"""
Show & Tell: Voice Activity Projection

Interspeech 2023

Introduce the VAP training objective
What can one do with the VAP model?
1. Next Speaker Propability Prediction
2. Turn-taking Perplexity
  - Compare dataset (Human-Robot, Human-Human, Single utterances)
3. Focus on events
  - Expectect turn-shift
  - Unexpected turn-shift
  - Ongoing speech and speaker shift prediction
  - Surprisal -> interruptions barge-in
4. Pragmatic score for generative speech
5. Change utterance to other utterance, from same speaker
  - Are future utterances less expected than the original utterance?
  - Where is this the case? Global? Local? Context?
6. Is a answer to a question more likely thatn a random utterance?
7. Button press prediction
  - sensitivity to prosody
8. Role of filler


"""

from argparse import ArgumentParser, Namespace
from glob import glob
from os.path import join
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vap.audio import load_waveform
from vap.extraction import VapExtractor
from vap.model import VapGPT, VapConfig
from vap.plot_utils import plot_melspectrogram, plot_vap_probs, plot_vad
from vap.utils import vad_list_to_onehot
from turngpt_dataset.corpus import load_corpus


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
        help="Path to state_dict",
    )
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.add_argument(
        "--context_time",
        type=float,
        default=25,
        help="Duration of each chunk processed by model",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=5,
        help="Increment to process in a step",
    )
    return parser.parse_args()


def load_vap_model(
    state_dict_path: str = "example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
):
    conf = VapConfig()
    model = VapGPT(conf)
    sd = torch.load(state_dict_path)
    model.load_state_dict(sd)

    device = "cpu"
    if torch.cuda.is_available():
        model = model.to("cuda")
        device = "cuda"

    # Set to evaluation mode
    model = model.eval()
    return model, device


def get_swb_audio_path(session, root="/home/erik/projects/data/switchboard/audio"):
    name = f"sw0{session}.wav"
    audio_path = glob(join(root, "**", name), recursive=True)
    if len(audio_path) == 0:
        print("No audio found for session: ", session)
        return None
    return audio_path[0]


def get_next_speaker_probs(logits, model):
    probs = logits.softmax(-1)  # (B, T, Classes)
    p0 = model.objective.probs_next_speaker_aggregate(probs, 0, 0).cpu()
    p1 = model.objective.probs_next_speaker_aggregate(probs, 1, 1).cpu()
    p2 = model.objective.probs_next_speaker_aggregate(probs, 2, 2).cpu()
    p3 = model.objective.probs_next_speaker_aggregate(probs, 3, 3).cpu()
    pa = model.objective.probs_next_speaker_aggregate(probs, 0, 3).cpu()
    pn = model.objective.probs_next_speaker_aggregate(probs, 0, 1).cpu()
    pf = model.objective.probs_next_speaker_aggregate(probs, 2, 3).cpu()
    return {
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p_now": pn,
        "p_fut": pf,
        "p_all": pa,
    }


def get_vad_list(d):
    vad_list = [[], []]
    for utt_starts, utt_ends, sp in zip(d["starts"], d["ends"], d["speaker"]):
        channel = 0 if sp == "A" else 1
        for s, e in zip(utt_starts, utt_ends):
            vad_list[channel].append([s, e])
    return vad_list


def plot_vap_agg_probs(x, p_all, pn, vad, frame_hz=50, fs=12, fw='bold', figsize=(12,8)):
    assert x.ndim == 2, f"x: Expected 2D input, got {x.ndim}D"
    assert p_all.ndim == 2, f"p_all: Expected 2D input, got {p_all.ndim}D"
    assert pn.ndim == 3, f"pn: Expected 3D input, got {pn.ndim}D"

    # Create a figure
    fig = plt.figure(figsize=figsize)

    # Define the GridSpec with nrows, ncols, and height_ratios
    gs = gridspec.GridSpec(nrows=7, ncols=1, height_ratios=[2, 2, 1, 1, 1, 1, 1])

    # Add the subplots to the figure
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[4, 0])
    ax6 = fig.add_subplot(gs[5, 0])
    ax7 = fig.add_subplot(gs[6, 0])
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    ###################################################
    # AUDIO
    plot_melspectrogram(x, ax=[ax1, ax2])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_ylabel("A", fontsize=fs, fontweight=fw)
    ax2.set_ylabel("B", fontsize=fs, fontweight=fw)

    # Plot vap prediction output
    if vad is not None:
        xv = torch.arange(vad.shape[0]) / frame_hz
        plot_vad(xv, vad[:, 0], ypad=2, ax=ax1, linewidth=3)
        plot_vad(xv, vad[:, 1], ypad=2, ax=ax2, linewidth=3)

    ###################################################
    # Probs
    plot_vap_probs(p_all[:, 0], ax=ax3, legend=False, alpha_ns=1.0)
    plot_vap_probs(pn[0][:, 0], ax=ax4, legend=False, alpha_ns=0.8)
    plot_vap_probs(pn[1][:, 0], ax=ax5, legend=False, alpha_ns=0.6)
    plot_vap_probs(pn[2][:, 0], ax=ax6, legend=False, alpha_ns=0.4)
    plot_vap_probs(pn[3][:, 0], ax=ax7, legend=False, alpha_ns=0.2)
    ax3.set_ylabel("P all", fontsize=fs, fontweight=fw)
    ax4.set_ylabel("P0", fontsize=fs, fontweight=fw)
    ax5.set_ylabel("P1", fontsize=fs, fontweight=fw)
    ax6.set_ylabel("P2", fontsize=fs, fontweight=fw)
    ax7.set_ylabel("P3", fontsize=fs, fontweight=fw)

    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])
    ax7.set_yticks([])

    # Add vad information
    if vad is not None:
        xv = torch.arange(vad.shape[0]) / frame_hz
        for a in [ax3, ax4, ax5, ax6, ax7]:
            xmin, xmax = a.get_xlim()
            a.fill_between(xv, y1=vad[:, 0] * xmax, y2=xmin, color="blue", alpha=0.1)
            a.fill_between(
                xv, y1=vad[:, 1] * xmax, y2=xmin, color="darkorange", alpha=0.1
            )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    return fig, axs


def plot_vap_output(p_all, pn, h, lprobs, vad, frame_hz: int = 50):
    """
    ##################################################################
    # Plot
    ##################################################################
    """
    fig, axs = plt.subplots(9, 1, sharex=True, figsize=(16, 9))
    plot_melspectrogram(x[0], ax=axs[:2])

    # Plot vap prediction output
    if vad is not None:
        xv = torch.arange(vad.shape[0]) / frame_hz
        plot_vad(xv, vad[:, 0], ypad=2, ax=axs[0])
        plot_vad(xv, vad[:, 1], ypad=2, ax=axs[1])

    plot_vap_probs(p_all[0, :, 0], ax=axs[2], legend=False, alpha_ns=0.9)
    plot_vap_probs(pn[0][0, :, 0], ax=axs[3], legend=False, alpha_ns=0.5)
    plot_vap_probs(pn[1][0, :, 0], ax=axs[4], legend=False, alpha_ns=0.4)
    plot_vap_probs(pn[2][0, :, 0], ax=axs[5], legend=False, alpha_ns=0.3)
    plot_vap_probs(pn[3][0, :, 0], ax=axs[6], legend=False, alpha_ns=0.2)
    # plot_vap_probs(p_all[0, :, 0], prob_label="P all", ax=axs[6])
    # plot_vap_probs(p_now[0, :, 0], prob_label="P now", ax=axs[7])
    # plot_vap_probs(p_fut[0, :, 0], prob_label="P fut", ax=axs[8])
    axs[2].set_ylabel("P all")
    axs[3].set_ylabel("P0")
    axs[4].set_ylabel("P1")
    axs[5].set_ylabel("P2")
    axs[6].set_ylabel("P3")

    # Likelihood
    xp = torch.arange(len(lprobs)) / frame_hz
    axs[-2].plot(
        xp, lprobs, label="P(x=y)", color="r", linewidth=3, alpha=0.6, zorder=10
    )
    # axs[-2].plot(xp, acc_t3 * 0.5, label="top 3", color="g", linewidth=2, alpha=0.6)
    # axs[-2].plot(xp, acc_t3 * 0.75, label="top 5", color="b", linewidth=2, alpha=0.6)
    axs[-2].set_ylim([0, 1])
    axs[-2].set_ylabel("Likelihood")
    axs[-2].legend(loc="upper left")

    # Entropy
    xh = torch.arange(len(h[0])) / frame_hz
    axs[-1].plot(xh, h[0], label="Entropy (bits)", color="g", linewidth=2)
    axs[-1].set_yticks([0, 4, 8])
    axs[-1].set_yticklabels([0, 4, 8])
    axs[-1].set_ylabel("Entropy")
    axs[-1].legend(loc="upper left")

    # Add vad information
    if vad is not None:
        xv = torch.arange(vad.shape[0]) / frame_hz
        for a in axs[2:]:
            xmin, xmax = a.get_xlim()
            a.fill_between(xv, y1=vad[:, 0] * xmax, y2=xmin, color="blue", alpha=0.1)
            a.fill_between(
                xv, y1=vad[:, 1] * xmax, y2=xmin, color="darkorange", alpha=0.1
            )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    return fig, axs


if __name__ == "__main__":

    args = get_args()
    model = VapExtractor(
        context_time=args.context_time,
        step_time=args.step_time,
        state_dict_path=args.state_dict,
    )
    # model, device = load_vap_model(args.state_dict)
    # dset = load_corpus("switchboard", split="test")

    dset = load_corpus(
        "fisher",
        "test",
        root="/home/erik/projects/data/Fisher/fe_03_p1_tran",
        word_level_root="/home/erik/projects/data/Fisher/fisher_word_level_montreal",
    )

    d = dset[6]

    # Get waveform and vad, extract output and likelihoods
    audio_path = get_swb_audio_path(d["session"])
    waveform, _ = load_waveform(audio_path)  # (2, n_samples)
    waveform = waveform.unsqueeze(0)  # (2, n_samples) -> (1, 2, n_samples)
    total_duration = int(waveform.shape[-1] / model.model.sample_rate)
    vad_list = get_vad_list(d)
    vad = vad_list_to_onehot(
        vad_list, duration=total_duration, frame_hz=model.model.frame_hz
    ).unsqueeze(0)
    y = model.model.extract_labels(vad.to(model.device)).cpu()
    out = model.extract(waveform)

    n_label_frames = y.shape[-1]

    likelihood_probs = out["probs"][0][torch.arange(n_label_frames), y]

    _, idx3 = out["probs"][0, :n_label_frames].topk(3, dim=-1)
    _, idx5 = out["probs"][0, :n_label_frames].topk(5, dim=-1)
    top3_acc = (y[0].unsqueeze(-1) == idx3).sum(-1)
    top5_acc = (y[0].unsqueeze(-1) == idx5).sum(-1)
    sample_rate = model.model.sample_rate frame_hz = model.model.frame_hz
    duration = 10
    start_time = 0
    for start_time in range(0, total_duration, 3 * duration // 4):
        end_time = start_time + duration
        print(start_time, end_time)
        # Get the corresponding data
        s = int(sample_rate * start_time)
        e = int(sample_rate * end_time)
        sf = int(frame_hz * start_time)
        ef = int(frame_hz * end_time)
        x = waveform[0, :, s:e]
        v = vad[0, sf:ef]
        pn = out["p"][:, 0, sf:ef]
        pa = out["p_all"][0, sf:ef]
        # h = out["H"][:, sf:ef]
        # probs = out["probs"][0, sf:ef]
        # lprobs = likelihood_probs[0, sf:ef]
        # acc_t3 = top3_acc[sf:ef]
        # acc_t5 = top5_acc[sf:ef]
        # fig, ax = plot_vap_output(p_all=pa, pn=pn, h=h, lprobs=lprobs, vad=v)
        fig, ax = plot_vap_agg_probs(x, p_all=pa, pn=pn, vad=v)
        plt.show()



    duration = 5
    # shift
    start_time = 6.8
    # bc
    # start_time = 132
    # overlap resolve
    # start_time = 111
    # Hold
    # start_time = 64
    end_time = start_time + duration
    s = int(sample_rate * start_time)
    e = int(sample_rate * end_time)
    sf = int(frame_hz * start_time)
    ef = int(frame_hz * end_time)
    x = waveform[0, :, s:e]
    v = vad[0, sf:ef]
    pn = out["p"][:, 0, sf:ef]
    pa = out["p_all"][0, sf:ef]
    # h = out["H"][:, sf:ef]
    # probs = out["probs"][0, sf:ef]
    # lprobs = likelihood_probs[0, sf:ef]
    # acc_t3 = top3_acc[sf:ef]
    # acc_t5 = top5_acc[sf:ef]
    # fig, ax = plot_vap_output(p_all=pa, pn=pn, h=h, lprobs=lprobs, vad=v)
    fig, ax = plot_vap_agg_probs(x, p_all=pa, pn=pn, vad=v, fs=28, fw=None, figsize=(9,9))
    plt.show()
