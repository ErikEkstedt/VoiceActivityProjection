from argparse import ArgumentParser
from pathlib import Path
from os.path import basename, join
from copy import deepcopy
from tqdm import tqdm
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("agg")

import vap.transforms as VT
from vap.model import VAPModel
from vap.phrases.dataset import PhraseDataset
from vap.plot_utils import plot_phrases_sample


EXAMPLE_TO_SCP_WORD = {
    "student": "student",
    "psychology": "psychology",
    "first_year": "student",
    "basketball": "basketball",
    "experiment": "before",
    "live": "yourself",
    "work": "side",
    "bike": "bike",
    "drive": "here",
}


def plot_phrases_evaluation(stats):
    def draw_completion_scores(completion_scores, x, ax, label=False):
        colors = {
            "regular": "k",
            "flat_f0": "g",
            "only_f0": "y",
            "flat_intensity": "r",
            "shift_f0": "gray",
            "duration_avg": "b",
        }
        for permutation, region_scores in completion_scores.items():
            y = [
                region_scores["hold"],
                region_scores["predictive"],
                region_scores["reactive"],
            ]
            ls = None
            alpha = 0.6
            zorder = None
            if permutation == "only_f0":
                ls = "dashed"
            if permutation == "regular":
                alpha = 1
                zorder = 100
            ax.plot(
                x,
                y,
                alpha=alpha,
                linewidth=4,
                linestyle=ls,
                color=colors[permutation],
                zorder=zorder,
            )

            _label = None
            if label:
                _label = permutation
            ax.scatter(
                x,
                y,
                s=100,
                alpha=alpha,
                color=colors[permutation],
                label=_label,
                zorder=zorder,
            )

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_xticks(list(range(9)))
    ax.set_xticklabels(["Hold", "Predictive", "Reactive"] * 3, fontsize=14)
    draw_completion_scores(stats["short"]["scp"], x=list(range(3)), ax=ax, label=True)
    draw_completion_scores(stats["long"]["scp"], x=list(range(3, 6)), ax=ax)
    # draw_completion_scores(stats["long"]["eot"], x=list(range(6, 9)), ax=ax)
    xmin, xmax = ax.get_xlim()
    ax.hlines(
        y=0.5,
        xmin=xmin,
        xmax=xmax,
        linewidth=2,
        linestyle="dashed",
        color="k",
        zorder=0,
    )
    ax.vlines(
        x=2.5,
        ymin=0,
        ymax=1,
        linewidth=2,
        color="k",
        zorder=0,
    )
    ax.text(
        s="Short phrases",
        y=0.1,
        x=1,
        fontsize=16,
        fontweight="bold",
        horizontalalignment="center",
    )
    ax.text(
        s="Long phrases",
        y=0.1,
        x=4,
        fontsize=16,
        fontweight="bold",
        horizontalalignment="center",
    )
    ax.set_xlim([xmin, xmax])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.legend(fontsize=14)
    ax.set_ylabel("Shift %", fontsize=14)
    ax.set_ylim([0, 1])
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.95)
    return fig, ax


def get_scp_end_time(sample):
    target_word = EXAMPLE_TO_SCP_WORD[sample["example"]]
    end_time = -1
    for w, end in zip(sample["words"], sample["ends"]):
        if w == target_word:
            end_time = end
            break
    return end_time


def get_region_shift_probs(
    probs,
    sample,
    predictive_region: float = 0.2,
    reactive_frames: int = 2,
    completion_point: str = "scp",
    frame_hz: int = 50,
):
    if completion_point.lower() == "scp":
        completion_point_time = get_scp_end_time(sample)
    else:  # simply the last word end time
        completion_point_time = sample["ends"][-1]

    last_frame = round(completion_point_time * frame_hz)
    pre_frames = round(predictive_region * frame_hz)
    predictive_start = last_frame - pre_frames

    # In phrases dataset the current speaker is always A (=0)
    # So the shift probability becomes the value for speaker B (=1)
    hold = probs["p"][0, :predictive_start, 1].mean()
    predictive = probs["p"][
        0, predictive_start : last_frame - reactive_frames, 1
    ].mean()
    reactive = probs["p"][0, last_frame - reactive_frames : last_frame + 1, 1].mean()
    return hold, predictive, reactive


def evaluation_phrases(args):
    # Load Model
    name = basename(args.checkpoint).replace(".ckpt", "")
    model = VAPModel.load_from_checkpoint(args.checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Load Dataset
    is_mono = not model.stereo
    dset = PhraseDataset(
        phrase_path=args.phrases,
        sample_rate=model.sample_rate,
        vad_hz=model.frame_hz,
        audio_mono=is_mono,
        vad=is_mono,
        vad_history=is_mono,
    )

    # Create savepaths
    root = join(args.savepath, name)
    fig_root = join(root, "figs")
    wav_root = join(root, "audio")
    Path(fig_root).mkdir(parents=True, exist_ok=True)
    Path(wav_root).mkdir(parents=True, exist_ok=True)

    # Transforms
    transforms = {
        "flat_f0": VT.FlatPitch(sample_rate=model.sample_rate),
        "only_f0": VT.LowPass(sample_rate=model.sample_rate),
        "shift_f0": VT.ShiftPitch(sample_rate=model.sample_rate),
        "flat_intensity": VT.FlatIntensity(sample_rate=model.sample_rate),
    }

    stats = {}
    total = len(dset) * len(transforms)
    pbar = tqdm(range(total), desc="Phrases evaluation (slow b/c non-batch)")
    for sample in dset:
        ex, ge, si = sample["example"], sample["gender"], sample["size"]

        sample_name = f"{ex}_{ge}_{si}_{sample['tts']}"
        fig_dir = join(fig_root, ex, si, ge)
        wav_dir = join(wav_root, ex, si, ge)
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        Path(wav_dir).mkdir(parents=True, exist_ok=True)

        # Statistics
        if si not in stats:
            stats[si] = {}  # short, long

        # Regular
        # Forward Pass, Figure and waveform save
        _, _, probs, batch = model.output(sample)
        fig, _ = plot_phrases_sample(
            sample, probs, frame_hz=dset.vad_hz, sample_rate=dset.sample_rate
        )
        fig.savefig(join(fig_dir, sample_name + ".png"))
        torchaudio.save(
            join(wav_dir, name + ".wav"),
            sample["waveform"],
            sample_rate=model.sample_rate,
        )

        p_hold, p_predictive, p_reactive = get_region_shift_probs(
            probs,
            sample,
            completion_point="scp",
            predictive_region=args.predictive_region,
        )

        ##############################################
        # Save shift probs
        ##############################################
        if "scp" not in stats[si]:
            stats[si]["scp"] = {}

        if "regular" not in stats[si]["scp"]:
            stats[si]["scp"]["regular"] = {"hold": [], "reactive": [], "predictive": []}

        stats[si]["scp"]["regular"]["hold"].append(p_hold)
        stats[si]["scp"]["regular"]["predictive"].append(p_predictive)
        stats[si]["scp"]["regular"]["reactive"].append(p_reactive)

        if si == "long":
            if "eot" not in stats[si]:
                stats[si]["eot"] = {}

            if "regular" not in stats[si]["eot"]:
                stats[si]["eot"]["regular"] = {
                    "hold": [],
                    "reactive": [],
                    "predictive": [],
                }

            p_hold, p_predictive, p_reactive = get_region_shift_probs(
                probs,
                sample,
                completion_point="eot",
                predictive_region=args.predictive_region,
            )
            stats[si]["eot"]["regular"]["hold"].append(p_hold)
            stats[si]["eot"]["regular"]["predictive"].append(p_predictive)
            stats[si]["eot"]["regular"]["reactive"].append(p_reactive)

        # Average duration
        for permutation, transform in transforms.items():
            batch = deepcopy(sample)
            if is_mono:
                batch["waveform"] = batch["waveform"].unsqueeze(1)
            batch["waveform"] = transform(batch["waveform"])  # , batch['vad'])

            # Forward Pass, Figure and waveform save
            _, _, probs, batch = model.output(batch)
            fig, _ = plot_phrases_sample(
                batch, probs, frame_hz=dset.vad_hz, sample_rate=dset.sample_rate
            )
            fig.savefig(join(fig_dir, sample_name + f"_{permutation}.png"))

            # save waveform
            torchaudio.save(
                join(wav_dir, name + f"_{permutation}.wav"),
                sample["waveform"],
                sample_rate=model.sample_rate,
            )

            if permutation not in stats[si]:
                stats[si]["scp"][permutation] = {
                    "hold": [],
                    "reactive": [],
                    "predictive": [],
                }
            p_hold, p_predictive, p_reactive = get_region_shift_probs(
                probs,
                sample,
                completion_point="scp",
                predictive_region=args.predictive_region,
            )

            ##############################################
            # Save shift probs
            ##############################################
            stats[si]["scp"][permutation]["hold"].append(p_hold)
            stats[si]["scp"][permutation]["predictive"].append(p_predictive)
            stats[si]["scp"][permutation]["reactive"].append(p_reactive)
            if si == "long":
                if permutation not in stats[si]["eot"]:
                    stats[si]["eot"][permutation] = {
                        "hold": [],
                        "reactive": [],
                        "predictive": [],
                    }
                p_hold, p_predictive, p_reactive = get_region_shift_probs(
                    probs,
                    sample,
                    completion_point="eot",
                    predictive_region=args.predictive_region,
                )
                stats[si]["eot"][permutation]["hold"].append(p_hold)
                stats[si]["eot"][permutation]["predictive"].append(p_predictive)
                stats[si]["eot"][permutation]["reactive"].append(p_reactive)

            pbar.update()
        plt.close("all")  # Close all plots

    # Condense the stats
    for short_long, completion_scores in stats.items():
        for completion_point, perm_scores in completion_scores.items():
            for perm, score_dict in perm_scores.items():
                stats[short_long][completion_point][perm]["hold"] = round(
                    torch.stack(score_dict["hold"]).mean().item(), 3
                )
                stats[short_long][completion_point][perm]["predictive"] = round(
                    torch.stack(score_dict["predictive"]).mean().item(), 3
                )
                stats[short_long][completion_point][perm]["reactive"] = round(
                    torch.stack(score_dict["reactive"]).mean().item(), 3
                )

    fig, _ = plot_phrases_evaluation(stats)
    fig.savefig(join(root, "plot_phrases_evaluation.png"))
    return stats


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument(
        "--phrases",
        type=str,
        default="dataset_phrases/phrases.json",
        help="Path (relative) to phrase-dataset or directly to the 'phrases.json' file used in 'PhraseDataset'",
    )
    parser.add_argument(
        "--predictive_region",
        type=float,
        default=0.2,
        help="Path to results directory",
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="runs_evaluation/phrases",
        help="Path to results directory",
    )
    args = parser.parse_args()
    stats = evaluation_phrases(args)
