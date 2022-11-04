from argparse import ArgumentParser
from pathlib import Path
from os.path import basename, join
from copy import deepcopy
from tqdm import tqdm
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.use("agg")

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


def get_scp_end_time(sample):
    target_word = EXAMPLE_TO_SCP_WORD[sample["example"]]
    end_time = -1
    for w, end in zip(sample["words"], sample["ends"]):
        if w == target_word:
            end_time = end
            break
    return end_time


def get_region_shift_probs(
    p_ns,
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
    hold = p_ns[0, :predictive_start, 1].mean()
    predictive = p_ns[0, predictive_start : last_frame - reactive_frames, 1].mean()
    reactive = p_ns[0, last_frame - reactive_frames : last_frame + 1, 1].mean()
    return hold, predictive, reactive


def create_dirs(ex, si, ge, fig_root, wav_root):
    fig_dir = join(fig_root, ex, si, ge)
    wav_dir = join(wav_root, ex, si, ge)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    Path(wav_dir).mkdir(parents=True, exist_ok=True)
    return fig_dir, wav_dir


def save_fig_and_waveform(
    sample, out, sample_name, fig_dir, wav_dir, vad_hz=50, sample_rate=16_000
):
    probs = {"p": out["p"]}
    fig, _ = plot_phrases_sample(
        sample, probs, frame_hz=vad_hz, sample_rate=sample_rate
    )
    fig.savefig(join(fig_dir, sample_name + ".png"))
    torchaudio.save(
        join(wav_dir, sample_name + ".wav"),
        sample["waveform"],
        sample_rate=sample_rate,
    )


class StatsPhraseEval(object):
    REGIONS = ["hold", "predictive", "reactive"]

    def __init__(
        self,
        permutations=[
            "regular",
            "flat_f0",
            "only_f0",
            "shift_f0",
            "flat_intensity",
            "duration_avg",
        ],
    ):
        self.permutations = permutations
        self.short_long = ["short", "long"]
        self.points = ["scp", "eot"]

        self.stats = {}
        self.data = {}
        for size in self.short_long:
            self.data[size] = {}
            for point in self.points:
                self.data[size][point] = {}
                for perm in self.permutations:
                    self.data[size][point][perm] = {}
                    for region in self.REGIONS:
                        self.data[size][point][perm][region] = []

    def __repr__(self):
        s = self.__class__.__name__
        s += str(self.data)
        return s

    def draw_completion_scores(self, completion_scores, x, ax, label=False):
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

    def plot_phrases_evaluation(self, stats=None, plot_long_eot=False):
        if stats is None:
            stats = self.stats

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        ax.set_xticks(list(range(9)))
        ax.set_xticklabels(["Hold", "Predictive", "Reactive"] * 3, fontsize=14)
        self.draw_completion_scores(
            stats["short"]["scp"], x=list(range(3)), ax=ax, label=True
        )
        self.draw_completion_scores(stats["long"]["scp"], x=list(range(3, 6)), ax=ax)
        if plot_long_eot:
            self.draw_completion_scores(
                stats["long"]["eot"], x=list(range(6, 9)), ax=ax
            )
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

        if plot_long_eot:
            ax.vlines(
                x=5.5,
                ymin=0,
                ymax=1,
                linewidth=1,
                color="k",
                linestyle="dashed",
                zorder=0,
            )
        ax.text(
            s="Short phrases\n@SCP",
            y=0.8,
            x=1,
            fontsize=16,
            fontweight="bold",
            horizontalalignment="center",
        )

        if plot_long_eot:
            ax.text(
                s="Long phrases\n@SCP     @EOT",
                y=0.3,
                x=5.5,
                fontsize=16,
                fontweight="bold",
                horizontalalignment="center",
            )
        else:
            ax.text(
                s="Long phrases\n@SCP",
                y=0.3,
                x=4,
                fontsize=16,
                fontweight="bold",
                horizontalalignment="center",
            )
        ax.set_xlim([xmin, xmax])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels([0, 25, 50, 75, 100])
        if plot_long_eot:
            ax.legend(fontsize=14, loc="upper center")
        else:
            ax.legend(fontsize=14)
        ax.set_ylabel("Shift %", fontsize=14)
        ax.set_ylim([0, 1])
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.95)
        return fig, ax

    def update(self, p, size, point, permutation, region):
        self.data[size][point][permutation][region].append(p)

    def finalize(self):
        # Condense the stats
        for size in self.short_long:
            self.stats[size] = {}
            for point in self.points:
                self.stats[size][point] = {}
                for perm in self.permutations:
                    self.stats[size][point][perm] = {}
                    for region in self.REGIONS:
                        self.stats[size][point][perm][region] = {}
                        p_list = self.data[size][point][perm][region]
                        if len(p_list) > 0:
                            # torchify stats
                            p_tensor = torch.stack(p_list)
                            self.data[size][point][perm][region] = p_tensor
                            self.stats[size][point][perm][region] = round(
                                p_tensor.mean().item(), 3
                            )


def main(args):
    """
    Evaluate model over phrases
    """
    ######################################################
    # LOAD MODEL
    ######################################################
    name = basename(args.checkpoint).replace(".ckpt", "")
    model = VAPModel.load_from_checkpoint(args.checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    ######################################################
    # LOAD DATASET
    ######################################################
    is_mono = not model.stereo
    dset = PhraseDataset(
        phrase_path=args.phrases,
        sample_rate=model.sample_rate,
        vad_hz=model.frame_hz,
        audio_mono=is_mono,
        vad=is_mono,
        vad_history=is_mono,
    )

    ######################################################
    # TRANSFORMS
    ######################################################
    transforms = {
        "flat_f0": VT.FlatPitch(sample_rate=model.sample_rate),
        "only_f0": VT.LowPass(sample_rate=model.sample_rate),
        "shift_f0": VT.ShiftPitch(sample_rate=model.sample_rate),
        "flat_intensity": VT.FlatIntensity(sample_rate=model.sample_rate),
        "duration_avg": None,
    }
    stats = StatsPhraseEval(permutations=list(transforms.keys()) + ["regular"])

    ######################################################
    # Create savepaths
    ######################################################
    root = join(args.savepath, name)
    fig_root = join(root, "figs")
    wav_root = join(root, "audio")
    Path(fig_root).mkdir(parents=True, exist_ok=True)
    Path(wav_root).mkdir(parents=True, exist_ok=True)

    total = len(dset) * len(transforms)
    pbar = tqdm(range(total), desc="Phrases evaluation (slow b/c non-batch)")
    for sample in dset:
        ex, ge, si = sample["example"], sample["gender"], sample["size"]
        sample_name = f"{ex}_{ge}_{si}_{sample['tts']}"
        fig_dir, wav_dir = create_dirs(ex, si, ge, fig_root, wav_root)

        ##################################################
        # REGULAR = 'Un-perterbed'
        # Forward Pass, Figure and waveform save
        ##################################################
        if model.stereo:
            out = model.output(waveform=sample["waveform"].to(model.device))
        else:
            vah = sample.get("vad_history", None)
            if vah is not None:
                vah = vah.to(model.device)
            out = model.output(
                waveform=sample["waveform"],
                va=sample["vad"],
                va_history=vah,
            )

        ##################################################
        # Save fig
        ##################################################
        # save_fig_and_waveform(
        #     sample,
        #     out,
        #     sample_name,
        #     fig_dir,
        #     wav_dir,
        #     vad_hz=model.frame_hz,
        #     sample_rate=model.sample_rate,
        # )

        p_hold, p_predictive, p_reactive = get_region_shift_probs(
            p_ns=out["p"],
            sample=sample,
            completion_point="scp",
            predictive_region=args.predictive_region,
        )
        stats.update(p_hold, si, "scp", "regular", "hold")
        stats.update(p_predictive, si, "scp", "regular", "predictive")
        stats.update(p_reactive, si, "scp", "regular", "reactive")
        if si == "long":  # Only long requires EOT probs
            p_eot_hold, p_eot_predictive, p_eot_reactive = get_region_shift_probs(
                p_ns=out["p"],
                sample=sample,
                completion_point="eot",
                predictive_region=args.predictive_region,
            )
            stats.update(p_eot_hold, si, "eot", "regular", "hold")
            stats.update(p_eot_predictive, si, "eot", "regular", "predictive")
            stats.update(p_eot_reactive, si, "eot", "regular", "reactive")

        for permutation, transform in transforms.items():
            if permutation == "duration_avg":
                batch = dset.sample_to_duration_sample(sample)
                batch["waveform"] = batch["waveform"].unsqueeze(1)
            else:
                batch = deepcopy(sample)
                if is_mono:
                    batch["waveform"] = batch["waveform"].unsqueeze(1)
                batch["waveform"] = transform(batch["waveform"])  # , batch['vad'])

            ##################################################
            # REGULAR = 'Un-perterbed'
            # Forward Pass, Figure and waveform save
            ##################################################
            if model.stereo:
                out = model.output(waveform=batch["waveform"].to(model.device))
            else:
                vah = batch.get("vad_history", None)
                if vah is not None:
                    vah = vah.to(model.device)
                out = model.output(
                    waveform=batch["waveform"],
                    va=batch["vad"],
                    va_history=vah,
                )
            ##################################################
            # Save fig
            ##################################################
            # save_fig_and_waveform(
            #     sample,
            #     out,
            #     sample_name,
            #     fig_dir,
            #     wav_dir,
            #     vad_hz=model.frame_hz,
            #     sample_rate=model.sample_rate,
            # )

            p_hold, p_predictive, p_reactive = get_region_shift_probs(
                out["p"],
                batch,
                completion_point="scp",
                predictive_region=args.predictive_region,
            )
            stats.update(p_hold, si, "scp", permutation, "hold")
            stats.update(p_predictive, si, "scp", permutation, "predictive")
            stats.update(p_reactive, si, "scp", permutation, "reactive")

            ##############################################
            # Save shift probs
            ##############################################
            if si == "long":
                p_eot_hold, p_eot_predictive, p_eot_reactive = get_region_shift_probs(
                    out["p"],
                    batch,
                    completion_point="eot",
                    predictive_region=args.predictive_region,
                )
                stats.update(p_eot_hold, si, "eot", permutation, "hold")
                stats.update(p_eot_predictive, si, "eot", permutation, "predictive")
                stats.update(p_eot_reactive, si, "eot", permutation, "reactive")
            pbar.update()
        plt.close("all")  # Close all plots

    stats.finalize()

    fig, _ = stats.plot_phrases_evaluation(plot_long_eot=True)
    fig.savefig(join(root, "plot_phrases_evaluation.png"))
    plt.show()
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
    stats = main(args)
