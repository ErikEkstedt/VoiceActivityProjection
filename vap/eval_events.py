import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score

from pathlib import Path
from os.path import join
import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from vap.data.dset_event import VAPClassificationDataset
from vap.utils.utils import write_json
from vap.utils.plot import plot_melspectrogram, plot_vap_probs, plot_vad
from vap.modules.lightning_module import VAPModule, VAP, everything_deterministic

everything_deterministic()


@torch.inference_mode()
def extract_preds_and_targets(
    model: VAP,
    dloader: DataLoader,
    region_start_time: float,
    region_end_time: float,
) -> pd.DataFrame:  # -> tuple[dict[str, Tensor], Tensor]:
    def get_targets(labels):
        if isinstance(labels, str):
            labels = [labels]
        targets = []
        for lab in labels:
            targets.append(1 if lab == "shift" else 0)
        return targets

    model = model.eval()
    data = {
        k: []
        for k in ["p1", "p2", "p3", "p4", "p_now", "p_fut", "tfo", "label", "target"]
    }
    for batch in tqdm.tqdm(dloader, desc="Event classification"):
        # Model prediction
        out = model.probs(batch["waveform"].to(model.device))
        batch_preds = model.get_shift_probability(
            out, region_start_time, region_end_time, speaker=batch["speaker"]
        )
        batch_targets = get_targets(batch["label"])
        for k, v in batch_preds.items():
            data[k].extend(v)
        data["tfo"].extend(batch["tfo"].tolist())
        data["label"].extend(batch["label"])
        data["target"].extend(batch_targets)
    return pd.DataFrame(data)


def plot_output(d, out, height_ratios=[2, 2, 1, 1, 1, 1], frame_hz: int = 50):
    # Create the figure and the GridSpec instance with the given height ratios
    fig, ax = plt.subplots(
        nrows=6,
        sharex=True,
        figsize=(15, 6),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.01},
    )
    plot_melspectrogram(d["waveform"], ax=ax[:2])
    # plot vad.
    x2 = torch.arange(out["vad"].shape[1]) / frame_hz
    plot_vad(x2, out["vad"][0, :, 0], ax[0], ypad=3, color="w", label="VAD pred")
    plot_vad(x2, out["vad"][0, :, 1], ax[1], ypad=3, color="w", label="VAD pred")
    for i in range(4):
        plot_vap_probs(out["p"][i, 0], ax=ax[2 + i])
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].legend()
    ax[1].legend()
    ax[-1].set_xticks(list(range(0, 1 + round(x2[-1].item()))))  # list(range(0, 20)))
    ax[-1].set_xlabel("Time (s)")
    return fig, ax


def plot_accuracy_now_vs_fut(af, N=None, figsize=(8, 6)):
    def plot_target_lines(x, y, ax, label=None, color="k", linestyle=None, marker=None):
        ax.plot(
            [0, x], [y, y], color=color, linestyle=linestyle, alpha=0.6
        )  # horizontal
        ax.plot([x, x], [0, y], color=color, linestyle=linestyle, alpha=0.6)  # vertical
        ax.scatter(x, y, color=color, label=label, marker=marker)

    def get_best(af, acc_type="bacc"):
        a = torch.from_numpy(af[f"{acc_type}_p_now"].values)
        b = torch.from_numpy(af[f"{acc_type}_p_fut"].values)
        prefix = "BAcc"
        if acc_type != "bacc":
            prefix = acc_type[0].upper() + acc_type[1:]

        y = a.max().item()
        x = af["threshold"][a.argmax().item()]
        label = f"{prefix} ({x:.2f},{y:.2f}) (now)"
        color = "r"
        bm = b.max().item()
        if bm > y:
            y = bm
            x = af["threshold"][b.argmax().item()]
            label = f"{prefix} ({x:.2f},{y:.2f}) (fut)"
            color = "r"
        return x, y, label, color

    bacc_t, bacc, bacc_label, bacc_color = get_best(af, acc_type="bacc")
    acc_t, acc, acc_label, acc_color = get_best(af, acc_type="acc")

    fig = plt.figure(figsize=figsize)
    if N is not None:
        fig.suptitle(
            f"{N['total']} Events. SHIFTs: ({100*N['shift']:.1f}%) HOLDs: ({100*N['hold']:.1f}%)"
        )
    ax = plt.subplot()
    plot_target_lines(bacc_t, bacc, ax, label=bacc_label, color=bacc_color)
    plot_target_lines(
        acc_t, acc, ax, label=acc_label, color=acc_color, linestyle="--", marker="x"
    )
    ax.plot(
        af["threshold"],
        af["bacc_p_now"],
        label="BAcc (now)",
        color="r",
        linewidth=2,
    )
    ax.plot(
        af["threshold"],
        af["bacc_p_fut"],
        label="BAcc (fut)",
        color="g",
        linewidth=2,
    )
    ax.plot(
        af["threshold"],
        af["acc_p_now"],
        label="Acc (now)",
        linestyle="--",
        linewidth=1,
        color="r",
        alpha=0.6,
    )
    ax.plot(
        af["threshold"],
        af["acc_p_fut"],
        label="Acc (fut)",
        linestyle="--",
        linewidth=1,
        color="g",
        alpha=0.6,
    )
    if "f1w_p_now" in af.columns:
        ax.plot(af["threshold"], af["f1w_p_now"], color="darkred", label="F1w (now)")
    if "f1w_p_fut" in af.columns:
        ax.plot(
            af["threshold"],
            af["f1w_p_fut"],
            color="darkgreen",
            label="F1w (fut)",
            linestyle=":",
        )
    ax.axhline(0.5, color="k", linestyle="--")
    ax.legend()
    ax.set_ylim([0, 1])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    return fig, ax


def calculate_accuracy(df):
    pred_names = ["p_now", "p_fut", "p1", "p2", "p3", "p4"]
    targets = torch.from_numpy(df["target"].values)
    thresholds = torch.arange(0, 1.05, 0.05)
    data = []
    for t in thresholds:
        row = {"threshold": t.item()}
        for name in pred_names:
            p_guess = (torch.from_numpy(df[name].values) >= t).float()
            correct = (p_guess == targets).float()
            shift_acc = correct[targets == 1].mean().item()
            hold_acc = correct[targets == 0].mean().item()
            row[f"acc_{name}"] = correct.mean().item()
            row[f"bacc_{name}"] = (shift_acc + hold_acc) / 2
            row[f"acc_shift_{name}"] = shift_acc
            row[f"acc_hold_{name}"] = hold_acc
            row[f"f1w_{name}"] = f1_score(
                p_guess,
                targets,
                task="multiclass",
                num_classes=2,
                average="weighted",
            ).item()
        data.append(row)
    return pd.DataFrame(data)


def simple_label_stats(df: pd.DataFrame):
    n = len(df)
    n_shift = len(df[df["target"] == 1])
    n_hold = len(df[df["target"] == 0])
    return {"total": n, "shift": n_shift / n, "hold": n_hold / n}


def evaluation(args):
    """Event Evaluation"""
    # Load Model
    model: VAP = VAPModule.load_model(args.checkpoint).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    # Load Dataset
    dset = VAPClassificationDataset(
        df_path=args.csv,
        context=args.context,
        post_silence=args.post_silence,
        min_event_silence=0,
    )
    dloader = DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=False,
        pin_memory=True,
    )

    # Extract all prediction and calculate accuracy
    region_start_time = args.context + args.region_sil_pad_time
    region_end_time = region_start_time + args.region_duration

    # Columns: ['p1', 'p2', 'p3', 'p4', 'p_now', 'p_fut', 'tfo', 'label', 'target']
    df = extract_preds_and_targets(model, dloader, region_start_time, region_end_time)
    af = calculate_accuracy(df)
    label_stats = simple_label_stats(df)
    fig, ax = plot_accuracy_now_vs_fut(af, N=label_stats)

    metadata = vars(args)
    metadata["events_total"] = label_stats["total"]
    metadata["events_shift"] = label_stats["shift"]
    metadata["events_hold"] = label_stats["hold"]

    # Save Results
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    df.to_csv(join(args.output_dir, "predictions.csv"), index=False)
    af.to_csv(join(args.output_dir, "accuracy.csv"), index=False)
    fig.savefig(join(args.output_dir, "accuracy.jpeg"))
    write_json(metadata, join(args.output_dir, "metadata.json"))

    print("Saved files -> ", args.output_dir)
    print(join(args.output_dir, "predictions.csv"))
    print(join(args.output_dir, "accuracy.csv"))
    print(join(args.output_dir, "accuracy.jpeg"))
    print(join(args.output_dir, "metadata.json"))

    if args.plot:
        plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    # Evaluation arguments
    parser.add_argument("--context", type=float, default=20)
    parser.add_argument("--region_sil_pad_time", type=float, default=0.2)
    parser.add_argument("--region_duration", type=float, default=0.2)
    parser.add_argument("--post_silence", type=float, default=1.0)
    parser.add_argument("--min_event_silence", type=float, default=0)
    # DataLoader arguments
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    args = parser.parse_args()

    assert (
        args.checkpoint is not None and args.csv is not None
    ), "Must provide --checkpoint and --csv"

    if args.output_dir is None:
        args.output_dir = join(
            "data/results/classification", Path(args.checkpoint).stem
        )

    print("\nArguments")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 40)
    print()

    evaluation(args)
