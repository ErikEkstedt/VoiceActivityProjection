from argparse import ArgumentParser
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

mpl.use("tkagg")

from vap.model import VAPModel
from vap.utils import everything_deterministic, read_json, write_json

from vap.plot_utils import plot_phrases_sample

everything_deterministic()


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="example/50hz_48_10s-epoch20-val_1.85.ckpt",
        help="Path to trained model",
    )
    parser.add_argument(
        "-w",
        "--wav",
        type=str,
        default="example/student_long_female_en-US-Wavenet-G.wav",
        help="Path to waveform",
    )
    parser.add_argument(
        "-v",
        "--vad_list",
        type=str,
        default="example/student_long_female_en-US-Wavenet-G_vad_list.json",
        help="The voice activity see `example/student_long_female_en-US-Wavenet-G_vad_list.json` for format",
    )
    parser.add_argument(
        "-o",
        "--savepath",
        type=str,
        default="vap_output.json",
        help="The path of the output file: `PATH/TO/filename.json",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Visualize output (matplotlib)"
    )
    parser.add_argument("--full", action="store_true", help="Save all information")
    args = parser.parse_args()
    return args


def serialize_sample(sample):
    return {"vad": sample["vad"].tolist(), "waveform": sample["waveform"].tolist()}


def save_model_output(loss, out, probs, savepath, full=False):
    data = {
        "loss": loss["loss"].item(),
        "labels": out["va_labels"].tolist(),
        "p": probs["p"].tolist(),
        "p_bc": probs["bc_prediction"].tolist(),
        "model": {
            "sample_rate": model.sample_rate,
            "frame_hz": model.frame_hz,
            "checkpoint": args.checkpoint,
        },
        "va": vad_list,
    }

    if full:
        print("Adding full information i.e. loss-frames and all state probabilitities")
        data["loss_frames"] = loss["loss_frames"].tolist()
        data["probs"] = out["logits"].softmax(-1).tolist()

    write_json(data, savepath)
    print("Wrote output -> ", savepath)


if __name__ == "__main__":

    args = get_args()
    print("-" * 40)
    print("Arguments")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 40)
    print()

    print("Load Model...")
    model = VAPModel.load_from_checkpoint(args.checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # get sample and process
    vad_list = read_json(args.vad_list)
    sample = model.load_sample(args.wav, vad_list)
    loss, out, probs, sample = model.output(sample)

    save_model_output(loss, out, probs, savepath=args.savepath, full=args.full)

    print("loss: ", loss.keys())
    if args.plot:
        print("PLOT THE RESULT")
        fig, _ = plot_phrases_sample(
            sample, probs, frame_hz=model.frame_hz, sample_rate=model.sample_rate
        )
        plt.show()
        input()
