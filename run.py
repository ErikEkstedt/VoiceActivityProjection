from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from vap.modules.VAP import load_model_from_state_dict, step_extraction
from vap.modules.lightning_module import VAPModule
from vap.utils.audio import load_waveform
from vap.utils.plot import plot_stereo
from vap.utils.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)

everything_deterministic()
torch.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to waveform",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to waveform",
    )
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default=None,  # "example/checkpoints/VAP_state_dict.pt",
        help="Path to state_dict",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,  # "example/checkpoints/VAP_state_dict.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--chunk_time",
        type=float,
        default=20,
        help="Duration of each chunk processed by model (total duration of clips to evaluate)",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=5,
        help="Increment to process in a step. (Uses the last `chunk_time - step_time` as context to predict `step_time` for each relevant clip/chunk)",
    )
    parser.add_argument(
        "--force_no_chunk",
        action="store_true",
        help="Don't use chunking but process the entire audio in one pass.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Visualize output (matplotlib)"
    )
    args = parser.parse_args()

    assert Path(args.audio).exists(), f"Audio {args.audio} does not exist"
    assert (
        args.state_dict is not None or args.checkpoint is not None
    ), "Must provide state_dict or checkpoint"

    if args.state_dict:
        assert Path(
            args.state_dict
        ).exists(), f"State-dict {args.state_dict} does not exist"
    elif args.checkpoint:
        assert Path(
            args.checkpoint
        ).exists(), f"Checkpoint {args.checkpoint} does not exist"
    return args


def load_vap_model(args):
    if args.state_dict:
        model = load_model_from_state_dict(args.state_dict)
    elif args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VAPModule.load_model(args.checkpoint, map_location=device)
    else:
        raise ValueError("Must provide state_dict or checkpoint")
    return model


if __name__ == "__main__":
    args = get_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    ###########################################################
    # Load the Model
    ###########################################################
    print("Load Model...")
    model = load_vap_model(args)
    model = model.eval()

    ###########################################################
    # Load the Audio
    ###########################################################
    print("Loading Audio...")
    waveform = load_waveform(args.audio, sample_rate=model.sample_rate, mono=False)[
        0
    ].unsqueeze(0)
    duration = round(waveform.shape[-1] / model.sample_rate)

    ###########################################################
    # Model Forward
    ###########################################################
    # For consistency with training, we need to ensure that we use the
    # normal context length (default: 20s)
    print("Model Forward...")
    if duration > 20:
        print("Duration > 20: ", duration)
        if args.force_no_chunk:
            out = model.probs(waveform.to(model.device))
        else:
            out = step_extraction(
                waveform, model, chunk_time=args.chunk_time, step_time=args.step_time
            )
    else:
        out = model.probs(waveform.to(model.device))
    out = batch_to_device(out, "cpu")  # to cpu for plot/save

    ###########################################################
    # Save Output
    ###########################################################
    if args.output is None:
        args.output = "vap_output.json"
    data = tensor_dict_to_json(out)
    write_json(data, args.output)
    print("wavefile: ", args.audio)
    print("Saved output -> ", args.output)

    ###########################################################
    # Plot
    ###########################################################
    if args.plot:
        fig, ax = plot_stereo(
            waveform[0].cpu(),
            p_now=out["p_now"][0].cpu(),
            p_fut=out["p_future"][0].cpu(),
            vad=out["vad"][0].cpu(),
        )
        # Save figure
        figpath = args.output.replace(".json", ".png")
        fig.savefig(figpath)
        print(f"Saved figure as {figpath}.png")
        print("Close figure to exit")
        plt.show()
