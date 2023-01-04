from argparse import ArgumentParser
from os.path import basename
import matplotlib.pyplot as plt
import torch
import torchaudio


from vap.model import VapGPT, VapConfig, load_older_state_dict
from vap.audio import load_waveform
from vap.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)
from vap.plot_utils import plot_stereo


everything_deterministic()
torch.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to waveform",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Path to waveform",
    )
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
        help="Path to state_dict",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model",
    )
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Process the audio in chunks (longer > 164s on 24Gb GPU audio)",
    )
    parser.add_argument(
        "--chunk_time",
        type=float,
        default=30,
        help="Duration of each chunk processed by model",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=5,
        help="Increment to process in a step",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Visualize output (matplotlib)"
    )
    args = parser.parse_args()

    conf = VapConfig.args_to_conf(args)
    return args, conf


if __name__ == "__main__":
    args, conf = get_args()

    ###########################################################
    # Load the model
    ###########################################################
    print("Load Model...")
    if args.checkpoint is None:
        print("From state-dict: ", args.state_dict)
        model = VapGPT(conf)
        sd = torch.load(args.state_dict)
        model.load_state_dict(sd)
    else:
        from vap.train import VAPModel

        print("From Lightning checkpoint: ", args.checkpoint)
        raise NotImplementedError("Not implemeted from checkpoint...")
        # model = VAPModel.load_from_checkpoint(args.checkpoint)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = model.eval()

    ###########################################################
    # Load the Audio
    ###########################################################
    waveform, _ = load_waveform(args.audio, sample_rate=model.sample_rate)
    duration = round(waveform.shape[-1] / model.sample_rate, 2)
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, torch.zeros_like(waveform)))
    waveform = waveform.unsqueeze(0)

    # Maximum known duration with a 24Gb 'NVIDIA GeForce RTX 3090' is 164s
    if duration > 160:
        print(
            f"WARNING: Can't fit {duration} > 160s on 24Gb 'NVIDIA GeForce RTX 3090' GPU"
        )
        print("WARNING: Change code if this is not what you want.")
        args.chunk = True

    ###########################################################
    # Model Forward
    ###########################################################
    if args.chunk:
        raise NotImplementedError("step extraction not implemented")
        # out = step_extraction(waveform, model)
    else:
        if torch.cuda.is_available():
            waveform = waveform.to("cuda")
        out = model.probs(waveform)
        out = batch_to_device(out, "cpu")  # to cpu for plot/save

    ###########################################################
    # Print shapes
    ###########################################################
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: ", tuple(v.shape))

    ###########################################################
    # Save Output
    ###########################################################
    if args.filename is None:
        args.filename = basename(args.audio).replace(".wav", ".json")

    if not args.filename.endswith(".json"):
        args.filename += ".json"

    data = tensor_dict_to_json(out)
    write_json(data, args.filename)
    print("wavefile: ", args.audio)
    print("Saved output -> ", args.filename)

    ###########################################################
    # Plot
    ###########################################################
    if args.plot:
        print(out.keys())
        vad = out["vad"][0].cpu()
        p_ns = out["p_now"][0, :, 0].cpu()
        # p_bc = out["p_bc"][0].cpu()
        fig, ax = plot_stereo(waveform[0].cpu(), p_ns, vad, plot=False)
        # Save figure
        figpath = args.filename.replace(".json", ".png")
        fig.savefig(figpath)
        print(f"Saved figure as {figpath}.png")
        print("Close figure to continue")
        plt.show()
