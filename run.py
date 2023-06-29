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


def step_extraction(
    waveform,
    model,
    device="cpu",
    context_time=20,
    step_time=5,
    vad_thresh=0.5,
    ipu_time=0.1,
    pbar=True,
    verbose=False,
):
    """
    Takes a waveform, the model, and extracts probability output in chunks with
    a specific context and step time. Concatenates the output accordingly and returns full waveform output.
    """

    n_samples = waveform.shape[-1]
    duration = round(n_samples / model.sample_rate, 2)

    chunk_time = context_time + step_time

    # Samples
    # context_samples = int(context_time * model.sample_rate)
    step_samples = int(step_time * model.sample_rate)
    chunk_samples = int(chunk_time * model.sample_rate)

    # Frames
    # context_frames = int(context_time * model.frame_hz)
    chunk_frames = int(chunk_time * model.frame_hz)
    step_frames = int(step_time * model.frame_hz)

    # Fold the waveform to get total chunks
    folds = waveform.unfold(
        dimension=-1, size=chunk_samples, step=step_samples
    ).permute(2, 0, 1, 3)
    print("folds: ", tuple(folds.shape))

    expected_frames = round(duration * model.frame_hz)
    n_folds = int((n_samples - chunk_samples) / step_samples + 1.0)
    total = (n_folds - 1) * step_samples + chunk_samples

    # First chunk
    # Use all extracted data. Does not overlap with anything prior.
    out = model.probs(folds[0].to(device))
    # OUT:
    # {
    #   "probs": probs,
    #   "vad": vad,
    #   "p_now": p_now,
    #   "p_future": p_future,
    #   "H": H,
    # }

    if pbar:
        from tqdm import tqdm

        pbar = tqdm(folds[1:], desc=f"Context: {context_time}s, step: {step_time}")
    else:
        pbar = folds[1:]
    # Iterate over all other folds
    # and add simply the new processed step
    for w in pbar:
        o = model.probs(w.to(device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -step_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -step_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -step_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -step_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -step_frames:]], dim=1)
        # out["p_zero_shot"] = torch.cat([out["p_zero_shot"], o["p_zero_shot"][:, -step_frames:]], dim=1)

    processed_frames = out["p_now"].shape[1]

    ###################################################################
    # Handle LAST SEGMENT (not included in `unfold`)
    ###################################################################
    if expected_frames != processed_frames:
        omitted_frames = expected_frames - processed_frames

        omitted_samples = model.sample_rate * omitted_frames / model.frame_hz

        if verbose:
            print(f"Expected frames {expected_frames} != {processed_frames}")
            print(f"omitted frames: {omitted_frames}")
            print(f"omitted samples: {omitted_samples}")
            print(f"chunk_samples: {chunk_samples}")

        w = waveform[..., -chunk_samples:]
        o = model.probs(w.to(device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -omitted_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -omitted_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -omitted_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -omitted_frames:]], dim=1)

    # ###################################################################
    # # Extract Vad-list over entire vad
    # ###################################################################
    # out["vad_list"] = vad_output_to_vad_list(
    #     out["vad"],
    #     frame_hz=model.frame_hz,
    #     vad_thresh=vad_thresh,
    #     ipu_thresh_time=ipu_time,
    # )
    out = batch_to_device(out, "cpu")  # to cpu for plot/save
    return out


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
    device = "cpu"
    if torch.cuda.is_available():
        model = model.to("cuda")
        device = "cuda"
    model = model.eval()

    ###########################################################
    # Load the Audio
    ###########################################################
    waveform, _ = load_waveform(args.audio, sample_rate=model.sample_rate)
    duration = round(waveform.shape[-1] / model.sample_rate)
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
        # raise NotImplementedError("step extraction not implemented")
        out = step_extraction(waveform, model, device)
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
        p_ns = out["p_now"][0].cpu()
        fig, ax = plot_stereo(
            waveform[0].cpu(), p_ns, vad, plot=False, figsize=(100, 6)
        )
        # Save figure
        figpath = args.filename.replace(".json", ".png")
        fig.savefig(figpath)
        print(f"Saved figure as {figpath}.png")
        print("Close figure to continue")
        plt.show()
