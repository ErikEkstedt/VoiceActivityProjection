import torch
import torchaudio
from argparse import ArgumentParser
from os.path import basename
from tqdm import tqdm
import matplotlib.pyplot as plt

from vap.model import VAPModel
from vap.utils import (
    load_sample,
    batch_to_device,
    everything_deterministic,
    vad_output_to_vad_list,
    tensor_dict_to_json,
    write_json,
)
from vap.plot_utils import plot_stereo


everything_deterministic()

torch.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
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
        "--vad_thresh",
        type=float,
        default=0.5,
        help="Probability to consider a frame as active (Voice Activity Detection)",
    )
    parser.add_argument(
        "--ipu_thresh_time",
        type=float,
        default=0.1,
        help="Time to consider two consecutive VAD segments, from the same speaker, as one",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="output filename to save output",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Visualize output (matplotlib)"
    )
    parser.add_argument("--full", action="store_true", help="Save all information")
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="copy the audio",
    )
    ###########################################################
    # Chunk times
    ###########################################################
    parser.add_argument(
        "--context_time",
        type=float,
        default=20,
        help="Process the audio in chunks (longer > 164s on 24Gb GPU audio)",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=5,
        help="Increment to process in a step",
    )
    ###########################################################
    # Chunk times
    ###########################################################
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
        "--chunk_overlap_time",
        type=float,
        default=20,
        help="Duration of overlap between each chunk processed by model",
    )
    args = parser.parse_args()
    return args


def step_extraction(
    waveform, model, context_time=20, step_time=5, vad_thresh=0.5, ipu_time=0.1
):
    chunk_time = context_time + step_time

    # context_samples = int(context_time * model.sample_rate)
    step_samples = int(step_time * model.sample_rate)
    chunk_samples = int(chunk_time * model.sample_rate)

    # context_frames = int(context_time * model.frame_hz)
    chunk_frames = int(chunk_time * model.frame_hz)
    step_frames = int(step_time * model.frame_hz)

    n_samples = waveform.shape[-1]
    duration = round(n_samples / model.sample_rate, 2)

    # Fold the waveform to get total chunks
    folds = waveform.unfold(
        dimension=-1, size=chunk_samples, step=step_samples
    ).permute(2, 0, 1, 3)

    expected_frames = round(duration * model.frame_hz)
    n_folds = int((n_samples - chunk_samples) / step_samples + 1.0)
    total = (n_folds - 1) * step_samples + chunk_samples

    # First chunk
    # Use all extracted data. Does not overlap with anything prior.
    out = model.probs(folds[0])
    # OUT:
    # {
    #   "probs": probs,
    #   "vad": vad,
    #   "p_bc": p_bc,
    #   "p_now": p_now,
    #   "p_future": p_future,
    #   "H": H,
    # }

    # Iterate over all other folds
    # and add simply the new processed step
    for w in tqdm(
        folds[1:], desc=f"Context: {args.context_time}s, step: {args.step_time}"
    ):
        o = model.probs(w)
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -step_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -step_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -step_frames:]], dim=1
        )
        out["p_bc"] = torch.cat([out["p_bc"], o["p_bc"][:, -step_frames:]], dim=1)
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
        print(f"Expected frames {expected_frames} != {processed_frames}")
        print(f"omitted frames: {omitted_frames}")
        print(f"omitted samples: {omitted_samples}")
        print(f"chunk_samples: {chunk_samples}")

        w = waveform[..., -chunk_samples:]
        o = model.probs(w)
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -omitted_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -omitted_frames:]], dim=1
        )
        out["p_bc"] = torch.cat([out["p_bc"], o["p_bc"][:, -omitted_frames:]], dim=1)
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -omitted_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -omitted_frames:]], dim=1)

    ###################################################################
    # Extract Vad-list over entire vad
    ###################################################################
    out["vad_list"] = vad_output_to_vad_list(
        out["vad"],
        frame_hz=model.frame_hz,
        vad_thresh=vad_thresh,
        ipu_thresh_time=ipu_time,
    )
    out = batch_to_device(out, "cpu")  # to cpu for plot/save
    return out


if __name__ == "__main__":
    args = get_args()

    ###########################################################
    # Load the model
    ###########################################################
    print("Load Model...")
    model = VAPModel.load_from_checkpoint(args.checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    ###########################################################
    # Load the Audio
    ###########################################################
    waveform, _ = load_sample(
        path=args.wav,
        sample_rate=model.sample_rate,
        frame_hz=model.frame_hz,
        force_stereo=True,
        noise_scale=0,
        device=model.device,
    )
    duration = round(waveform.shape[-1] / model.sample_rate, 2)

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
        out = step_extraction(
            waveform,
            model,  # , context_time=args.context_time, step_time=args.step_time
        )
    else:
        out = model.probs(waveform=waveform)
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

    savepath = args.filename
    if args.filename is None:
        savepath = basename(args.wav).replace(".wav", ".json")

    if not savepath.endswith(".json"):
        savepath += ".json"

    data = tensor_dict_to_json(out)
    write_json(data, savepath)
    print("wavefile: ", args.wav)
    print("Saved output -> ", savepath)

    if args.save_audio:
        wpath = savepath.replace(".json", ".wav")
        torchaudio.save(wpath, waveform[0].cpu(), sample_rate=model.sample_rate)
        print("Saved audio -> ", wpath)

    ###########################################################
    # Plot
    ###########################################################
    if args.plot:
        wav = waveform[0].cpu()
        vad = out["vad"][0].cpu()
        p_ns = out["p"][0, :, 0].cpu()
        p_bc = out["p_bc"][0].cpu()
        fig, ax = plot_stereo(wav, p_ns, vad, plot=False)
        # Save figure
        fig.savefig(f"{savepath}.png")
        print(f"Saved figure as {savepath}.png")
        print("Close figure to continue")
        plt.show()
