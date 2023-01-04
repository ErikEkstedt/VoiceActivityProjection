import torch
import torchaudio
from argparse import ArgumentParser
from os.path import basename
import matplotlib.pyplot as plt

from vap.model import VapGPT
from vap.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)
from vap.plot_utils import plot_stereo


everything_deterministic()

torch.manual_seed(0)


# TODO: Fix this entire script to work with new model
def load_sample(
    path: str,
    vad_list_path: Optional[str] = None,
    sample_rate: int = 16000,
    frame_hz=50,
    noise_scale: float = 0.0,
    force_stereo: bool = True,
    device: Optional[str] = None,
):
    waveform, _ = load_waveform(path, sample_rate=sample_rate)

    # Add channel with minimal noise as second channel
    if force_stereo and waveform.ndim == 2 and waveform.shape[0] == 1:
        z = torch.randn_like(waveform) * noise_scale
        # waveform = torch.stack((waveform, z), dim=1)
        waveform = torch.stack((waveform, z), dim=1)

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    vad = None
    if vad_list_path is not None:
        vad_list = read_json(vad_list_path)
        duration = waveform.shape[-1] / sample_rate
        vad = vad_list_to_onehot(
            vad_list, hop_time=int(1 / frame_hz), duration=duration
        )

    if device is not None:
        waveform = waveform.to(device)
        if vad is not None:
            vad = vad.to(device)

    return waveform, vad


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
