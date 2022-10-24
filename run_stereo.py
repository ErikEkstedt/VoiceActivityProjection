import torch
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from vap.model import VAPModel
from vap.utils import load_sample, batch_to_device, everything_deterministic
from vap.plot_utils import plot_stereo


everything_deterministic()

torch.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="example/VAP_50Hz_ad20s_134-epoch1-val_2.55.ckpt",
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


if __name__ == "__main__":
    args = get_args()

    print("Load Model...")
    model = VAPModel.load_from_checkpoint(args.checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    waveform, _ = load_sample(
        path=args.wav,
        sample_rate=model.sample_rate,
        frame_hz=model.frame_hz,
        force_stereo=True,
        device=model.device,
    )
    print("waveform: ", tuple(waveform.shape), waveform.device)
    out = model.output(waveform=waveform)
    out = batch_to_device(out, "cpu")
    print(out.keys())

    # Plot
    sample_rate = model.sample_rate
    frame_hz = model.frame_hz
    horizon_samples = int(model.horizon_time * model.sample_rate)
    wav = waveform[0].cpu()
    vad = out["vad"][0].cpu()
    p_ns = out["p"][0, :, 0].cpu()
    p_bc = out["p_bc"][0].cpu()

    fig, ax = plot_stereo(wav, p_ns, vad, plot=False)
    plt.show()
