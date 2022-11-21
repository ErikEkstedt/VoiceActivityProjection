import torch
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

    duration = waveform.shape[-1] / model.sample_rate

    # Maximum known duration with a 24Gb 'NVIDIA GeForce RTX 3090' is 164s

    ###########################################################
    # Model Forward
    ###########################################################
    if args.chunk:

        n_samples = waveform.shape[-1]
        chunk_sample = int(model.sample_rate * args.chunk_time)
        chunk_overlap_sample = int(model.sample_rate * args.chunk_overlap_time)
        chunk_step_sample = chunk_sample - chunk_overlap_sample
        chunk_frames = int(model.frame_hz * args.chunk_time)
        chunk_overlap_frames = int(model.frame_hz * args.chunk_overlap_time)
        chunk_step_frames = chunk_frames - chunk_overlap_frames
        # print("n_samples: ", n_samples)
        # print("chunk_sample: ", chunk_sample)
        # print("chunk_overlap_sample: ", chunk_overlap_sample)
        # print("chunk_step_sample: ", chunk_step_sample)
        # print("chunk_frames: ", chunk_frames)
        # print("chunk_overlap_frames: ", chunk_overlap_frames)
        # print("chunk_step_frames: ", chunk_step_frames)

        folds = waveform.unfold(
            dimension=-1, size=chunk_sample, step=chunk_step_sample
        ).permute(2, 0, 1, 3)
        # print("folds: ", tuple(folds.shape))

        # 8209

        # first segment contains the longest possible context
        out = model.output(folds[0])
        out.pop("logits")
        out.pop("shift")
        out.pop("hold")
        out.pop("long")
        out.pop("short")
        out.pop("pred_shift")
        out.pop("pred_shift_neg")
        out.pop("pred_backchannel")
        out.pop("pred_backchannel_neg")
        out.pop("vad_list")
        for w in tqdm(folds[1:], desc=f"Processing {args.chunk_time} chunks"):
            # print("w: ", tuple(w.shape))
            o = model.output(w)
            # Concatenate
            out["vad"] = torch.cat(
                [out["vad"], o["vad"][:, chunk_overlap_frames:]], dim=1
            )
            out["p"] = torch.cat([out["p"], o["p"][:, chunk_overlap_frames:]], dim=1)
            out["p_all"] = torch.cat(
                [out["p_all"], o["p_all"][:, chunk_overlap_frames:]], dim=1
            )
            out["p_bc"] = torch.cat(
                [out["p_bc"], o["p_bc"][:, chunk_overlap_frames:]], dim=1
            )
            out["probs"] = torch.cat(
                [out["probs"], o["probs"][:, chunk_overlap_frames:]], dim=1
            )
        ###################################################################
        # Handle LAST SEGMENT (not included in `unfold`)
        ###################################################################
        n_folds = int((n_samples - chunk_sample) / chunk_step_sample + 1.0)
        total = (n_folds - 1) * chunk_step_sample + chunk_sample
        omitted = n_samples - total
        omitted_frames = int(omitted * model.frame_hz / model.sample_rate)
        assert folds.shape[0] == n_folds, "Fold calculation inaccurate"
        assert (
            omitted < chunk_sample
        ), f"`omitted` is greater than chunk_sample: {omitted} >  {chunk_sample}"
        assert (
            omitted_frames < chunk_frames
        ), f"`omitted_frames` is greater than chunk_frame: {omitted_frames} >  {chunk_frames}"
        ###################################################################
        # Last forward
        ###################################################################
        w = waveform[..., -chunk_sample:]
        o = model.output(w)
        # Concatenate
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
        out["p"] = torch.cat([out["p"], o["p"][:, -omitted_frames:]], dim=1)
        out["p_all"] = torch.cat([out["p_all"], o["p_all"][:, -omitted_frames:]], dim=1)
        out["p_bc"] = torch.cat([out["p_bc"], o["p_bc"][:, -omitted_frames:]], dim=1)
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -omitted_frames:]], dim=1)
        ###################################################################
        # Extract Vad-list over entire vad
        ###################################################################
        out["vad_list"] = vad_output_to_vad_list(
            out["vad"],
            frame_hz=model.frame_hz,
            vad_thresh=args.vad_thresh,
            ipu_thresh_time=args.ipu_thresh_time,
        )
        out = batch_to_device(out, "cpu")  # to cpu for plot/save
        # for k, v in out.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {tuple(v.shape)}")
        #     else:
        #         print(f"{k}: {v}")
    else:
        out = model.output(
            waveform=waveform,
            vad_thresh=args.vad_thresh,
            ipu_thresh_time=args.ipu_thresh_time,
        )
        out = batch_to_device(out, "cpu")  # to cpu for plot/save

    ###########################################################
    # Save Model Output
    ###########################################################

    if args.filename is None:
        savepath = basename(args.wav).replace(".wav", "")
    else:
        savepath = args.filename.replace(".json", "")
    model.save_output(
        out,
        savepath=f"{savepath}.json",
        checkpoint=basename(args.checkpoint),
        full=args.full,
    )

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
