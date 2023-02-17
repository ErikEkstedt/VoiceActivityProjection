from argparse import ArgumentParser
import torch
from tqdm import tqdm

from vap.model import VapGPT, VapConfig
from vap.audio import load_waveform
from vap.utils import vad_list_to_onehot, batch_to_device
from vap_dataset.corpus.candor import CandorReader


SAMPLE_RATE = 16_000

STEP_EXTRACTION_LIMIT = 160  # seconds
CONTEXT_TIME = 20
STEP_TIME = 5

# TODO: Why is the CUDA memory exploding??


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


def get_duration(x):
    return x.shape[-1] / SAMPLE_RATE


def get_minimal_output_json(out, loss, vad):
    min_out = {
        "p_now": out["p_now"][0, :, 0].tolist(),
        "p_future": out["p_future"][0, :, 0].tolist(),
        "loss": loss.tolist(),
        "vad0": vad[0, :, 0].tolist(),
        "vad0": vad[0, :, 1].tolist(),
        "model_vad0": out["vad"][0, :, 0].tolist(),
        "model_vad1": out["vad"][0, :, 1].tolist(),
        "H": out["H"][0].tolist(),
    }
    return min_out


class VapExtractor:
    def __init__(
        self,
        context_time: float = 20,
        step_time: float = 5,
        state_dict_path="../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
    ):
        self.model = self.load_model(state_dict_path)  # sets self.device

        self.context_time = context_time
        self.step_time = step_time
        self.chunk_time = self.context_time + self.step_time
        self.chunk_lable_time = (
            self.context_time + self.step_time + self.model.horizon_time
        )

        # Samples
        self.step_samples = int(self.step_time * self.model.sample_rate)
        self.chunk_samples = int(self.chunk_time * self.model.sample_rate)

        # Frames
        self.step_frames = int(self.step_time * self.model.frame_hz)
        self.chunk_frames = int(self.chunk_time * self.model.frame_hz)
        self.chunk_label_frames = int(self.chunk_time * self.model.frame_hz)

    def __repr__(self):
        s = "VapExtractor\n"
        s += f"Context time: {self.context_time}s\n"
        s += f"Step time: {self.step_time}s\n"
        s += f"Chunk time: {self.chunk_time}s\n"
        s += f"Step samples: {self.step_samples}\n"
        s += f"Chunk samples: {self.chunk_samples}\n"
        s += f"Step frames: {self.step_frames}\n"
        s += f"Chunk frames: {self.chunk_frames}\n"
        return s

    def load_model(self, state_dict_path: str):
        conf = VapConfig()
        model = VapGPT(conf)
        sd = torch.load(state_dict_path)
        model.load_state_dict(sd)

        self.device = "cpu"
        if torch.cuda.is_available():
            model = model.to("cuda")
            self.device = "cuda"

        # Set to evaluation mode
        model = model.eval()
        return model

    def model_forward(
        self,
        waveform,
        vad=None,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ):
        print("tmp waveform: ", tuple(waveform.shape))
        if vad is not None:
            print("tmp vad: ", tuple(vad.shape))
            out = self.model(waveform.to(self.device), vad.to(self.device))
        else:
            out = self.model(waveform.to(self.device))
        probs = out["logits"].softmax(dim=-1)
        model_vad = out["vad"].sigmoid()

        # Calculate entropy over each projection-window prediction (i.e. over
        # frames/time) If we have C=256 possible states the maximum bit entropy
        # is 8 (2^8 = 256) this means that the model have a one in 256 chance
        # to randomly be right. The model can't do better than to uniformly
        # guess each state, it has learned (less than) nothing. We want the
        # model to have low entropy over the course of a dialog, "thinks it
        # understands how the dialog is going", it's a measure of how close the
        # information in the unseen data is to the knowledge encoded in the
        # training data.
        h = -probs * probs.log2()  # Entropy
        H = h.sum(dim=-1)  # average entropy per frame

        # first two bins
        p_now = self.model.objective.probs_next_speaker_aggregate(
            probs, from_bin=now_lims[0], to_bin=now_lims[-1]
        )
        p_future = self.model.objective.probs_next_speaker_aggregate(
            probs, from_bin=future_lims[0], to_bin=future_lims[-1]
        )
        return {
            "probs": probs.to("cpu"),
            "vad": model_vad.to("cpu"),
            "p_now": p_now.to("cpu"),
            "p_future": p_future.to("cpu"),
            "H": H.to("cpu"),
            "logits": out["logits"].to("cpu"),
        }

    def forward_and_join(self, x, out, vad=None):
        o = self.model_forward(x, vad=vad)

        if vad is not None:
            labels = self.model.objective.get_labels(vad)
            vap_loss = self.model.objective.loss_vap(
                o["logits"], labels, reduction="none"
            )
            out["loss"] = torch.cat(
                [out["loss"], vap_loss[:, -self.step_frames].cpu()], dim=1
            )

        out["vad"] = torch.cat(
            [out["vad"], o["vad"][:, -self.step_frames :].cpu()], dim=1
        )
        out["p_now"] = torch.cat(
            [out["p_now"], o["p_now"][:, -self.step_frames :].cpu()], dim=1
        )
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -self.step_frames :].cpu()], dim=1
        )
        out["probs"] = torch.cat(
            [out["probs"], o["probs"][:, -self.step_frames :].cpu()], dim=1
        )
        out["H"] = torch.cat([out["H"], o["H"][:, -self.step_frames :].cpu()], dim=1)
        return out

    def forward_and_join_last(
        self,
        waveform,
        out,
        expected_frames,
        processed_frames,
        vad=None,
        verbose: bool = False,
    ):
        omitted_frames = expected_frames - processed_frames
        omitted_samples = self.model.sample_rate * omitted_frames / self.model.frame_hz

        w = waveform[..., -self.chunk_samples :]
        if vad is not None:
            vad = vad[..., -omitted_frames:, :].to(self.device)

        if verbose:
            print(f"Expected frames {expected_frames} != {processed_frames}")
            print(f"omitted frames: {omitted_frames}")
            print(f"omitted samples: {omitted_samples}")
            print(f"chunk_samples: {self.chunk_samples}")
            print("-------------------------------------")
            print("waveform: ", tuple(w.shape))
            if vad is not None:
                print("vad: ", tuple(vad.shape))

        o = self.model_forward(w, vad=vad)
        if vad is not None:
            labels = self.model.objective.get_labels(vad)
            vap_loss = self.model.objective.loss_vap(
                o["logits"], labels, reduction="none"
            )
            out["loss"] = torch.cat(
                [out["loss"], vap_loss[:, -omitted_frames].cpu()], dim=1
            )

        out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -omitted_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -omitted_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -omitted_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -omitted_frames:]], dim=1)
        return out

    def step_extraction(
        self,
        waveform,
        vad=None,
        pbar=True,
        verbose=False,
    ):
        """
        Takes a waveform, the model, and extracts probability output in chunks with
        a specific context and step time. Concatenates the output accordingly and returns full waveform output.
        """

        n_samples = waveform.shape[-1]
        duration = round(n_samples / self.model.sample_rate, 2)

        # Fold the waveform to get total chunks
        folds = waveform.unfold(
            dimension=-1, size=self.chunk_samples, step=self.step_samples
        ).permute(2, 0, 1, 3)
        print("folds: ", tuple(folds.shape))

        vfolds = [None]
        if vad is not None:
            vfolds = vad.unfold(
                dimension=1, size=self.chunk_label_frames, step=self.step_frames
            ).permute(2, 0, 1, 3)
            print("vfolds: ", tuple(vfolds.shape))

        input()

        expected_frames = round(duration * self.model.frame_hz)

        # First chunk
        # Use all extracted data. Does not overlap with anything prior.

        out = self.model_forward(folds[0], vfolds[0])
        actual_pbar = range(len(folds[1:]))
        if pbar:
            actual_pbar = tqdm(
                actual_pbar,
                desc=f"Context: {self.context_time}s, step: {self.step_time}",
            )

        # Iterate over all other folds
        # and add simply the new processed step
        tmp_vad = None
        for ii in actual_pbar:
            w = folds[ii]
            if vfolds[0] is not None:
                tmp_vad = vfolds[ii]
            out = self.forward_and_join(w, out, vad=tmp_vad)
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {tuple(v.shape)}, {v.device}")
                else:
                    print(f"{k}: {v}")

        # ###################################################################
        # # Handle LAST SEGMENT (not included in `unfold`)
        # ###################################################################
        # processed_frames = out["p_now"].shape[1]
        # if expected_frames != processed_frames:
        #     out = self.forward_and_join_last(
        #         waveform,
        #         out,
        #         expected_frames=expected_frames,
        #         processed_frames=processed_frames,
        #         vad=vad,
        #         verbose=verbose,
        #     )

        return out

    def extract(self, waveform, vad=None):
        duration = get_duration(waveform)
        if duration > STEP_EXTRACTION_LIMIT:
            out = self.step_extraction(waveform, vad=vad, pbar=True, verbose=False)
        else:
            out = self.model_forward(waveform, vad=vad)
            out = batch_to_device(out, "cpu")

        return out


if __name__ == "__main__":

    reader = CandorReader()
    extractor = VapExtractor()
    print(extractor)

    # Save sample
    d = reader[0]
    waveform, _ = load_waveform(
        d["audio_path"], sample_rate=SAMPLE_RATE
    )  # (2, n_samples)
    waveform = waveform.unsqueeze(0)  # (2, n_samples) -> (1, 2, n_samples)
    duration = get_duration(waveform)
    vad = vad_list_to_onehot(d["vad_list"], duration=duration, frame_hz=50).unsqueeze(0)

    print("waveform: ", tuple(waveform.shape))
    print("vad: ", tuple(vad.shape))

    out = extractor.extract(waveform)
