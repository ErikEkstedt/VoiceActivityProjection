from argparse import ArgumentParser
from tqdm import tqdm
import torch
import pandas as pd
from os.path import basename

from vap.model import VapGPT, VapConfig
from vap.audio import load_waveform
from vap.utils import vad_list_to_onehot, batch_to_device, read_json, write_json


SAMPLE_RATE = 16_000
STEP_EXTRACTION_LIMIT = 160  # seconds
CONTEXT_TIME = 20
STEP_TIME = 5


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to waveform",
    )
    parser.add_argument("-v", "--vad", type=str, help="Path to vad list", default=None)
    parser.add_argument(
        "--output_format",
        type=str,
        default="json",
        help="output format: ['json', 'csv']. Default='json'",
    )
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
        help="Path to state_dict",
    )
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.add_argument(
        "--context_time",
        type=float,
        default=CONTEXT_TIME,
        help="Duration of each chunk processed by model",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=STEP_TIME,
        help="Increment to process in a step",
    )
    args = parser.parse_args()

    conf = VapConfig.args_to_conf(args)
    return args, conf


def get_duration(x):
    return x.shape[-1] / SAMPLE_RATE


def json_data_to_df(out):
    data = []
    for i in range(len(out["p_now"])):
        dd = {}
        for k, v in out.items():
            if k == "loss":
                if i >= len(v):
                    dd[k] = 0
                else:
                    dd[k] = v[i]
            else:
                try:
                    dd[k] = v[i]
                except:
                    dd[k] = 0
                    print(f"{k}[{i}]: {len(v)} -> setting to 0")
        data.append(dd)
    return pd.DataFrame(data)


def get_minimal_output_json(out, vad):
    min_out = {
        "p_now": out["p_now"][0, :, 0].tolist(),
        "p_future": out["p_future"][0, :, 0].tolist(),
        "model_vad0": out["vad"][0, :, 0].tolist(),
        "model_vad1": out["vad"][0, :, 1].tolist(),
        "H": out["H"][0].tolist(),
    }
    if "loss" in out:
        min_out["loss"] = out["loss"][0].tolist()
    if vad is not None:
        min_out["vad0"] = vad[0, :, 0].tolist()
        min_out["vad1"] = vad[0, :, 1].tolist()
    return min_out


class VapExtractor:
    """
    input: |------------ chunk time ---------------------|
    input: |------ context time -------|--- step time ---|
    """

    def __init__(
        self,
        context_time: float = 20,
        step_time: float = 5,
        state_dict_path="../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
    ):
        self.model = self.load_model(state_dict_path)  # sets self.device

        # Time
        self.context_time = context_time
        self.step_time = step_time
        self.chunk_time = self.context_time + self.step_time
        self.chunk_label_time = self.chunk_time + self.model.horizon_time

        # Samples
        self.step_samples = int(self.step_time * self.model.sample_rate)
        self.chunk_samples = int(self.chunk_time * self.model.sample_rate)

        # Frames
        self.step_frames = int(self.step_time * self.model.frame_hz)
        self.chunk_frames = int(self.chunk_time * self.model.frame_hz)
        self.chunk_label_frames = int(self.chunk_label_time * self.model.frame_hz)

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

    def join_out_dict(self, tmp_out, out, last_n_frames):
        out["vad"] = torch.cat(
            [out["vad"], tmp_out["vad"][:, -last_n_frames:].cpu()], dim=1
        )
        out["p_now"] = torch.cat(
            [out["p_now"], tmp_out["p_now"][:, -last_n_frames:].cpu()], dim=1
        )
        out["p_future"] = torch.cat(
            [out["p_future"], tmp_out["p_future"][:, -last_n_frames:].cpu()], dim=1
        )
        out["probs"] = torch.cat(
            [out["probs"], tmp_out["probs"][:, -last_n_frames:].cpu()], dim=1
        )
        out["H"] = torch.cat([out["H"], tmp_out["H"][:, -last_n_frames:].cpu()], dim=1)

        if "loss" in tmp_out:
            if "loss" in out:
                out["loss"] = torch.cat(
                    [out["loss"], tmp_out["loss"][:, -last_n_frames:].cpu()], dim=1
                )
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

        # Fold the waveform to get total chunks
        folds = waveform.unfold(
            dimension=-1, size=self.chunk_samples, step=self.step_samples
        ).permute(2, 0, 1, 3)
        if vad is not None:
            vfolds = vad.unfold(
                dimension=1, size=self.chunk_label_frames, step=self.step_frames
            ).permute(2, 0, 1, 3)
            # print("vfolds: ", tuple(vfolds.shape), vfolds.device)

        ###################################################################
        # First chunk
        # Use all extracted data. Does not overlap with anything prior.
        ###################################################################
        out = self.model.probs(
            folds[0].to(self.device),
            vad=vad if vad is None else vfolds[0].to(self.device),
        )
        out = batch_to_device(out, "cpu")

        ###################################################################
        # Iterate over all other folds and add the new processed results
        ###################################################################
        actual_pbar = range(1, len(folds[1:]))
        if pbar:
            actual_pbar = tqdm(
                actual_pbar,
                desc=f"Context: {self.context_time}s, step: {self.step_time}",
            )

        for ii in actual_pbar:
            o = self.model.probs(
                folds[ii].to(self.device),
                vad=vad if vad is None else vfolds[ii].to(self.device),
            )
            out = self.join_out_dict(o, out, self.step_frames)

        ###################################################################
        # Handle LAST SEGMENT (not included in `unfold`)
        ###################################################################
        n_samples = waveform.shape[-1]
        duration = round(n_samples / self.model.sample_rate, 2)
        expected_frames = round(duration * self.model.frame_hz)
        processed_frames = out["p_now"].shape[1]
        if expected_frames != processed_frames:
            print("Expected frames != processed frames")

            omitted_frames = expected_frames - processed_frames
            omitted_samples = (
                self.model.sample_rate * omitted_frames / self.model.frame_hz
            )

            assert (
                omitted_frames < self.chunk_frames
            ), f"Omitted frames {omitted_frames} > chunk frames {self.chunk_frames}"

            w = waveform[..., -self.chunk_samples :].to(self.device)
            if verbose:
                print(f"Expected frames {expected_frames} != {processed_frames}")
                print(f"omitted frames: {omitted_frames}")
                print(f"omitted samples: {omitted_samples}")
                print(f"chunk_samples: {self.chunk_samples}")
                print("-------------------------------------")
                print("waveform: ", tuple(w.shape))

            o = self.model.probs(
                w,
                vad=vad
                if vad is None
                else vad[..., -self.chunk_frames :].to(self.device),
            )
            out = self.join_out_dict(o, out, last_n_frames=omitted_frames)
        return out

    @torch.no_grad()
    def extract(self, waveform, vad=None):
        duration = get_duration(waveform)
        if duration > STEP_EXTRACTION_LIMIT:
            out = self.step_extraction(waveform, vad=vad, pbar=True, verbose=False)
        else:
            out = self.model(waveform, vad=vad)
            out = batch_to_device(out, "cpu")
        return out


def test_with_dataset():
    from vap_dataset.corpus import CandorReader, SwbReader

    reader = CandorReader()
    # reader = SwbReader()
    extractor = VapExtractor()
    print(extractor)

    index = list(range(5))
    for i in index:
        d = reader[i]
        waveform, _ = load_waveform(
            d["audio_path"], sample_rate=SAMPLE_RATE
        )  # (2, n_samples)
        waveform = waveform.unsqueeze(0)  # (2, n_samples) -> (1, 2, n_samples)
        duration = get_duration(waveform)
        vad = vad_list_to_onehot(
            d["vad_list"], duration=duration, frame_hz=50
        ).unsqueeze(0)
        out = extractor.extract(waveform)
        min_out = get_minimal_output_json(out, vad)
        df = json_data_to_df(min_out)
        savepath = d["session"] + ".csv"
        df.to_csv(savepath)
        print("Saved -> ", savepath)

    for k, v in min_out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        elif isinstance(v, list):
            print(f"{k}: {len(v)}")
        else:
            print(f"{k}: {v}")

    # from vap.plot_utils import plot_vap
    # import matplotlib.pyplot as plt
    # window_duration = 10
    # frame_hz = 50
    # # for start_time in range(0, int(duration), 20):
    # for start_time in range(120, int(duration), window_duration//2):
    #     end_time = start_time + window_duration
    #     #
    #     start_frame = round(start_time * frame_hz)
    #     end_frame = round(end_time * frame_hz)
    #     start_sample = round(start_time * SAMPLE_RATE)
    #     end_sample = round(end_time * SAMPLE_RATE)
    #     #
    #     p_now = out["p_now"][0, start_frame:end_frame, 0]
    #     p_fut = out["p_future"][0, start_frame:end_frame, 0]
    #     v = vad[0, start_frame:end_frame]
    #     loss = out["loss"][0, start_frame:end_frame]
    #     w = waveform[0, :, start_sample:end_sample]
    #     plt.close("all")
    #     fig, ax = plot_vap(
    #         waveform=w,
    #         p_now=p_now,
    #         p_fut=p_fut,
    #         vad=v,
    #         plot=False,
    #         future_colors=['teal', 'yellow'],
    #         figsize=(20, 10)
    #     )
    #     xx = torch.arange(len(loss)) / frame_hz
    #     lax = ax[-2].twinx()
    #     lax.plot(xx, loss, linewidth=3, color="red")
    #     lax.set_ylim([0, 15])
    #     plt.show()


if __name__ == "__main__":
    args, conf = get_args()
    extractor = VapExtractor(
        context_time=args.context_time,
        step_time=args.step_time,
        state_dict_path=args.state_dict,
    )

    waveform, _ = load_waveform(args.audio, sample_rate=SAMPLE_RATE)  # (2, n_samples)
    waveform = waveform.unsqueeze(0)  # (2, n_samples) -> (1, 2, n_samples)
    print("waveform: ", tuple(waveform.shape))
    duration = get_duration(waveform)

    vad = None
    if args.vad is not None:
        vad_list = read_json(args.vad)
        vad = vad_list_to_onehot(vad_list, duration=duration, frame_hz=50).unsqueeze(0)
        print("vad: ", tuple(vad.shape))

    out = extractor.extract(waveform)
    min_out = get_minimal_output_json(out, vad)
    print("Keys:    Frames")
    for k, v in min_out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}:     {tuple(v.shape)}")
        if isinstance(v, list):
            print(f"{k}:     {len(v)}")
        else:
            print(f"{k}:     {v}")

    # save output
    savename = basename(args.audio).replace(".wav", "")
    if args.output_format == "json":
        write_json(min_out, savename + ".json")
        print("Saved -> ", savename + ".json")
    else:
        df = json_data_to_df(min_out)
        df.to_csv(savename + ".csv")
        print("Saved -> ", savename + ".csv")
