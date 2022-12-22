from os.path import exists
from typing import Any, Dict, List, Tuple, Union
import json
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import torch
from torch import Tensor

from vap.audio import time_to_frames, load_waveform, get_audio_info, time_to_samples
from vap.utils import read_json, add_zero_channel, vad_list_to_onehot, batch_to_device

PHRASE_JSON = "dataset_phrases/phrases.json"
PHRASE_CSV = "dataset_phrases/phrases.csv"
EXAMPLE_TO_SCP_WORD = {
    "student": "student",
    "psychology": "psychology",
    "first_year": "student",
    "basketball": "basketball",
    "experiment": "before",
    "live": "yourself",
    "work": "side",
    "bike": "bike",
    "drive": "here",
}


def load_phrase_dataframe(path=PHRASE_CSV):
    def _time(x):
        if isinstance(x, str):
            if len(x) > 0:
                try:
                    return json.loads(x)
                except:
                    print(x)
                    input()
        return x

    def _text(x):
        return [a[1:-1] for a in x.strip("][").split(", ")]

    converters = {
        "starts": _time,
        "ends": _time,
        "vad_list": _time,
        "phone_starts": _time,
        "phone_ends": _time,
        "words": _text,
        "phones": _text,
    }
    return pd.read_csv(path, converters=converters)


# kept for reference
def to_pandas():
    from textgrids import TextGrid

    # save better data
    def read_text_grid(path: str) -> dict:
        grid = TextGrid(path)
        data = {"words": [], "phones": []}
        for word_phones, vals in grid.items():
            for w in vals:
                if w.text == "":
                    continue
                # what about words spoken multiple times?
                # if word_phones == 'words':
                #     data[word_phones][w.text] = (w.xmin, w.xmax)
                data[word_phones].append((w.xmin, w.xmax, w.text))
        return data

    def get_scp_end_time(sample):
        target_word = EXAMPLE_TO_SCP_WORD[sample["phrase"]]
        end_time = -1
        for w, end in zip(sample["words"], sample["ends"]):
            if w == target_word:
                end_time = end
                break
        return end_time

    p = read_json(PHRASE_JSON)
    samples = []
    for exp_name, v in p.items():
        for long_short, vv in v.items():
            for gender, vvv in vv.items():
                for phrase_idx, x in enumerate(vvv):
                    x.pop("size")
                    x.pop("vad")
                    x["phrase"] = exp_name
                    x["gender"] = gender
                    x["long_short"] = long_short
                    x["phrase_idx"] = phrase_idx
                    tg = read_text_grid(
                        x["audio_path"]
                        .replace("/audio/", "/alignment/")
                        .replace(".wav", ".TextGrid")
                    )
                    phones, phone_starts, phone_ends = [], [], []
                    for ph in tg["phones"]:
                        phone_starts.append(ph[0])
                        phone_ends.append(ph[1])
                        phones.append(str(ph[-1]))
                    x["phones"] = phones
                    x["phone_starts"] = phone_starts
                    x["phone_ends"] = phone_ends
                    words, starts, ends = [], [], []
                    vad_list = []
                    for ph in tg["words"]:
                        vad_list.append([ph[0], ph[1]])
                        starts.append(ph[0])
                        ends.append(ph[1])
                        words.append(ph[-1])
                    x["words"] = words
                    x["starts"] = starts
                    x["ends"] = ends
                    x["vad_list"] = [vad_list, []]
                    scp_end_time = x["ends"][-1]
                    if long_short == "long":
                        scp_end_time = get_scp_end_time(x)
                    x["scp"] = scp_end_time
                    samples.append(x)
    df = pd.DataFrame(samples)
    return df


def phrases_collate_fn(batch):
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for kk, vv in b.items():
            if kk == "waveform":
                out[kk].append(vv[0].permute(1, 0))
            elif kk == "vad":
                out[kk].append(vv[0])
            else:
                out[kk].append(vv)
    out["waveform"] = pad_sequence(out["waveform"], batch_first=True).permute(0, 2, 1)
    out["vad"] = pad_sequence(out["vad"], batch_first=True)
    for k, v in out.items():
        if k not in ["waveform", "vad"]:
            if not isinstance(v[0], str):
                out[k] = torch.tensor(v)
    return out


def get_region_shift_probs(p, end, region_frames, speaker=1):
    assert (
        p.ndim == 2
    ), f"get_region_probs expects single `p` (n_frames, 2) != {p.shape}"
    pred_start = end - region_frames
    react_end = end + region_frames
    hold = p[:pred_start, speaker]
    pred = p[pred_start:end, speaker]
    react = p[end:react_end, speaker]
    return hold, pred, react


class PhrasesCallback(pl.Callback):
    """
    A callback to evaluate the performance of the model over the artificially create `phrases` dataset
    """

    def __init__(
        self,
        region_time=0.2,
        silence=2,
        batch_size=5,
        num_workers=2,
        sample_rate=16_000,
        frame_hz=50,
        mono=False,
    ):
        self.dset = PhraseDataset(
            sample_rate=sample_rate,
            vad_hz=frame_hz,
            audio_mono=mono,
            silence=silence,
        )
        self.region_time = region_time
        self.region_frames = time_to_frames(region_time, self.dset.vad_hop_time)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def extract_stats(self, model):
        dloader = DataLoader(
            self.dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=phrases_collate_fn,
        )

        region_data = {
            "short": {
                "now": {"hold": [], "pred": [], "react": []},
                "future": {"hold": [], "pred": [], "react": []},
                "tot": {"hold": [], "pred": [], "react": []},
            },
            "long": {
                "now": {"hold": [], "pred": [], "react": []},
                "future": {"hold": [], "pred": [], "react": []},
                "tot": {"hold": [], "pred": [], "react": []},
                "scp_now": {"hold": [], "pred": [], "react": []},
                "scp_future": {"hold": [], "pred": [], "react": []},
                "scp_tot": {"hold": [], "pred": [], "react": []},
            },
        }

        for batch in tqdm(dloader, desc="Phrases"):
            batch = batch_to_device(batch, model.device)
            out = model(batch["waveform"])
            probs = model.objective.get_probs(out["logits"])
            for ii in range(batch["waveform"].shape[0]):
                end = batch["end"][ii]
                scp = batch["scp"][ii]
                long_short = batch["long_short"][ii]
                for pp in ["p_now", "p_future", "p_tot"]:
                    pp_name = pp.replace("p_", "")
                    h, p, r = get_region_shift_probs(
                        probs[pp][ii], end, self.region_frames, speaker=1
                    )
                    region_data[long_short][pp_name]["hold"].append(h)
                    region_data[long_short][pp_name]["pred"].append(p)
                    region_data[long_short][pp_name]["react"].append(r)
                    if long_short == "long":
                        h, p, r = get_region_shift_probs(
                            probs[pp][ii], scp, self.region_frames, speaker=1
                        )
                        region_data[long_short][f"scp_{pp_name}"]["hold"].append(h)
                        region_data[long_short][f"scp_{pp_name}"]["pred"].append(p)
                        region_data[long_short][f"scp_{pp_name}"]["react"].append(r)

        # Extract stats
        mstats = {}
        sstats = {}
        for long_short, v in region_data.items():
            for pp, vv in v.items():
                for region, data in vv.items():
                    name = f"{long_short}_{pp}_{region}"
                    data = torch.cat(data)
                    mstats[name] = data.mean().cpu().item()
                    sstats[name] = data.std().cpu().item()
        return mstats, sstats

    def log_mean_stats(self, stats, pl_module, split):
        for name, s in stats.items():
            pl_module.log(f"{split}_{name}", s, sync_dist=True)

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):
        means, _ = self.extract_stats(pl_module)

        # Log only values we care about in val
        pl_module.log(f"val_ps_hold", means["short_future_hold"], sync_dist=True)
        pl_module.log(f"val_ps_pred", means["short_future_pred"], sync_dist=True)
        pl_module.log(f"val_ps_react", means["short_now_react"], sync_dist=True)
        # Long end
        pl_module.log(f"val_pl_hold", means["long_future_hold"], sync_dist=True)
        pl_module.log(f"val_pl_pred", means["long_future_pred"], sync_dist=True)
        pl_module.log(f"val_pl_react", means["long_now_react"], sync_dist=True)
        # Long scp
        pl_module.log(f"val_pls_hold", means["long_scp_future_hold"], sync_dist=True)
        pl_module.log(f"val_pls_pred", means["long_scp_future_pred"], sync_dist=True)
        pl_module.log(f"val_pls_react", means["long_scp_now_react"], sync_dist=True)

    def on_test_epoch_start(self, trainer, pl_module, *args, **kwargs):
        means, _ = self.extract_stats(pl_module)

        # log everything during test
        for name, s in means.items():
            pl_module.log(f"test_{name}", s, sync_dist=True)


class PhraseDataset(Dataset):
    def __init__(
        self,
        csv_path: str = PHRASE_CSV,
        # AUDIO #################################
        sample_rate: int = 16000,
        audio_mono: bool = False,
        silence: float = 2.0,
        # VAD #################################
        vad: bool = True,
        vad_hz: int = 50,
        vad_horizon: int = 2,
    ):
        super().__init__()
        self.df = load_phrase_dataframe(csv_path)

        # Audio (waveforms)
        self.sample_rate = sample_rate
        self.audio_mono = audio_mono
        self.silence = silence

        # VAD parameters
        self.vad = vad  # use vad or not
        self.vad_hz = vad_hz
        self.vad_hop_time = 1.0 / vad_hz
        self.horizon_time = vad_horizon
        self.vad_horizon = time_to_frames(vad_horizon, hop_time=self.vad_hop_time)

    def __len__(self) -> int:
        return len(self.df)

    def get_sample(
        self, phrase: str, long_short: str, gender: str, phrase_idx: int
    ) -> Dict[str, Any]:
        return (
            self.df.loc[
                (self.df["phrase"] == phrase)
                & (self.df["long_short"] == long_short)
                & (self.df["gender"] == gender)
                & (self.df["phrase_idx"] == phrase_idx)
            ]
            .iloc[0]
            .to_dict()
        )

    def sample_to_output(
        self, sample: Dict
    ) -> Dict[str, Union[list, Tensor, float, str]]:
        # Load audio

        w, _ = load_waveform(
            sample["audio_path"],
            sample_rate=self.sample_rate,
            mono=self.audio_mono,
        )
        duration = w.shape[-1] / self.sample_rate
        last_activity = sample["ends"][-1]
        with_silence = last_activity + self.silence
        n_samples = time_to_samples(with_silence - duration, self.sample_rate)
        w = torch.cat((w, torch.zeros((1, n_samples))), dim=-1)
        if not self.audio_mono:
            z = torch.zeros_like(w)
            w = torch.cat((w, z))
        return {
            "waveform": w.unsqueeze(0),
            "vad": vad_list_to_onehot(
                sample["vad_list"],
                hop_time=self.vad_hop_time,
                duration=with_silence,
                channel_last=True,
            ).unsqueeze(0),
            "scp": time_to_frames(sample["scp"], hop_time=self.vad_hop_time),
            "end": time_to_frames(sample["ends"][-1], hop_time=self.vad_hop_time),
            "phrase": sample["phrase"],
            "long_short": sample["long_short"],
            "gender": sample["gender"],
            "phrase_idx": sample["phrase_idx"],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.df.iloc[idx]
        output = self.sample_to_output(sample)
        return output


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from vap.plot_utils import plot_phrases_sample
    from vap.model import VapGPT, load_older_state_dict

    model = VapGPT()
    sd = load_older_state_dict()
    model.load_state_dict(sd, strict=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    clb = PhrasesCallback()

    dset = PhraseDataset()
    dloader = DataLoader(
        dset, batch_size=4, num_workers=0, collate_fn=phrases_collate_fn
    )

    batch = next(iter(dloader))
