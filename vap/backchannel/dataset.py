from glob import glob
from os.path import basename, exists, join
from tqdm import tqdm
import json
import pandas as pd
from typing import Dict, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from vap.audio import load_waveform, get_audio_info, time_to_frames, time_to_samples
from vap.utils import read_json, read_txt


# WARNING: Source: https://github.com/phiresky/backchannel-prediction/tree/master/data
JSON_PATH = "dataset_swb/utterance_is_backchannel.json"
BC_CSV = "dataset_swb/backchannels.csv"
SWB_DA_PATH = "dataset_swb/swb_dialog_acts_words"
SWB_ANNO_PATH = "dataset_swb/swb_ms98_transcriptions"
SWB_SPLIT_PATH = "dataset_swb/splits"
REL_PATH = "dataset_swb/relative_audio_path.json"


class SWBReader:
    def __init__(self, anno_path=SWB_ANNO_PATH, da_path=SWB_DA_PATH):
        self.anno_path = anno_path
        self.da_path = da_path
        self.session_to_path = self.get_session_paths()
        # self.sessions = list(self.session_to_path.keys())
        self.sessions = read_txt(join(SWB_SPLIT_PATH, "test.txt")) + read_txt(
            join(SWB_SPLIT_PATH, "test.txt")
        )
        self.audio_rel_paths = read_json(REL_PATH)

    def __len__(self):
        return len(self.sessions)

    def session_to_audio_path(self, session, audio_root):
        rel = self.audio_rel_paths[str(session)] + ".wav"
        return join(audio_root, rel)

    def get_session_paths(self):
        def _session_name(p):
            return (
                basename(p)
                .split("-")[0]
                .replace("sw", "")
                .replace("A", "")
                .replace("B", "")
            )

        files = glob(join(self.anno_path, "**/*A-ms98-a-trans.text"), recursive=True)
        files.sort()

        paths = {}
        for p in files:
            session = _session_name(p)
            paths[session] = {
                "A": {
                    "trans": p,
                    "words": p.replace("A-ms98-a-trans.text", "A-ms98-a-word.text"),
                    "da_words": join(self.da_path, f"sw{session}A-word-da.csv"),
                },
                "B": {
                    "trans": p.replace("A-ms98-a-trans.text", "B-ms98-a-trans.text"),
                    "words": p.replace("A-ms98-a-trans.text", "B-ms98-a-word.text"),
                    "da_words": join(self.da_path, f"sw{session}B-word-da.csv"),
                },
            }
        return paths

    def read_utter_trans(self, path):
        """extract utterance annotation"""
        # trans = []
        trans = {}
        for row in read_txt(path):
            utt_idx, start, end, *text = row.split(" ")
            text = " ".join(text)
            start = float(start)
            end = float(end)
            if text == "[silence]":
                continue
            if text == "[noise]" or text == "[noise] [noise]":
                continue

            if "[noise]" in text:
                abort = True
                for w in text.split():
                    if w != "[noise]":
                        abort = False
                        break
                if abort:
                    continue
            # trans.append({"utt_idx": utt_idx, "start": start, "end": end, "text": text})
            trans[utt_idx] = {"start": start, "end": end, "text": text}
        return trans

    def read_word_trans(self, path):
        trans = []
        for row in read_txt(path):
            utt_idx, start, end, text = row.strip().split()
            start = float(start)
            end = float(end)
            if text == "[silence]":
                continue
            if text == "[noise]":
                continue
            trans.append({"utt_idx": utt_idx, "start": start, "end": end, "text": text})
        return trans

    def combine_utterance_and_words(
        self, speaker, words, utters, da_words, return_pandas=True
    ):
        utterances = []
        for utt_idx, utterance in utters.items():

            word_list, starts, ends = [], [], []
            for w in words:
                if utterance["end"] + 1 < w["start"]:
                    break

                if w["utt_idx"] == utt_idx:
                    word_list.append(w["text"])
                    starts.append(w["start"])
                    ends.append(w["end"])

            utterance["utt_idx"] = utt_idx
            utterance["speaker"] = speaker
            try:
                utterance["start"] = starts[0]
            except:
                print(utterance)
                print(starts)
                input()
            utterance["end"] = ends[-1]
            utterance["starts"] = starts
            utterance["ends"] = ends
            utterance["words"] = word_list

            if da_words is not None:
                das = da_words[da_words["utt_idx"] == utt_idx]
                utterance["da"] = das["da"].to_list()
                utterance["da_boi"] = das["boi"].to_list()

            utterances.append(utterance)

        if return_pandas:
            utterances = pd.DataFrame(utterances)
        return utterances

    def read_da_words(self, path):
        return pd.read_csv(
            path, names=["utt_idx", "start", "end", "word", "boi", "da", "da_idx"]
        )

    def get_session(self, session):
        if not isinstance(session, str):
            session = str(session)
        p = self.session_to_path[session]
        A_utters = self.read_utter_trans(p["A"]["trans"])
        A_words = self.read_word_trans(p["A"]["words"])

        B_utters = self.read_utter_trans(p["B"]["trans"])
        B_words = self.read_word_trans(p["B"]["words"])

        A_da_words = None
        B_da_words = None
        if exists(p["A"]["da_words"]):
            A_da_words = self.read_da_words(p["A"]["da_words"])
        if exists(p["B"]["da_words"]):
            B_da_words = self.read_da_words(p["B"]["da_words"])
        info = {
            "A": self.combine_utterance_and_words("A", A_words, A_utters, A_da_words),
            "B": self.combine_utterance_and_words("B", B_words, B_utters, B_da_words),
        }
        info["dialog"] = pd.concat((info["A"], info["B"])).sort_values("start")
        return info

    def iter_sessions(self):
        for session in self.sessions:
            yield session, self.get_session(session)


def load_bc_dataframe(path=BC_CSV):
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
        # "vad_list": _time,
        "phone_starts": _time,
        "phone_ends": _time,
        "words": _text,
        "da": _text,
        "da_boi": _text,
        "phones": _text,
    }
    return pd.read_csv(path, converters=converters)


# Required once
def load_bc_samples_from_original(path=JSON_PATH):
    import numpy as np

    reader = SWBReader()
    data = read_json(path)

    def utt_idx_to_session(utt_idx):
        return utt_idx.split("-")[0].replace("sw", "").replace("A", "").replace("B", "")

    def utt_idx_to_speaker(utt_idx):
        speaker = "B"
        if "A" in utt_idx:
            speaker = "A"
        return speaker

    labels = np.array(list(data.values()))
    utt_idx = np.array(list(data.keys()))
    idx = np.where(labels != "non-bc")[0]
    labels = labels[idx]
    utt_idx = utt_idx[idx]

    samples = []
    for ii, (u, l) in enumerate(zip(utt_idx, labels)):
        session = utt_idx_to_session(u)
        speaker = utt_idx_to_speaker(u)
        samples.append(
            {"session": session, "speaker": speaker, "utt_idx": u, "label": l}
        )
    bcs = pd.DataFrame(samples)

    sessions = bcs.session.unique()

    new_samples = []
    for session in tqdm(sessions):
        d = reader.get_session(session)["dialog"]
        tmp_bcs = bcs[bcs["session"] == session]
        for ii in range(len(tmp_bcs)):
            bc = tmp_bcs.iloc[ii]
            utt = d[d["utt_idx"] == bc.utt_idx].iloc[0].copy()
            utt["bc_label"] = bc.label
            utt["session"] = session
            new_samples.append(utt)
    df = pd.DataFrame(new_samples)
    df.to_csv(BC_CSV, index=False)
    return df


# TODO: add VAD so we can train regular objective at the same time
# TODO: utterance leading into bc?
# TODO: Onehot labels
class BackchannelDataset(Dataset):
    splits = ["train", "val", "test", "all"]

    def __init__(
        self,
        split="train",
        pre_context=15,
        post_context=5,
        sample_rate=16_000,
        frame_hz=50,
        bc_path=BC_CSV,
        audio_root="../../data/switchboard/audio",
    ):
        self.reader = SWBReader()
        self.split = split

        df = load_bc_dataframe(bc_path)
        if split == "all":
            self.df = df
        else:
            self.df = self.get_split_df(df, split)

        self.audio_duration = pre_context + post_context
        self.audio_nsamples = time_to_samples(pre_context + post_context, sample_rate)

        self.audio_root = audio_root
        self.pre_context = pre_context
        self.post_context = post_context
        self.frame_hz = frame_hz
        self.sample_rate = sample_rate

    def get_split_df(self, df, split):
        valid_sessions = read_txt(join(SWB_SPLIT_PATH, split + ".txt"))
        splits = []
        for session in valid_sessions:
            splits.append(df[df["session"] == int(session)])
        return pd.concat(splits)

    def __len__(self):
        return len(self.df)

    def get_sample(self, utt):
        audio_path = self.reader.session_to_audio_path(utt.session, self.audio_root)
        duration = get_audio_info(audio_path)["duration"]

        start_time = round(utt.start - self.pre_context, 2)
        rel_bc_start = self.pre_context
        end_time = round(utt.start + self.post_context, 2)

        pad_pre = -1
        if start_time < 0:
            pad_pre = time_to_samples(-start_time, self.sample_rate)
            start_time = 0
            rel_bc_start = utt.start

        pad_post = -1
        if end_time > duration:
            pad_post = time_to_samples(end_time - duration, self.sample_rate)
            end_time = duration

        utt_duration = utt.end - utt.start
        w, _ = load_waveform(
            audio_path,
            start_time=start_time,
            end_time=end_time,
            sample_rate=self.sample_rate,
        )

        if pad_pre > 0:
            z = torch.zeros((2, pad_pre))
            w = torch.cat((z, w), dim=-1)

        if pad_post > 0:
            z = torch.zeros((2, pad_post))
            w = torch.cat((w, z), dim=-1)

        w = w[..., : self.audio_nsamples]

        if w.shape[-1] != self.audio_nsamples:
            diff = self.audio_nsamples - w.shape[-1]
            w = torch.cat((w, torch.zeros(2, diff)), dim=-1)

        fs = time_to_frames(rel_bc_start, hop_time=1 / self.frame_hz)
        fe = time_to_frames(rel_bc_start + utt_duration, hop_time=1 / self.frame_hz)
        return {
            "waveform": w,
            "speaker": 0 if utt.speaker == "A" else 1,
            "bc_start_time": rel_bc_start,
            "bc_start_frame": fs,
            "bc_end_time": rel_bc_start + utt_duration,
            "bc_start_frame": fe,
            "label": utt.bc_label,
        }

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, float, str]]:
        utt = self.df.iloc[idx]

        sample = self.get_sample(utt)

        return sample


if __name__ == "__main__":
    from vap.plot_utils import plot_mel_spectrogram
    import matplotlib.pyplot as plt

    # load_bc_samples_from_original()

    dset = BackchannelDataset(split="train", pre_context=5)

    dloader = DataLoader(dset, batch_size=20, num_workers=0)
    batch = next(iter(dloader))

    for batch in dloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)}")
            else:
                print(f"{k}: {v}")
        for ii in range(batch["waveform"].shape[0]):
            w = batch["waveform"][ii]
            speaker = batch["speaker"][ii]
            bc_start = batch["bc_start_time"][ii]
            bc_end = batch["bc_end_time"][ii]
            plt.close("all")
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
            plot_mel_spectrogram(w, ax=ax, hop_time=0.01)
            ax[speaker].axvline(x=bc_start, linewidth=2, color="r")
            ax[speaker].axvline(x=bc_end, linewidth=2, color="r", label="bc")
            ax[speaker].legend()
            plt.tight_layout()
            plt.show()
