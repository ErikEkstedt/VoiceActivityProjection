import torch
from torch.utils.data import Dataset
from os.path import exists
from typing import Any, Dict, List, Tuple

from vap.audio import time_to_frames, load_waveform, get_audio_info
from vap.utils import read_json
from vap_turn_taking.utils import vad_list_to_onehot, get_activity_history

try:
    from textgrids import TextGrid
except ModuleNotFoundError:
    raise Warning("ImportError 'textgrids' not found. 'pip install praat-textgrids'.")


class PhraseDataset(Dataset):
    def __init__(
        self,
        phrase_path: str,
        # AUDIO #################################
        sample_rate: int = 16000,
        audio_mono: bool = True,
        audio_duration: float = 10.0,
        # VAD #################################
        vad: bool = True,
        vad_hz: int = 50,
        vad_horizon: int = 2,
        vad_history: bool = True,
        vad_history_times: List[int] = [60, 30, 10, 5],
    ):
        super().__init__()
        self.data = read_json(phrase_path)
        self.indices = self._map_phrases_to_idx()

        # Audio (waveforms)
        self.sample_rate = sample_rate
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration

        # VAD parameters
        self.vad = vad  # use vad or not
        self.vad_hz = vad_hz
        self.vad_hop_time = 1.0 / vad_hz
        self.horizon_time = vad_horizon
        self.vad_horizon = time_to_frames(vad_horizon, hop_time=self.vad_hop_time)

        # Vad history
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times
        self.vad_history_frames = (
            (torch.tensor(vad_history_times) / self.vad_hop_time).long().tolist()
        )

    def _map_phrases_to_idx(self) -> List:
        indices = []
        for example, v in self.data.items():
            for long_short, vv in v.items():
                for gender, sample_list in vv.items():
                    for ii in range(len(sample_list)):
                        indices.append([example, long_short, gender, ii])
        return indices

    def audio_path_text_grid_path(self, path: str) -> str:
        """
        audio_path:
            "dataset_phrases/audio/student_short_female_en-US-Wavenet-G.wav"
        textgrid_path:
            "dataset_phrases/alignment/student_short_female_en-US-Wavenet-G.TextGrid"
        """
        return path.replace("/audio/", "/alignment/").replace(".wav", ".TextGrid")

    def read_text_grid(self, path: str) -> dict:
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

    def tg_words_to_vad(self, words: List[Tuple[float, float, str]]) -> List[List]:
        vad_list = []
        for s, e, _ in words:
            vad_list.append([s, e])
        return [vad_list, []]

    def __len__(self) -> int:
        return len(self.indices)

    def sample_to_duration_sample(self, sample):
        """
        Load duration average waveform and duration average textgrids (word timings)
        """
        new_sample = {"example": sample["example"]}
        # change audio_path
        tg_path = self.audio_path_text_grid_path(sample["audio_path"])

        new_sample["audio_path"] = sample["audio_path"].replace(
            "/audio/", "/duration_audio/"
        )
        new_sample["waveform"], _ = load_waveform(
            new_sample["audio_path"],
            sample_rate=self.sample_rate,
            mono=self.audio_mono,
        )

        # change tg_path
        tg_path = tg_path.replace("/alignment/", "/duration_alignment/")
        tg = self.read_text_grid(tg_path)
        new_sample["phones"] = tg["phones"]
        new_sample["starts"], new_sample["ends"], new_sample["words"] = [], [], []
        vad_list = [[], []]
        for start, end, word in tg["words"]:
            new_sample["starts"].append(start)
            new_sample["ends"].append(end)
            new_sample["words"].append(word)
            vad_list[0].append([start, end])

        new_sample["vad"] = vad_list
        new_sample = self.get_vad(new_sample)
        return new_sample

    def get_vad(self, sample):
        duration = get_audio_info(sample["audio_path"])["duration"]
        end_frame = time_to_frames(duration, self.vad_hop_time)
        all_vad_frames = vad_list_to_onehot(
            sample["vad"],
            hop_time=self.vad_hop_time,
            duration=duration,
            channel_last=True,
        )

        if self.vad_history:
            # history up until the current features arrive
            vad_history, _ = get_activity_history(
                all_vad_frames,
                bin_end_frames=self.vad_history_frames,
                channel_last=True,
            )
            # vad history is always defined as speaker 0 activity
            sample["vad_history"] = vad_history[:end_frame][..., 0].unsqueeze(0)

        if end_frame + self.vad_horizon > all_vad_frames.shape[0]:
            lookahead = torch.zeros(
                (self.vad_horizon + 1, 2)
            )  # add horizon after end (silence)
            all_vad_frames = torch.cat((all_vad_frames, lookahead))
        sample["vad"] = all_vad_frames[: end_frame + self.vad_horizon].unsqueeze(0)
        return sample

    def get_sample(
        self, example: str, long_short: str, gender: str, id: int
    ) -> Dict[str, Any]:
        """ """
        # dict_keys(['text', 'audio_path', 'gender', 'words', 'starts', 'size', 'tts', 'name', 'vad'])
        sample = self.data[example][long_short][gender][id]
        sample["dataset"] = "phrases"
        sample["example"] = example

        # TextGrids for phoneme times
        tg_path = self.audio_path_text_grid_path(sample["audio_path"])
        if exists(tg_path):
            tg = self.read_text_grid(tg_path)
            sample["phones"] = tg["phones"]
            sample["starts"], sample["ends"], sample["words"] = [], [], []
            for start, end, word in tg["words"]:
                sample["starts"].append(start)
                sample["ends"].append(end)
                sample["words"].append(word)

        # Load audio
        sample["waveform"], _ = load_waveform(
            sample["audio_path"],
            sample_rate=self.sample_rate,
            mono=self.audio_mono,
        )

        if sample["waveform"].ndim == 2:
            sample["waveform"] = sample["waveform"].unsqueeze(1)

        # VAD-frame of relevant part
        if self.vad:
            sample = self.get_vad(sample)

        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example, long_short, gender, nidx = self.indices[idx]
        return self.get_sample(example, long_short, gender, nidx)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from vap.plot_utils import plot_phrases_sample
    from vap.model import VAPModel

    model = VAPModel.load_from_checkpoint("example/50hz_48_10s-epoch20-val_1.85.ckpt")
    model.eval()

    dset = PhraseDataset(phrase_path="dataset_phrases/phrases.json")
    print("len(dset): ", len(dset))
    sample = dset[0]
    sample = dset.get_sample("student", "long", "female", 0)
    loss, out, probs, batch = model.output(sample)

    dur_sample = dset.sample_to_duration_sample(sample)

    loss, out, probs, batch = model.output(dur_sample)

    fig, ax = plot_phrases_sample(
        sample, probs, frame_hz=dset.vad_hz, sample_rate=dset.sample_rate
    )
    plt.show()
    # plt.pause(0.1)
