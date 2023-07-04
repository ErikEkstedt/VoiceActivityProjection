from os.path import exists
import pandas as pd
import json
from typing import Optional, Mapping

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import lightning as L

from vap.audio import load_waveform
from vap.utils import vad_list_to_onehot


class VapDataset(Dataset):
    def __init__(
        self,
        path,
        horizon: float = 2,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
    ):
        self.path = path
        self.df = self.load_df(path)

        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.horizon = horizon
        self.mono = mono

    def load_df(self, path):
        def _vl(x):
            return json.loads(x)

        def _session(x):
            return str(x)

        converters = {
            "vad_list": _vl,
            "session": _session,
        }
        return pd.read_csv(path, converters=converters)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Mapping[str, Tensor]:
        d = self.df.iloc[idx]
        # Duration can be 19.99999999999997 for some clips and result in wrong vad-shape
        # so we round it to nearest second
        dur = round(d["end"] - d["start"])
        w, _ = load_waveform(
            d["audio_path"],
            start_time=d["start"],
            end_time=d["end"],
            sample_rate=self.sample_rate,
            mono=self.mono,
        )
        vad = vad_list_to_onehot(
            d["vad_list"], duration=dur + self.horizon, frame_hz=self.frame_hz
        )

        return {
            "session": d["session"],
            "waveform": w,
            "vad": vad,
            "dataset": d["dataset"],
        }


class VapDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        horizon: float = 2,
        sample_rate: int = 16000,
        frame_hz: int = 50,
        mono: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Files
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # values
        self.mono = mono
        self.horizon = horizon
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz

        # DataLoder
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tTrain: {self.train_path}"
        s += f"\n\tVal: {self.val_path}"
        s += f"\n\tTest: {self.test_path}"
        s += f"\n\tHorizon: {self.horizon}"
        s += f"\n\tSample rate: {self.sample_rate}"
        s += f"\n\tFrame Hz: {self.frame_hz}"
        s += f"\nData"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"
        return s

    def prepare_data(self):
        if self.train_path is not None:
            assert self.path_exists("train"), f"No TRAIN file found: {self.train_path}"

        if self.val_path is not None:
            assert self.path_exists("val"), f"No TRAIN file found: {self.train_path}"

        if self.test_path is not None:
            assert exists(self.test_path), f"No TEST file found: {self.test_path}"

    def path_exists(self, split):
        path = getattr(self, f"{split}_path")
        if path is None:
            return False

        if not exists(path):
            return False
        return True

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""

        if stage in (None, "fit"):
            assert self.path_exists("train"), f"Train path not found: {self.train_path}"
            assert self.path_exists("val"), f"Val path not found: {self.val_path}"
            self.train_dset = VapDataset(
                self.train_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
            )
            self.val_dset = VapDataset(
                self.val_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
            )

        if stage in (None, "test"):
            assert self.path_exists("test"), f"Test path not found: {self.test_path}"
            self.test_dset = VapDataset(
                self.test_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    from tqdm import tqdm

    dset = VapDataset(path="example/data/sliding_dev.csv")

    d = dset[0]
    dm = VapDataModule(
        train_path="example/data/sliding_dev.csv",
        val_path="example/data/sliding_dev.csv",
        test_path="example/data/sliding_dev.csv",
        batch_size=4,
        num_workers=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    print(dm)
    print("Train: ", len(dm.train_dset))
    print("Val: ", len(dm.val_dset))

    dloader = dm.train_dataloader()
    for batch in tqdm(dloader, total=len(dloader)):
        pass
