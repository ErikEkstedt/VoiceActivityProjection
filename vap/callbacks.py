import pytorch_lightning as pl
import wandb
import random

import vap.transforms as VT


class AudioAugmentationCallback(pl.Callback):
    def __init__(
        self,
        probability: float = 0.5,
        noise_amplitude: float = 0.01,
        pitch_steps: list[int] = [-2, -1, 1, 2],
        freq_mask_param: int = 100,
        iid_masks: bool = True,
        sample_rate: int = 16_000,
        device: str = "cpu",
    ):
        self.augmentation = VT.Augmentation(
            probability=probability,
            noise_amplitude=noise_amplitude,
            pitch_steps=pitch_steps,
            freq_mask_param=freq_mask_param,
            iid_masks=iid_masks,
            sample_rate=sample_rate,
            device=device,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        batch["waveform"] = self.augmentation(batch["waveform"])


class SymmetricSpeakersCallback(pl.Callback):
    """
    Randomly "flips" the speakers such that we get a fair evaluation not dependent on the
    biased speaker-order / speaker-activity

    The audio is mono which requires no change.

    The only change we apply is to flip the channels in the VAD-tensor and get the corresponding VAD-history
    which is defined as the ratio of speaker 0 (i.e. vad_history_flipped = 1 - vad_history)
    """

    def __init__(
        self,
        probability: float = 0.5,
        on_train: bool = True,
        on_val: bool = False,
        on_test: bool = False,
    ):
        self.probability = probability
        self.on_train = on_train
        self.on_val = on_val
        self.on_test = on_test

    def get_flipped_batch(self, batch):
        """Appends a flipped version of the batch-samples"""
        for k, v in batch.items():
            if k == "vad":
                v = v.flip(-1)  # (B, N_FRAMES, 2)
            elif k == "waveform":
                if v.shape[1] == 2:  # stereo audio
                    v = v.flip(-2)  # (B, 2, N_SAMPLES)
                else:
                    continue
            batch[k] = v
        return batch

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_train and random.random() < self.probability:
            batch = self.get_flipped_batch(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_test:
            batch = self.get_flipped_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_val:
            batch = self.get_flipped_batch(batch)
