import lightning as L
import random


def flip_batch_channels(batch):
    """flipped version of the batch-samples"""
    for k, v in batch.items():
        if k == "vad":
            v = v.flip(-1)  # (B, N_FRAMES, 2)
        elif k == "waveform":
            if v.shape[1] == 2:  # stereo audio
                v = v.flip(-2)  # (B, 2, N_SAMPLES)
        batch[k] = v
    return batch


class FlipChannelCallback(L.Callback):
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

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_train and random.random() < self.probability:
            batch = flip_batch_channels(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_test:
            batch = flip_batch_channels(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_val:
            batch = flip_batch_channels(batch)
