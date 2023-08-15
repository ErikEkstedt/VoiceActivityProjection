import torch
import lightning as L
import random
import torch.nn.functional as F

SAMPLE_RATE = 16000
FRAME_HZ = 50


@torch.no_grad()
def vad_mask_batch(
    waveform: torch.Tensor,
    vad: torch.Tensor,
    scale: float = 0,
) -> torch.Tensor:
    """"""
    # Expand vad to match waveform
    # Use the vad activations to create a mask where there is SILENCE (logical_not))
    vad_mask = vad.permute(0, 2, 1)

    # If mono audio we combine
    if waveform.shape[1] == 1:
        vad_mask = vad_mask.sum(1).unsqueeze(1).clamp(min=0, max=1)  # -> B, 1, N_frames

    nv = F.interpolate(vad_mask, size=waveform.shape[-1])  # -> B, 2, N_samples

    # Scale the appropriate values
    if scale > 0:
        waveform[torch.where(nv)] *= scale
    else:
        waveform[torch.where(nv)] = 0
    return waveform


class VADMaskCallback(L.Callback):
    def __init__(
        self,
        probability: float = 0.5,
        scale: float = 0,
        on_train: bool = True,
        on_val: bool = False,
        on_test: bool = False,
        *args,
        **kwargs
    ):
        self.probability = probability
        self.scale = scale
        self.on_train = on_train
        self.on_val = on_val
        self.on_test = on_test

    def mask(self, batch):
        batch["waveform"] = vad_mask_batch(
            batch["waveform"], batch["vad"], scale=self.scale
        )
        return batch

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_train and random.random() < self.probability:
            batch = self.mask(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_val and random.random() < self.probability:
            batch = self.mask(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs) -> None:
        if self.on_test and random.random() < self.probability:
            batch = self.mask(batch)


if __name__ == "__main__":

    from vap.utils.utils import vad_list_to_onehot

    DUR = 5
    B = 4
    C = 2
    vad_list = [[[0, DUR / 4], [DUR / 2, 3 * DUR / 4]], []]
    vad = []
    for i in range(B):
        vad.append(vad_list_to_onehot(vad_list, duration=DUR, frame_hz=50))
    vad = torch.stack(vad)
    w = torch.ones(B, C, DUR * 16000)
    w_orig = w.clone()
    scale = 0
    print("w: ", tuple(w.shape))
    print("vad: ", tuple(vad.shape))

    new_w = vad_mask_batch(w, vad, scale=scale)
