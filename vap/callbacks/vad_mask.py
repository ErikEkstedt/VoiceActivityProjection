import torch
import lightning as L
import einops
import random
import torchaudio.functional as AF

SAMPLE_RATE = 16000
FRAME_HZ = 50


@torch.no_grad()
def mask_around_vad(
    waveform: torch.Tensor,
    vad: torch.Tensor,
    vad_hz: int = FRAME_HZ,
    sample_rate: int = SAMPLE_RATE,
    scale: float = 0.1,
) -> torch.Tensor:
    assert (
        vad.shape[-1] == 2
    ), f"Expects vad of shape (B, N_FRAMES, 2) but got {vad.shape}"

    non_vad_mask = vad.permute(0, 2, 1).logical_not().float()  # -> B, 2, N_frames

    B, C, _ = waveform.shape
    if B > 1 and C > 1:
        w_tmp = einops.rearrange(waveform, "b c s -> (b c) s")
        non_vad_mask = einops.rearrange(non_vad_mask, "b c f -> (b c) f")
        if vad_hz != sample_rate:
            non_vad_mask = AF.resample(
                non_vad_mask, orig_freq=vad_hz, new_freq=sample_rate
            )
            non_vad_mask = non_vad_mask > 0.5
        non_vad_mask = non_vad_mask[..., : waveform.shape[-1]]
        # z_mask *= scale
        w_tmp[non_vad_mask] *= scale
        # w_tmp = w_tmp * v_mask[:, : w_tmp.shape[-1]]
        waveform = einops.rearrange(w_tmp, "(b c) s -> b c s", b=B, c=C)
    else:
        if vad_hz != sample_rate:
            non_vad_mask = AF.resample(
                non_vad_mask, orig_freq=vad_hz, new_freq=sample_rate
            )
        if C == 1:
            non_vad_mask = non_vad_mask.sum(-2).unsqueeze(1)

        non_vad_mask = non_vad_mask > 0.5
        non_vad_mask = non_vad_mask[..., : waveform.shape[-1]]
        # z_mask *= scale
        waveform[non_vad_mask] *= scale
        # waveform = waveform * non_vad_mask[:, :, : waveform.shape[-1]]
    return waveform


class VADMaskCallback(L.Callback):
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
        sample_rate: int = SAMPLE_RATE,
        frame_hz: int = FRAME_HZ,
        on_train: bool = True,
        on_val: bool = False,
        on_test: bool = False,
    ):
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.probability = probability
        self.on_train = on_train
        self.on_val = on_val
        self.on_test = on_test

    def mask(self, batch):
        batch["waveform"] = mask_around_vad(batch["waveform"], batch["vad"])
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
