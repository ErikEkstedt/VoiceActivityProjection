import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import lightning as L
from typing import Optional, Mapping, Iterable, Callable

from vap.encoder import EncoderCPC
from vap.events import TurnTakingEvents
from vap.objective import ObjectiveVAP
from vap.modules import GPT, GPTStereo
from vap.utils import (
    everything_deterministic,
    vad_fill_silences,
    vad_omit_spikes,
)


class TransformerStereo(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        self_layers: int = 1,
        cross_layers: int = 3,
        num_heads: int = 4,
        dff_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.self_layers = self_layers
        self.cross_layers = cross_layers
        self.num_heads = num_heads
        self.dff_k = dff_k
        self.dropout = dropout

        # Single channel
        self.ar_channel = GPT(
            dim=dim,
            dff_k=dff_k,
            num_layers=self_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=dim,
            dff_k=dff_k,
            num_layers=cross_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self, x1: Tensor, x2: Tensor, attention: bool = False
    ) -> Mapping[str, Tensor]:
        o1 = self.ar_channel(x1, attention=attention)  # ["x"]
        o2 = self.ar_channel(x2, attention=attention)  # ["x"]
        out = self.ar(o1["x"], o2["x"], attention=attention)

        if attention:
            out["cross_self_attn"] = out["self_attn"]
            out["self_attn"] = torch.stack([o1["attn"], o2["attn"]], dim=1)
            out["cross_attn"] = out["cross_attn"]
        return out


class VAP(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        bin_times: list[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
    ):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.objective = ObjectiveVAP(bin_times=bin_times, frame_hz=frame_hz)
        self.frame_hz = frame_hz
        self.dim: int = getattr(self.transformer, "dim", 256)

        # Outputs
        # Voice activity objective -> x1, x2 -> logits ->  BCE
        self.va_classifier = nn.Linear(self.dim, 1)
        self.vap_head = nn.Linear(self.dim, self.objective.n_classes)

    @property
    def horizon_time(self):
        return self.objective.horizon_time

    def extract_labels(self, vad: Tensor) -> Tensor:
        return self.objective.get_labels(vad)

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    def encode_audio(self, audio: torch.Tensor) -> tuple[Tensor, Tensor]:
        assert (
            audio.shape[1] == 2
        ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
        x1 = self.encoder(audio[:, :1])  # speaker 1
        x2 = self.encoder(audio[:, 1:])  # speaker 2
        return x1, x2

    def head(self, x: Tensor, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        v1 = self.va_classifier(x1)
        v2 = self.va_classifier(x2)
        vad = torch.cat((v1, v2), dim=-1)
        logits = self.vap_head(x)
        return logits, vad

    def forward(self, waveform: Tensor, attention: bool = False) -> dict[str, Tensor]:
        x1, x2 = self.encode_audio(waveform)
        out = self.transformer(x1, x2, attention=attention)
        logits, vad = self.head(out["x"], out["x1"], out["x2"])
        out["logits"] = logits
        out["vad"] = vad
        return out

    def entropy(self, probs: Tensor) -> Tensor:
        """
        Calculate entropy over each projection-window prediction (i.e. over
        frames/time) If we have C=256 possible states the maximum bit entropy
        is 8 (2^8 = 256) this means that the model have a one in 256 chance
        to randomly be right. The model can't do better than to uniformly
        guess each state, it has learned (less than) nothing. We want the
        model to have low entropy over the course of a dialog, "thinks it
        understands how the dialog is going", it's a measure of how close the
        information in the unseen data is to the knowledge encoded in the
        training data.
        """
        h = -probs * probs.log2()  # Entropy
        return h.sum(dim=-1).cpu()  # average entropy per frame

    def aggregate_probs(
        self,
        probs: Tensor,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ) -> dict[str, Tensor]:
        # first two bins
        p_now = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=now_lims[0], to_bin=now_lims[-1]
        ).cpu()
        p_future = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=future_lims[0], to_bin=future_lims[1]
        ).cpu()
        # P over all
        max_idx = self.objective.n_bins - 1
        pa = self.objective.probs_next_speaker_aggregate(probs, 0, max_idx).cpu()
        p = []
        for i in range(0, max_idx + 1):
            p.append(self.objective.probs_next_speaker_aggregate(probs, i, i).cpu())
        p = torch.stack(p)
        return {
            "p_now": p_now,
            "p_future": p_future,
            "p_all": pa,
            "p": p,
        }

    @torch.inference_mode()
    def probs(
        self,
        waveform: Tensor,
        vad: Optional[Tensor] = None,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ) -> dict[str, Tensor]:
        """"""
        out = self(waveform)
        probs = out["logits"].softmax(dim=-1)
        vap_vad = out["vad"].sigmoid()
        h = self.entropy(probs)
        ret = {
            "probs": probs,
            "vad": vap_vad,
            "H": h,
        }

        # Next speaker aggregate probs
        probs_agg = self.aggregate_probs(probs, now_lims, future_lims)
        ret.update(probs_agg)

        # If ground truth voice activity is known we can calculate the loss
        if vad is not None:
            labels = self.objective.get_labels(vad)
            ret["loss"] = self.objective.loss_vap(
                out["logits"], labels, reduction="none"
            ).cpu()
        return ret

    @torch.inference_mode()
    def vad(
        self,
        waveform: Tensor,
        max_fill_silence_time: float = 0.02,
        max_omit_spike_time: float = 0.02,
        vad_cutoff: float = 0.5,
    ) -> Tensor:
        """
        Extract (binary) Voice Activity Detection from model
        """
        vad = (self(waveform)["vad"].sigmoid() >= vad_cutoff).float()
        for b in range(vad.shape[0]):
            # TODO: which order is better?
            vad[b] = vad_fill_silences(
                vad[b], max_fill_time=max_fill_silence_time, frame_hz=self.frame_hz
            )
            vad[b] = vad_omit_spikes(
                vad[b], max_omit_time=max_omit_spike_time, frame_hz=self.frame_hz
            )
        return vad


class VAPModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optim_fn: Callable[Iterable[Parameter], Optimizer],
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        super().__init__()
        self.model = model
        self.optim: Optimizer = optim_fn(self.model.parameters())
        self.lr_scheduler: Optional[_LRScheduler] = (
            lr_scheduler(self.optim) if lr_scheduler else None
        )
        self.save_hyperparameters()  # ignore=["model"])

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        return self.model(waveform)

    def configure_optimizers(self) -> dict:
        lr_scheduler = {
            "scheduler": self.lr_scheduler,
            "monitor": "val_loss",
        }
        return {"optimizer": self.optim, "lr_scheduler": lr_scheduler}

    def _step(self, batch: dict, reduction: str = "mean") -> dict[str, torch.Tensor]:
        """
        Arguments:
            batch:      dict, containing 'waveform', va, va_history

        Returns:
            out:        dict, ['logits', 'vad', 'vap_loss', 'vad_loss']
        """
        labels = self.model.extract_labels(batch["vad"])
        out = self(batch["waveform"])
        out["vap_loss"] = self.model.objective.loss_vap(
            out["logits"], labels, reduction=reduction
        )
        out["vad_loss"] = self.model.objective.loss_vad(out["vad"], batch["vad"])
        return out

    def training_step(self, batch, **_):
        out = self._step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)
        loss = out["vap_loss"] + out["vad_loss"]
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **_):
        out = self._step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("val_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("val_loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)

        # # Event Metrics
        # if self.event_extractor is not None:
        #     events = self.event_extractor(batch["vad"])
        #     # probs = self.zero_shot.get_probs(out["logits"], batch["vad"])
        #     # preds, targets = self.zero_shot.extract_prediction_and_targets(
        #     #     p=probs["p"], p_bc=probs["p_bc"], events=events
        #     # )
        #     probs = self.objective.get_probs(out["logits"])
        #     preds, targets = self.objective.extract_prediction_and_targets(
        #         p_now=probs["p_now"], p_fut=probs["p_future"], events=events
        #     )
        #     self.val_metrics = self.metrics_step(preds, targets, self.val_metrics)


if __name__ == "__main__":

    encoder = EncoderCPC()
    transformer = TransformerStereo()
    model = VAP(encoder, transformer)

    # trainer = pl.Trainer()
    # trainer.fit(model)
    # trainer.save_checkpoint("test.ckpt")
    x = torch.randn((1, 2, 160_000))
    out = model(x)

    # commandline argument to print distribution
    # in bash
