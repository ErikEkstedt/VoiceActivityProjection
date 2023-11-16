from typing import Optional, Mapping, Iterable, Callable

import torch
import torch.nn as nn

# import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import lightning as L

from vap.modules.transformer import VapStereoTower
from vap.modules.modules import ProjectionLayer, Combinator
from vap.objective import VAPObjective
from vap.metrics import VAPMetric
from vap.utils.utils import everything_deterministic


Batch = Mapping[str, Tensor]

everything_deterministic()


class VAP(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        bin_times: list[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
    ):
        super().__init__()
        self.enc_dim = getattr(encoder, "dim")
        self.dim: int = getattr(transformer, "dim")
        self.frame_hz = frame_hz
        self.objective = VAPObjective(bin_times=bin_times, frame_hz=frame_hz)

        # Layers
        self.encoder = encoder
        self.feature_projection = (
            ProjectionLayer(self.enc_dim, self.dim)
            if encoder.dim != transformer.dim
            else nn.Identity()
        )
        self.transformer = transformer
        self.combinator = Combinator(dim=self.dim, activation="GELU")

        # Output Projections
        self.va_classifier = nn.Linear(self.dim, 1)
        self.vap_head = nn.Linear(self.dim, self.objective.n_classes)

    def extract_labels(self, vad: Tensor) -> Tensor:
        return self.objective.get_labels(vad)

    @torch.inference_mode()
    def probs(
        self,
        waveform: Tensor,
        vad: Optional[Tensor] = None,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ):
        """"""
        out = self(waveform)

        # Probabilites and Entropy
        probs = out["logits"].softmax(dim=-1)
        h = -probs * probs.log2()  # Entropy
        h = h.sum(dim=-1).cpu()  # average entropy per frame
        ret = {"probs": probs, "vad": out["vad"].sigmoid(), "H": h}

        # Next speaker aggregate probs
        probs_agg = self.objective.aggregate_probs(probs, now_lims, future_lims)
        ret.update(probs_agg)

        # If ground truth voice activity is known we can calculate the loss
        if vad is not None:
            labels = self.objective.get_labels(vad)
            ret["loss"] = self.objective.loss_vap(
                out["logits"], labels, reduction="none"
            ).cpu()
        return ret

    def encode_audio(self, audio: torch.Tensor) -> tuple[Tensor, Tensor]:
        assert (
            audio.shape[1] == 2
        ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
        x1 = self.encoder(audio[:, :1])  # speaker 1
        x2 = self.encoder(audio[:, 1:])  # speaker 2
        return x1, x2

    def forward(self, waveform: Tensor, **kwargs):
        x1, x2 = self.encode_audio(waveform)
        s1 = self.feature_projection(x1)
        s2 = self.feature_projection(x2)
        p1, p2 = self.transformer(s1, s2)
        p = self.combinator(p1, p2)

        # VAD logits
        vap_logits = self.vap_head(p)

        # VAD logits
        v1 = self.va_classifier(p1)
        v2 = self.va_classifier(p2)
        vad_logits = torch.cat((v1, v2), dim=-1)

        return {
            "logits": vap_logits,
            "vad": vad_logits,
            "s1": s1,
            "s2": s2,
            "p1": p1,
            "p2": p2,
            "p": p,
        }


class VAPModule(L.LightningModule):
    def __init__(
        self,
        model: VAP,
        optim_fn: Optional[Callable[Iterable[Parameter], Optimizer]] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
        train_metric: Optional[VAPMetric] = None,
        val_metric: Optional[VAPMetric] = None,
        test_metric: Optional[VAPMetric] = None,
    ):
        super().__init__()
        self.model = model
        self.optim: Optional[Optimizer] = (
            optim_fn(self.model.parameters()) if optim_fn else None
        )
        self.lr_scheduler: Optional[_LRScheduler] = (
            lr_scheduler(self.optim) if lr_scheduler else None
        )
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.save_hyperparameters()  # ignore=["model"])

    def forward(self, waveform: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        return self.model(waveform, *args, **kwargs)

    @staticmethod
    def load_model(path: str, *args, **kwargs) -> VAP:
        return VAPModule.load_from_checkpoint(path, *args, **kwargs).model

    def configure_optimizers(self) -> dict:
        lr_scheduler = {
            "scheduler": self.lr_scheduler,
            "monitor": "val_loss",
        }
        return {"optimizer": self.optim, "lr_scheduler": lr_scheduler}

    def metric_update(self, logits, vad, split: str = "val"):
        m = getattr(self, f"{split}_metric", None)
        if m:
            probs = self.model.objective.get_probs(logits)
            m.update_batch(probs, vad)

    def metric_finalize(self, split: str = "val") -> None:
        m = getattr(self, f"{split}_metric", None)
        if m:
            scores = m.compute()
            m.reset()
            for event_name, score in scores.items():
                self.log(f"{split}_f1_{event_name}", score["f1"], sync_dist=True)
                self.log(f"{split}_acc_{event_name}_0", score["acc"][0], sync_dist=True)
                self.log(f"{split}_acc_{event_name}_1", score["acc"][1], sync_dist=True)

    def _step(
        self, batch: Batch, split: str = "train", reduction: str = "mean"
    ) -> Mapping[str, torch.Tensor]:
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
        out["va_loss"] = self.model.objective.loss_vad(out["vad"], batch["vad"])
        self.metric_update(out["logits"], batch["vad"], split=split)

        # Log results
        batch_size = batch["waveform"].shape[0]
        self.log(
            f"{split}_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True
        )
        self.log(
            f"{split}_loss_va",
            out["va_loss"],
            batch_size=batch_size,
            sync_dist=True,
        )
        return out

    def training_step(self, batch: Batch, *args, **kwargs):
        out = self._step(batch)
        loss = out["vap_loss"] + out["va_loss"]
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs):
        _ = self._step(batch, split="val")

    def test_step(self, batch: Batch, *args, **kwargs):
        _ = self._step(batch, split="test")

    def on_train_epoch_end(self) -> None:
        self.metric_finalize(split="train")

    def on_validation_epoch_end(self) -> None:
        self.metric_finalize(split="val")

    def on_test_epoch_end(self) -> None:
        self.metric_finalize(split="test")


if __name__ == "__main__":

    from vap.modules.encoder import EncoderCPC
    from vap.data.datamodule import VAPDataModule

    encoder = EncoderCPC()
    transformer = VapStereoTower()
    model = VAP(encoder, transformer)
    # x = torch.randn(2, 2, 16000)
    # out = model(x)
    module = VAPModule(
        model,
        optim_fn=torch.optim.AdamW,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    dm = VAPDataModule(
        train_path="data/splits/fisher_swb/train_sliding.csv",
        val_path="data/splits/fisher_swb/val_sliding.csv",
        num_workers=0,
        prefetch_factor=None,
    )

    trainer = L.Trainer(fast_dev_run=1)
    trainer.fit(module, dm)
