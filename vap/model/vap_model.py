from typing import Optional, Mapping

import torch
import torch.nn as nn
from torch import Tensor

# import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy

from vap.modules.transformer import VapStereoTower
from vap.modules.modules import ProjectionLayer, Combinator
from vap.objective import VAPObjective
from vap.utils.utils import everything_deterministic
from vap.events.new_events import HoldShiftEvents, BackchannelEvents


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

    @property
    def device(self):
        return next(self.parameters()).device

    def extract_labels(self, vad: Tensor) -> Tensor:
        return self.objective.get_labels(vad)

    def prediction(self, logits):
        # Probabilites and Entropy
        probs = logits.softmax(dim=-1)
        h = -probs * probs.log2()  # Entropy
        h = h.sum(dim=-1).cpu()  # average entropy per frame

        # Next speaker aggregate probs
        p_now = self.objective.probs_next_speaker_aggregate(probs, from_bin=0, to_bin=1)
        p_fut = self.objective.probs_next_speaker_aggregate(probs, from_bin=2, to_bin=3)
        return {
            "p_now": p_now,
            "p_fut": p_fut,
            "probs": probs,
            "H": h,
        }

    @torch.inference_mode()
    def probs(
        self,
        waveform: Tensor,
        vad: Optional[Tensor] = None,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ):
        """"""
        out = self(waveform)  # logits, vad, s1, s2, p1, p2, p

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


class VapMetrics:
    def __init__(self):
        self.reaction_time = 0.2
        self.frame_hz = 50
        self.start_offset = int(self.reaction_time * self.frame_hz)
        self.n_frames = 2

        self.hs = HoldShiftEvents(
            solo_pre_time=1.0,
            solo_post_time=1.0,
            min_context_time=0,
            min_silence_time=0.5,
            condition="pause_inclusion",
            frame_hz=50,
        )
        self.bc = BackchannelEvents(
            max_bc_time=1.0,
            solo_pre_time=1.0,
            solo_post_time=1.0,
            min_context_time=0,
            min_silence_time=0.2,
            frame_hz=50,
        )

        # Holds
        self.hold_now_accuracy = Accuracy(task="binary", threshold=0.5)
        self.hold_fut_accuracy = Accuracy(task="binary", threshold=0.5)
        self.hold_both_accuracy = Accuracy(task="binary", threshold=0.5)
        self.hold_total = 0

        # Shifts
        self.shift_now_accuracy = Accuracy(task="binary", threshold=0.5)
        self.shift_fut_accuracy = Accuracy(task="binary", threshold=0.5)
        self.shift_both_accuracy = Accuracy(task="binary", threshold=0.5)
        self.shift_total = 0

    @staticmethod
    def extract_prediction(events, pn, pf, start_offset, n_frames):
        ev_preds = {"now": [], "fut": []}

        if len(events) == 0:
            return {"now": torch.empty(0), "fut": torch.empty(0)}

        for s, cidx, bidx in zip(events[:, 0], events[:, -2], events[:, -1]):
            s += start_offset
            e = s + n_frames
            ev_preds["now"].append((cidx - pn[bidx, s:e].mean()).abs())
            ev_preds["fut"].append((cidx - pf[bidx, s:e].mean()).abs())
        ev_preds = {k: torch.stack(v) for k, v in ev_preds.items()}
        return ev_preds

    def batch_update(self, pn, pf, vad):
        n_max = pn.shape[1]
        if pn.device != self.hold_now_accuracy.device:
            for ev_name in ["hold", "shift"]:
                for proj_name in ["now", "fut", "both"]:
                    s = f"{ev_name}_{proj_name}_accuracy"
                    getattr(self, s).to(pn.device)

        # extract events
        hold, shift = self.hs(vad[:, :n_max])

        # Extract predictions
        hold_preds = VapMetrics.extract_prediction(
            hold, pf, pf, self.start_offset, self.n_frames
        )

        if len(hold_preds["now"]) > 0:
            self.hold_total += len(hold_preds["now"])

            # now
            self.hold_now_accuracy.update(
                hold_preds["now"], torch.ones_like(hold_preds["now"])
            )
            self.hold_fut_accuracy.update(
                hold_preds["fut"], torch.ones_like(hold_preds["fut"])
            )
            self.hold_both_accuracy.update(
                (hold_preds["fut"] + hold_preds["now"]) / 2,
                torch.ones_like(hold_preds["fut"]),
            )

        shift_preds = VapMetrics.extract_prediction(
            shift, pf, pf, self.start_offset, self.n_frames
        )
        if len(shift_preds["now"]) > 0:
            self.shift_total += len(shift_preds["now"])
            # Shift
            self.shift_now_accuracy.update(
                shift_preds["now"], torch.ones_like(shift_preds["now"])
            )
            self.shift_fut_accuracy.update(
                shift_preds["fut"], torch.ones_like(shift_preds["fut"])
            )
            self.shift_both_accuracy.update(
                (shift_preds["fut"] + shift_preds["now"]) / 2,
                torch.ones_like(shift_preds["fut"]),
            )

    def epoch_update(self):
        acc = {}
        for ev_name in ["hold", "shift"]:
            for proj_name in ["now", "fut", "both"]:
                s = f"{ev_name}_{proj_name}_accuracy"
                m = getattr(self, s)
                tmp_acc = m.compute()
                acc[s] = tmp_acc
                m.reset()
        return acc


class VAPModule(L.LightningModule):
    def __init__(
        self,
        model: VAP,
        opt_lr: float = 3.63e-4,
        opt_betas: tuple[float, float] = (0.9, 0.999),
        opt_weight_decay: float = 0.001,
    ):
        super().__init__()
        self.model = model
        self.opt = {
            "lr": opt_lr,
            "betas": opt_betas,
            "weight_decay": opt_weight_decay,
        }
        self.save_hyperparameters(ignore=["model"])
        # self.train_metrics = VapMetrics()
        self.metric_val = VapMetrics()
        self.metric_test = VapMetrics()

    def forward(self, waveform: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        return self.model(waveform, *args, **kwargs)

    @staticmethod
    def load_model(path: str, *args, **kwargs) -> VAP:
        return VAPModule.load_from_checkpoint(path, *args, **kwargs).model

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opt["lr"],
            betas=self.opt["betas"],
            weight_decay=self.opt["weight_decay"],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.5,
                    patience=5,
                ),
                # "monitor": "loss_vap_val",
                "monitor": "loss_vap_train",
            },
        }

    def metric_update(self, logits, vad, split: str = "val"):
        m = getattr(self, f"{split}_metric", None)
        if m:
            probs = self.model.objective.get_probs(logits)
            m.update_batch(probs, vad)

    def metric_finalize(self, split: str = "val") -> None:
        m = getattr(self, f"metric_{split}", None)
        if m:
            scores = m.epoch_update()
            self.log(
                f"{split}_acc_hold_now", scores["hold_now_accuracy"], sync_dist=True
            )
            self.log(
                f"{split}_acc_hold_fut", scores["hold_fut_accuracy"], sync_dist=True
            )
            self.log(
                f"{split}_acc_hold_both", scores["hold_both_accuracy"], sync_dist=True
            )
            self.log(
                f"{split}_acc_shift_now", scores["shift_now_accuracy"], sync_dist=True
            )
            self.log(
                f"{split}_acc_shift_fut", scores["shift_fut_accuracy"], sync_dist=True
            )
            self.log(
                f"{split}_acc_shift_both", scores["shift_both_accuracy"], sync_dist=True
            )

    def _step(self, batch: Batch, split: str = "train") -> Mapping[str, torch.Tensor]:
        """
        Arguments:
            batch:      dict, keys(['waveform', 'vad'])

        Returns:
            out:        dict, ['logits', 'vad', 'vap_loss', 'vad_loss']
        """

        # Forward
        out = self(batch["waveform"])
        labels = self.model.extract_labels(batch["vad"])

        # Losses
        out["vap_loss"] = self.model.objective.loss_vap(
            out["logits"], labels, reduction="mean"
        )
        out["vad_loss"] = self.model.objective.loss_vad(out["vad"], batch["vad"])

        # Log results
        batch_size = batch["waveform"].shape[0]
        self.log(
            f"loss_vap_{split}",
            out["vap_loss"],
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            f"loss_vad_{split}", out["vad_loss"], batch_size=batch_size, sync_dist=True
        )
        if split in ["val", "test"]:
            with torch.inference_mode():
                p = self.model.prediction(out["logits"])
                out.update(p)
            metric = getattr(self, f"metric_{split}")
            metric.batch_update(out["p_now"], out["p_fut"], batch["vad"])
        return out

    def training_step(self, batch: Batch, *args, **kwargs):
        out = self._step(batch)
        loss = out["vap_loss"] + out["vad_loss"]
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
        print("TEST END")
        self.metric_finalize(split="test")


def test_forward():
    from vap.modules.encoder import EncoderCPC
    from vap.data.datamodule import VAPDataModule
    from tqdm import tqdm
    import time

    encoder = EncoderCPC()
    transformer = VapStereoTower()
    model = VAP(encoder, transformer)
    module = VAPModule(model)
    dm = VAPDataModule(
        train_path="data/splits/fisher_swb/train_sliding.csv",
        val_path="data/splits/fisher_swb/val_sliding.csv",
        num_workers=0,
        prefetch_factor=None,
    )
    dm.setup()
    dloader = dm.train_dataloader()
    batch = next(iter(dloader))

    def time_model(model, batch):
        batch["waveform"] = batch["waveform"].to(model.device)
        batch["vad"] = batch["vad"].to(model.device)
        t = time.time()
        for _ in range(20):
            out = model(batch["waveform"])
        t = time.time() - t
        return t

    model = model.to("cuda")
    t = time_model(model, batch)
    print(f"Forward time: {t:.3f} s")

    model = torch.compile(model)
    t = time_model(model, batch)
    print(f"Forward time: {t:.3f} s")


if __name__ == "__main__":
    from vap.modules.encoder import EncoderCPC
    from vap.data.datamodule import VAPDataModule
    from tqdm import tqdm

    dm = VAPDataModule(
        train_path="data/splits/fisher_swb/train_sliding.csv",
        val_path="data/splits/fisher_swb/val_sliding.csv",
        num_workers=0,
        prefetch_factor=None,
    )
    dm.setup()
    # dm.setup()
    # # module = module.to("cuda")
    # batch = next(iter(dm.train_dataloader()))
    # module = VAPModule(model)
    # for batch in tqdm(dm.train_dataloader()):
    #     module._step(batch, split="val")

    encoder = EncoderCPC()
    transformer = VapStereoTower(
        dim=256, num_heads=4, num_self_attn_layers=1, num_cross_attn_layers=4
    )
    model = VAP(encoder, transformer)
    module = VAPModule(model, opt_lr=2e-5)
    trainer = L.Trainer(
        limit_train_batches=10,
        limit_val_batches=10,
        logger=L.pytorch.loggers.CSVLogger("logs/"),
        # gradient_clip_val=5.0,
        max_epochs=20,
    )
    trainer.fit(
        module,
        train_dataloaders=dm.train_dataloader(),
    )

    def plot_csv_metrics(path):
        import pandas as pd
        import matplotlib.pyplot as plt

        path = "logs/lightning_logs/version_11/metrics.csv"
        df = pd.read_csv(path)
        # loss_vap_val,
        # loss_vad_val,
        # val_acc_hold_now,
        # val_acc_hold_fut,
        # val_acc_hold_both,
        # val_acc_shift_now,
        # val_acc_shift_fut,
        # val_acc_shift_both,
        # epoch,
        # step,
        # loss_vap_train,
        # loss_vad_train
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        for col in df.columns:
            if "acc" in col:
                c = "g"
                if "hold" in col:
                    c = "r"
                ax[0].plot(df[col], label=col, color=c)
            if "loss" in col:
                c = "b"
                if "vap" in col:
                    c = "k"
                ax[1].plot(df[col], label=col, color=c)
        for a in ax:
            a.legend()
        plt.show()
