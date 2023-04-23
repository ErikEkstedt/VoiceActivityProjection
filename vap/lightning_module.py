from dataclasses import dataclass
from typing import Optional

import torch
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score

from vap.zero_shot import ZeroShot
from vap.events import TurnTakingEvents, EventConfig
from vap.model import VapGPT, VapConfig, VapGPTMono, VapMonoConfig


@dataclass
class OptConfig:
    learning_rate: float = 3.63e-4
    find_learning_rate: bool = False
    betas = [0.9, 0.999]
    weight_decay: float = 0.001
    lr_scheduler_interval: str = "step"
    lr_scheduler_freq: int = 100
    lr_scheduler_tmax: int = 2500
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5

    # early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor: str = "val_loss"
    mode: str = "min"

    @staticmethod
    def add_argparse_args(parser):
        for k, v in OptConfig.__dataclass_fields__.items():
            parser.add_argument(f"--opt_{k}", type=v.type, default=v.default)
        return parser

    @staticmethod
    def args_to_conf(args):
        return OptConfig(
            **{
                k.replace("opt_", ""): v
                for k, v in vars(args).items()
                if k.startswith("opt_")
            }
        )


def load_model(checkpoint: str, conf: Optional[VapConfig] = None) -> tuple[VapGPT, str]:
    if conf is None:
        conf = VapConfig()
    model = VapGPT(conf)
    std = torch.load(checkpoint)
    model.load_state_dict(std, strict=False)
    model.eval()
    device = "cpu"
    if torch.cuda.is_available():
        model = model.to("cuda")
        device = "cuda"
    return model, device


class VapMetrics:
    def get_metrics(self, device="cpu"):
        metrics = {"acc": {}, "f1": {}}

        metrics["acc"]["hs"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(device)
        metrics["acc"]["ls"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(device)
        metrics["acc"]["sp"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(device)
        metrics["acc"]["bp"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(device)

        metrics["f1"]["hs"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(device)
        metrics["f1"]["ls"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(device)
        metrics["f1"]["sp"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(device)
        metrics["f1"]["bp"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(device)

        return metrics

    def metrics_step(self, preds, targets, m):
        # The metrics don't work if the predictions are not rounded
        # I don't know why...
        if preds["hs"] is not None:
            m["f1"]["hs"].update(preds=preds["hs"].round(), target=targets["hs"])
            m["acc"]["hs"].update(preds=preds["hs"].round(), target=targets["hs"])

        if preds["ls"] is not None:
            m["f1"]["ls"].update(preds=preds["ls"].round(), target=targets["ls"])
            m["acc"]["ls"].update(preds=preds["ls"].round(), target=targets["ls"])

        if preds["pred_shift"] is not None:
            m["f1"]["sp"].update(
                preds=preds["pred_shift"].round(), target=targets["pred_shift"]
            )
            m["acc"]["sp"].update(
                preds=preds["pred_shift"].round(), target=targets["pred_shift"]
            )

        if preds["pred_backchannel"] is not None:
            m["f1"]["bp"].update(
                preds=preds["pred_backchannel"], target=targets["pred_backchannel"]
            )
            m["acc"]["bp"].update(
                preds=preds["pred_backchannel"], target=targets["pred_backchannel"]
            )
        return m

    def calc_metrics_epoch(self, m):
        f1 = {}
        for name, metric in m["f1"].items():
            f1[name] = metric.compute()
            metric.reset()

        acc = {}
        for name, metric in m["acc"].items():
            a, b = metric.compute()
            acc[name] = [a, b]
            metric.reset()
        return m, f1, acc


class VAPModel(VapGPT, pl.LightningModule, VapMetrics):
    def __init__(
        self,
        conf: VapConfig,
        opt_conf: Optional[OptConfig] = None,
        event_conf: Optional[EventConfig] = None,
    ):
        super().__init__(conf)

        self.opt_conf = opt_conf
        self.event_conf = event_conf

        # Training params
        self.save_hyperparameters()

        # Metrics
        self.event_extractor = None
        if event_conf is not None:
            self.zero_shot = ZeroShot(bin_times=conf.bin_times, frame_hz=conf.frame_hz)
            self.event_extractor = TurnTakingEvents(event_conf)

    def shared_step(
        self, batch: dict, reduction: str = "mean"
    ) -> dict[str, torch.Tensor]:
        """
        Arguments:
            batch:      dict, containing 'waveform', va, va_history

        Returns:
            out:        dict, ['logits', 'vad', 'vap_loss', 'vad_loss']
        """
        labels = self.extract_labels(batch["vad"])
        out = self(waveform=batch["waveform"])
        out["vap_loss"] = self.objective.loss_vap(
            out["logits"], labels, reduction=reduction
        )
        out["vad_loss"] = self.objective.loss_vad(out["vad"], batch["vad"])
        return out

    def configure_optimizers(self) -> dict:
        assert self.opt_conf is not None, "configure_optimizers: No Opt conf!"
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_conf.learning_rate,
            betas=tuple(self.opt_conf.betas),
            weight_decay=self.opt_conf.weight_decay,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=self.opt_conf.lr_scheduler_factor,
                patience=self.opt_conf.lr_scheduler_patience,
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, **_):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)
        loss = out["vap_loss"] + out["vad_loss"]
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **_):
        """validation step"""
        if not hasattr(self, "val_metrics"):
            self.val_metrics = self.get_metrics(self.device)

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("val_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("val_loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["vad"])
            # probs = self.zero_shot.get_probs(out["logits"], batch["vad"])
            # preds, targets = self.zero_shot.extract_prediction_and_targets(
            #     p=probs["p"], p_bc=probs["p_bc"], events=events
            # )
            probs = self.objective.get_probs(out["logits"])
            preds, targets = self.objective.extract_prediction_and_targets(
                p_now=probs["p_now"], p_fut=probs["p_future"], events=events
            )
            self.val_metrics = self.metrics_step(preds, targets, self.val_metrics)

    def validation_epoch_end(self, *_):
        if hasattr(self, "val_metrics"):
            # self.metrics_epoch("val")
            self.val_metrics, f1, acc = self.calc_metrics_epoch(self.val_metrics)
            self.log(
                "val_hs",
                {"shift_acc": acc["hs"][1], "f1w": f1["hs"]},
                prog_bar=True,
                sync_dist=True,
            )
            self.log("val_pred_sh", {"shift": acc["sp"][1]}, sync_dist=True)
            self.log("val_ls", {"short": acc["ls"][1]}, sync_dist=True)
            self.log("val_pred_bc", {"bc_pred": acc["bp"][1]}, sync_dist=True)

    def test_step(self, batch, **_):
        """validation step"""
        if not hasattr(self, "test_metrics"):
            self.test_metrics = self.get_metrics(self.device)
            for name, events in self.test_metrics.items():
                for event, metric in events.items():
                    strname = f"test_{name}_{event}"
                    self.register_module(strname, metric)

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("test_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("test_loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["vad"])
            # probs = self.zero_shot.get_probs(out["logits"], batch["vad"])
            # preds, targets = self.zero_shot.extract_prediction_and_targets(
            #     p=probs["p"], p_bc=probs["p_bc"], events=events
            # )
            probs = self.objective.get_probs(out["logits"])
            preds, targets = self.objective.extract_prediction_and_targets(
                p_now=probs["p_now"], p_fut=probs["p_future"], events=events
            )
            self.test_metrics = self.metrics_step(preds, targets, self.test_metrics)

    def test_epoch_end(self, *_):
        if hasattr(self, "test_metrics"):
            # self.metrics_epoch("test")
            self.test_metrics, f1, acc = self.calc_metrics_epoch(self.test_metrics)
            self.log(
                "test_hs",
                {"shift_acc": acc["hs"][1], "f1w": f1["hs"]},
                prog_bar=True,
                sync_dist=True,
            )
            self.log("test_pred_sh", {"shift": acc["sp"][1]}, sync_dist=True)
            self.log("test_ls", {"short": acc["ls"][1]}, sync_dist=True)
            self.log("test_pred_bc", {"bc_pred": acc["bp"][1]}, sync_dist=True)


class VAPMonoModel(VapGPTMono, pl.LightningModule, VapMetrics):
    def __init__(
        self,
        conf: VapMonoConfig,
        opt_conf: Optional[OptConfig] = None,
        event_conf: Optional[EventConfig] = None,
    ):
        super().__init__(conf)

        self.opt_conf = opt_conf
        self.event_conf = event_conf

        # Training params
        self.save_hyperparameters()

        # Metrics
        self.event_extractor = None
        if event_conf is not None:
            self.zero_shot = ZeroShot(bin_times=conf.bin_times, frame_hz=conf.frame_hz)
            self.event_extractor = TurnTakingEvents(event_conf)

    def configure_optimizers(self) -> dict:
        assert self.opt_conf is not None, "configure_optimizers: No Opt conf!"
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_conf.learning_rate,
            betas=tuple(self.opt_conf.betas),
            weight_decay=self.opt_conf.weight_decay,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=self.opt_conf.lr_scheduler_factor,
                patience=self.opt_conf.lr_scheduler_patience,
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def validation_epoch_end(self, *_):
        if hasattr(self, "val_metrics"):
            self.val_metrics, f1, acc = self.calc_metrics_epoch(self.val_metrics)
            self.log(
                "val_hs",
                {"shift_acc": acc["hs"][1], "f1w": f1["hs"]},
                prog_bar=True,
                sync_dist=True,
            )
            self.log("val_pred_sh", {"shift": acc["sp"][1]}, sync_dist=True)
            self.log("val_ls", {"short": acc["ls"][1]}, sync_dist=True)
            self.log("val_pred_bc", {"bc_pred": acc["bp"][1]}, sync_dist=True)

    def test_epoch_end(self, *_):
        if hasattr(self, "test_metrics"):
            # self.metrics_epoch("test")
            self.test_metrics, f1, acc = self.calc_metrics_epoch(self.test_metrics)
            self.log(
                "test_hs",
                {"shift_acc": acc["hs"][1], "f1w": f1["hs"]},
                prog_bar=True,
                sync_dist=True,
            )
            self.log("test_pred_sh", {"shift": acc["sp"][1]}, sync_dist=True)
            self.log("test_ls", {"short": acc["ls"][1]}, sync_dist=True)
            self.log("test_pred_bc", {"bc_pred": acc["bp"][1]}, sync_dist=True)

    def shared_step(
        self, batch: dict, reduction: str = "mean"
    ) -> dict[str, torch.Tensor]:
        """
        Arguments:
            batch:      dict, containing 'waveform', va, va_history

        Returns:
            out:        dict, ['logits', 'vad', 'vap_loss', 'vad_loss']
        """
        labels = self.objective.get_labels(batch["vad"])
        out = self(waveform=batch["waveform"])
        out["vap_loss"] = self.objective.loss_vap(
            out["logits"], labels, reduction=reduction
        )
        return out

    def training_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        return {"loss": out["vap_loss"]}

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        if not hasattr(self, "val_metrics"):
            self.val_metrics = self.get_metrics()

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]

        self.log("val_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["vad"])
            probs = self.objective.get_probs(out["logits"])
            preds, targets = self.objective.extract_prediction_and_targets(
                p_now=probs["p_now"], p_fut=probs["p_future"], events=events
            )
            self.val_metrics = self.metrics_step(preds, targets, self.val_metrics)

    def test_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        if not hasattr(self, "test_metrics"):
            self.test_metrics = self.get_metrics()

            for name, events in self.test_metrics.items():
                for event, metric in events.items():
                    strname = f"test_{name}_{event}"
                    self.register_module(strname, metric)

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("test_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["vad"])
            probs = self.objective.get_probs(out["logits"])
            preds, targets = self.objective.extract_prediction_and_targets(
                p_now=probs["p_now"], p_fut=probs["p_future"], events=events
            )
            self.test_metrics = self.metrics_step(preds, targets, self.test_metrics)


if __name__ == "__main__":

    conf = VapConfig()
    opt_conf = OptConfig()
    model = VAPModel(conf, opt_conf=opt_conf)
    model.load_state_dict(
        torch.load("example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt")
    )
