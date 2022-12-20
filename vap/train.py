from omegaconf import DictConfig, OmegaConf
from os import environ
import hydra
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from datasets_turntaking import DialogAudioDM

from vap.callbacks import PhrasesCallback
from vap.events import TurnTakingEvents
from vap.model import VapGPT


# Used in training (LightningModule) but not required for inference
class VAPModel(VapGPT, pl.LightningModule):
    def __init__(self, conf, event_conf):
        super().__init__(conf)

        # Training params
        self.save_hyperparameters()

        # Metrics
        self.event_extractor = TurnTakingEvents(event_conf)
        self.val_metrics = self.get_metrics()

    def get_metrics(self):
        metrics = {"acc": {}, "f1": {}}
        metrics["f1"]["hs"] = F1Score(
            task="multiclass", num_classes=2, average="weighted", multiclass=True
        )
        metrics["f1"]["ls"] = F1Score(
            task="multiclass", num_classes=2, average="weighted", multiclass=True
        )
        metrics["f1"]["sp"] = F1Score(
            task="multiclass", num_classes=2, average="weighted", multiclass=True
        )
        metrics["f1"]["bp"] = F1Score(
            task="multiclass", num_classes=2, average="weighted", multiclass=True
        )
        metrics["acc"]["hs"] = Accuracy(
            task="multiclass", num_classes=2, average="none", multiclass=True
        )
        metrics["acc"]["ls"] = Accuracy(
            task="multiclass", num_classes=2, average="none", multiclass=True
        )
        metrics["acc"]["sp"] = Accuracy(
            task="multiclass", num_classes=2, average="none", multiclass=True
        )
        metrics["acc"]["bp"] = Accuracy(
            task="multiclass", num_classes=2, average="none", multiclass=True
        )
        return metrics

    def metrics_step(self, preds, targets, split="val"):
        m = self.val_metrics if split == "val" else self.test_metrics

        if preds["hs"] is not None:
            m["f1"]["hs"].update(preds=preds["hs"], target=targets["hs"])
            m["acc"]["hs"].update(preds=preds["hs"], target=targets["hs"])

        if preds["ls"] is not None:
            m["f1"]["ls"].update(preds=preds["ls"], target=targets["ls"])
            m["acc"]["ls"].update(preds=preds["ls"], target=targets["ls"])

        if preds["pred_shift"] is not None:
            m["f1"]["sp"].update(
                preds=preds["pred_shift"], target=targets["pred_shift"]
            )
            m["acc"]["sp"].update(
                preds=preds["pred_shift"], target=targets["pred_shift"]
            )

        if preds["pred_backchannel"] is not None:
            m["f1"]["bp"].update(
                preds=preds["pred_backchannel"], target=targets["pred_backchannel"]
            )
            m["acc"]["bp"].update(
                preds=preds["pred_backchannel"], target=targets["pred_backchannel"]
            )

    def metrics_epoch(self, split="val"):
        m = self.val_metrics if split == "val" else self.test_metrics

        f1 = {}
        for name, metric in m["f1"].items():
            f1[name] = metric.compute()
            metric.reset()

        # Accuracy
        acc = {}
        for name, metric in m["acc"].items():
            a, b = metric.compute()
            f1[name] = {"a": a, "b": b}
            metric.reset()

        # Log
        self.log(
            f"{split}_hs",
            {"hold_acc": acc["hs"][0], "shift_acc": acc["hs"][0], "f1": f1["hs"]},
            sync_dist=True,
        )
        self.log(
            f"{split}_ls",
            {"long": acc["ls"][0], "short": acc["ls"][1], "f1": f1["ls"]},
            sync_dist=True,
        )
        self.log(
            f"{split}_pred_sh",
            {"hold": acc["sp"][0], "shift": acc["sp"][1], "f1": f1["sp"]},
            sync_dist=True,
        )
        self.log(
            f"{split}_pred_bc",
            {"non": acc["bp"][0], "bc": acc["bp"][1], "f1": f1["bp"]},
            sync_dist=True,
        )

    @property
    def run_name(self):
        s = "VAP"
        s += f"_{self.conf.frame_hz}Hz"
        s += f"_ad{self.conf.audio_duration}s"
        s += f"_{self.conf.channel_layers}"
        s += str(self.conf.cross_layers)
        s += str(self.conf.num_heads)
        return s

    def shared_step(
        self, batch: Dict, reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
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
        out["vad_loss"] = self.objective.loss_vad(out["vad"], batch["vad"])
        return out

    def configure_optimizers(self) -> Dict:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.conf.opt_lr,
            betas=self.conf.opt_betas,
            weight_decay=self.conf.opt_weight_decay,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=self.conf.opt_lr_scheduler_factor,
                patience=self.conf.opt_lr_scheduler_patience,
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("loss_va", out["loss_va"], batch_size=batch_size, sync_dist=True)
        loss = out["vap_loss"] + out["vad_loss"]
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("val_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("val_loss_va", out["loss_va"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["vad"])
            preds, targets = self.VAP.extract_prediction_and_targets(
                p=out["p"], p_bc=out["p_bc"], events=events
            )
            self.metrics_step(preds, targets, split="val")

    def validation_epoch_end(self, *_):
        self.metrics_epoch("val")

    def test_step(self, batch, batch_idx, **kwargs):
        """Test step"""

        if not hasattr(self, "test_metrics"):
            self.text_metrics = self.get_metrics()

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("test_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("test_loss_va", out["loss_va"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["vad"])
            preds, targets = self.VAP.extract_prediction_and_targets(
                p=out["p"], p_bc=out["p_bc"], events=events
            )
            self.metrics_step(preds, targets, split="test")

    def test_epoch_end(self, *_):
        self.metrics_epoch("test")


def print_training(cfg_dict):
    if cfg_dict["verbose"]:
        print("Model Conf")
        for k, v in cfg_dict["model"].items():
            print(f"{k}: {v}")
        print("#" * 60)

    if cfg_dict["verbose"]:
        print("DataModule")
        for k, v in cfg_dict["data"].items():
            print(f"{k}: {v}")
        print("#" * 60)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    pl.seed_everything(cfg_dict["seed"])
    local_rank = environ.get("LOCAL_RANK", 0)

    model = VAPModel(cfg_dict)
    dm = DialogAudioDM(audio_mono=not model.stereo, **cfg_dict["data"])
    dm.prepare_data()

    if "debug" in cfg_dict:
        environ["WANDB_MODE"] = "offline"
        print("DEBUG -> OFFLINE MODE")

    if cfg_dict["trainer"]["fast_dev_run"]:
        if not torch.cuda.is_available():
            cfg_dict["trainer"].pop("accelerator")
            cfg_dict["trainer"].pop("devices")
        print("NAME: " + model.run_name)
        print("-" * 40)
        print(dm)
        print(cfg_dict["model"])
        trainer = pl.Trainer(**cfg_dict["trainer"])
        trainer.fit(model, datamodule=dm)
    else:
        # Callbacks & Logger
        logger = WandbLogger(
            project=cfg_dict["wandb"]["project"],
            name=model.run_name,
            log_model=False,
            save_dir="runs",
        )

        if local_rank == 0:
            print("#" * 40)
            print(f"Early stopping (patience={cfg_dict['early_stopping']['patience']})")
            print("#" * 40)

        callbacks = [
            ModelCheckpoint(
                mode=cfg_dict["checkpoint"]["mode"],
                monitor=cfg_dict["checkpoint"]["monitor"],
                auto_insert_metric_name=False,
                filename=model.run_name + "-epoch{epoch}-val_{val_loss:.2f}",
            ),
            EarlyStopping(
                monitor=cfg_dict["early_stopping"]["monitor"],
                mode=cfg_dict["early_stopping"]["mode"],
                patience=cfg_dict["early_stopping"]["patience"],
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=False,
            ),
            LearningRateMonitor(),
            PhrasesCallback(model),
        ]

        # Find Best Learning Rate
        if cfg_dict["optimizer"].get("find_learning_rate", False):
            trainer = pl.Trainer(accelerator="gpu", devices=-1)
            lr_finder = trainer.tuner.lr_find(model, dm)
            model.learning_rate = lr_finder.suggestion()
        print("Learning Rate: ", model.learning_rate)
        print("#" * 40)

        # Actual Training
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=False),
            **cfg_dict["trainer"],
        )
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
