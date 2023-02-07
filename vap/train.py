# from omegaconf import DictConfig, OmegaConf
# import hydra
from argparse import ArgumentParser
from os import environ
from typing import Dict
from dataclasses import dataclass

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
from torchmetrics.classification import Accuracy, F1Score

# from datasets_turntaking import DialogAudioDM
from vap_dataset.datamodule import VapDataModule
from vap.phrases.dataset import PhrasesCallback
from vap.events import TurnTakingEvents, EventConfig
from vap.zero_shot import ZeroShot
from vap.model import VapGPT, VapConfig  # , load_older_state_dict


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


@dataclass
class DataConfig:
    train_path: str = "../vap_dataset/data/sliding_train.csv"
    val_path: str = "../vap_dataset/data/sliding_val.csv"
    test_path: str = "../vap_dataset/data/sliding_test.csv"
    flip_channels: bool = True
    flip_probability: float = 0.5
    mask_vad: bool = True
    mask_vad_probability: float = 0.5
    batch_size: int = 16
    num_workers: int = 24

    # not used for datamodule
    audio_duration: float = 20

    @staticmethod
    def add_argparse_args(parser):
        for k, v in DataConfig.__dataclass_fields__.items():
            parser.add_argument(f"--data_{k}", type=v.type, default=v.default)
        return parser

    @staticmethod
    def args_to_conf(args):
        return DataConfig(
            **{
                k.replace("data_", ""): v
                for k, v in vars(args).items()
                if k.startswith("data_")
            }
        )


def get_args():
    parser = ArgumentParser("VoiceActivityProjection")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="VapGPT")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = OptConfig.add_argparse_args(parser)
    parser = DataConfig.add_argparse_args(parser)
    parser, fields_added = VapConfig.add_argparse_args(parser)
    parser, fields_added = EventConfig.add_argparse_args(parser, fields_added)
    args = parser.parse_args()

    model_conf = VapConfig.args_to_conf(args)
    opt_conf = OptConfig.args_to_conf(args)
    data_conf = DataConfig.args_to_conf(args)
    event_conf = EventConfig.args_to_conf(args)

    # Remove all non trainer args
    cfg_dict = vars(args)
    for k, _ in list(cfg_dict.items()):
        if (
            k.startswith("data_")
            or k.startswith("vap_")
            or k.startswith("opt_")
            or k.startswith("event_")
        ):
            cfg_dict.pop(k)

    return {
        "args": args,
        "cfg_dict": cfg_dict,
        "model": model_conf,
        "event": event_conf,
        "opt": opt_conf,
        "data": data_conf,
    }


def get_run_name(configs) -> str:
    s = "VapGPT"
    s += f"_{configs['model'].frame_hz}Hz"
    s += f"_ad{configs['data'].audio_duration}s"
    s += f"_{configs['model'].channel_layers}"
    s += str(configs["model"].cross_layers)
    s += str(configs["model"].num_heads)
    return s


def train() -> None:
    configs = get_args()
    cfg_dict = configs["cfg_dict"]

    pl.seed_everything(cfg_dict["seed"])
    local_rank = environ.get("LOCAL_RANK", 0)

    model = VAPModel(
        configs["model"], opt_conf=configs["opt"], event_conf=configs["event"]
    )

    name = get_run_name(configs)

    dconf = configs["data"]
    dm = VapDataModule(
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        horizon=2,
        batch_size=dconf.batch_size,
        num_workers=dconf.num_workers,
    )
    dm.prepare_data()
    print(dm)

    if cfg_dict["debug"]:
        environ["WANDB_MODE"] = "offline"
        print("DEBUG -> OFFLINE MODE")

    if cfg_dict["fast_dev_run"]:
        print("NAME: " + name)
        for n in ["logger", "strategy", "debug", "seed", "wandb_project"]:
            cfg_dict.pop(n)
        trainer = pl.Trainer(**cfg_dict)
        trainer.fit(model, datamodule=dm)
    else:
        oconf = configs["opt"]

        # Callbacks & Logger
        logger = None
        callbacks = [
            ModelCheckpoint(
                mode=oconf.mode,
                monitor=oconf.monitor,
                auto_insert_metric_name=False,
                filename=name + "-epoch{epoch}-val_{val_loss:.2f}",
            ),
            EarlyStopping(
                monitor=oconf.monitor,
                mode=oconf.mode,
                patience=oconf.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=False,
            ),
            PhrasesCallback(),
        ]

        if not cfg_dict["debug"]:
            logger = WandbLogger(
                project=cfg_dict["wandb_project"],
                name=name,
                log_model=False,
                save_dir="runs",
            )
            callbacks.append(LearningRateMonitor())

        if local_rank == 0:
            print("#" * 40)
            print(f"Early stopping (patience={oconf.patience})")
            print("#" * 40)

        # Find Best Learning Rate
        if oconf.find_learning_rate:
            trainer = pl.Trainer(accelerator="gpu", devices=-1)
            lr_finder = trainer.tuner.lr_find(model, dm)
            model.learning_rate = lr_finder.suggestion()
        print("Learning Rate: ", model.opt_conf.learning_rate)
        print("#" * 40)

        # Actual Training

        if torch.cuda.is_available():
            cfg_dict["accelerator"] = "gpu"

        for n in ["logger", "strategy", "debug", "seed", "wandb_project", "debug"]:
            cfg_dict.pop(n)
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=False),
            **cfg_dict,
        )
        trainer.fit(model, datamodule=dm)


# Used in training (LightningModule) but not required for inference
class VAPModel(VapGPT, pl.LightningModule):
    def __init__(self, conf, opt_conf=None, event_conf=None):
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

    def get_metrics(self):
        metrics = {"acc": {}, "f1": {}}

        metrics["acc"]["hs"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(self.device)
        metrics["acc"]["ls"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(self.device)
        metrics["acc"]["sp"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(self.device)
        metrics["acc"]["bp"] = Accuracy(
            task="multiclass", num_classes=2, multiclass=True, average="none"
        ).to(self.device)

        metrics["f1"]["hs"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(self.device)
        metrics["f1"]["ls"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(self.device)
        metrics["f1"]["sp"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(self.device)
        metrics["f1"]["bp"] = F1Score(
            task="multiclass",
            num_classes=2,
            multiclass=True,
            average="weighted",
        ).to(self.device)

        return metrics

    def metrics_step(self, preds, targets, split="val"):
        m = self.val_metrics if split == "val" else self.test_metrics

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

    def metrics_epoch(self, split="val"):
        if split == "val":
            m = self.val_metrics
        else:
            m = self.test_metrics

        f1 = {}
        for name, metric in m["f1"].items():
            f1[name] = metric.compute()
            metric.reset()

        # Accuracy
        acc = {}
        for name, metric in m["acc"].items():
            a, b = metric.compute()
            acc[name] = [a, b]
            metric.reset()

        self.log(
            f"{split}_hs",
            {"shift_acc": acc["hs"][1], "f1w": f1["hs"]},
            prog_bar=True,
            sync_dist=True,
        )
        self.log(f"{split}_pred_sh", {"shift": acc["sp"][1]}, sync_dist=True)
        self.log(f"{split}_ls", {"short": acc["ls"][1]}, sync_dist=True)
        self.log(f"{split}_pred_bc", {"bc_pred": acc["bp"][1]}, sync_dist=True)

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
        assert self.opt_conf is not None, "configure_optimizers: No Opt conf!"
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_conf.learning_rate,
            betas=self.opt_conf.betas,
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

    def training_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", out["vap_loss"], batch_size=batch_size, sync_dist=True)
        self.log("loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)
        loss = out["vap_loss"] + out["vad_loss"]
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        if not hasattr(self, "val_metrics"):
            self.val_metrics = self.get_metrics()

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
            self.metrics_step(preds, targets, split="val")

    def validation_epoch_end(self, *_):
        if hasattr(self, "val_metrics"):
            self.metrics_epoch("val")

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
            self.metrics_step(preds, targets, split="test")

    def test_epoch_end(self, *_):
        if hasattr(self, "test_metrics"):
            self.metrics_epoch("test")


if __name__ == "__main__":
    train()
