from argparse import ArgumentParser
from os import environ
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


from vap.lightning_module import VAPMonoModel
from vap.model import VapMonoConfig
from vap.events import EventConfig

from vap_dataset.datamodule import VapDataModule
from vap.phrases.dataset import PhrasesCallback

# from vap.callbacks import SymmetricSpeakersCallback, AudioAugmentationCallback


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
    mono: bool = True

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
    parser, fields_added = VapMonoConfig.add_argparse_args(parser)
    parser, fields_added = EventConfig.add_argparse_args(parser, fields_added)
    args = parser.parse_args()

    model_conf = VapMonoConfig.args_to_conf(args)
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
    s = "VapGPTMono"
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

    model = VAPMonoModel(
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
        mono=True,
    )
    dm.prepare_data()
    print(dm)

    if cfg_dict["debug"]:
        environ["WANDB_MODE"] = "offline"
        print("DEBUG -> OFFLINE MODE")

    if cfg_dict["fast_dev_run"]:
        print("NAME: " + name)
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

        for n in ["logger", "strategy", "debug", "seed", "wandb_project"]:
            cfg_dict.pop(n)
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=False),
            **cfg_dict,
        )
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
