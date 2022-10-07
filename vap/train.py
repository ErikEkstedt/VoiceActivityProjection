from omegaconf import DictConfig, OmegaConf
from os import environ
import hydra
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from vap.callbacks import WandbArtifactCallback
from vap.model import VAPModel
from vap.utils import everything_deterministic
from datasets_turntaking import DialogAudioDM


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    if "debug" in cfg_dict:
        environ["WANDB_MODE"] = "offline"
        print("DEBUG -> OFFLINE MODE")

    pl.seed_everything(cfg_dict["seed"])
    local_rank = environ.get("LOCAL_RANK", 0)

    if cfg_dict["verbose"]:
        print("Model Conf")
        for k, v in cfg_dict["model"].items():
            print(f"{k}: {v}")
        print("#" * 60)

    model = VAPModel(cfg_dict)

    if cfg_dict["verbose"]:
        print("DataModule")
        for k, v in cfg_dict["data"].items():
            print(f"{k}: {v}")
        print("#" * 60)

    dm = DialogAudioDM(audio_mono=not model.stereo, **cfg_dict["data"])
    dm.prepare_data()

    if cfg_dict["trainer"]["fast_dev_run"]:
        trainer = pl.Trainer(**cfg_dict["trainer"])
        print("NAME: " + model.run_name)
        print(cfg_dict["model"])
        print("-" * 40)
        print(dm)
        trainer.fit(model, datamodule=dm)
    else:
        # Callbacks & Logger
        logger = WandbLogger(
            # save_dir=SA,
            project=cfg_dict["wandb"]["project"],
            name=model.run_name,
            log_model=False,
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
            WandbArtifactCallback(),
        ]

        if cfg_dict["optimizer"].get("swa_enable", False):
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=cfg_dict["optimizer"].get("swa_lrs", 0.05),
                    swa_epoch_start=cfg_dict["optimizer"].get("swa_epoch_start", 5),
                    annealing_epochs=cfg_dict["optimizer"].get(
                        "swa_annealing_epochs", 10
                    ),
                )
            )

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
