import torch
import logging
import hydra
import os
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from lightning.pytorch import Trainer

log: logging.Logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision(precision="medium")
# Inspired by:
# https://github.com/facebookresearch/recipes/blob/main/torchrecipes/audio/source_separation/main.py
@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", 0)
    seed_everything(seed, workers=True)
    log.info(OmegaConf.to_yaml(cfg))

    module = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)

    if getattr(cfg, "pretrained_checkpoint_path", None):
        module = module.load_from_checkpoint(
            checkpoint_path=cfg.pretrained_checkpoint_path
        )
        print("Loaded from checkpoint: ", cfg.pretrained_checkpoint_path)
        if getattr(cfg.module, "val_metric", False):
            metric = instantiate(cfg.module.val_metric)
            module.val_metric = metric
            print("Added val metrics")
        input("Press enter to continue: ")

    if getattr(cfg, "debug", False):
        trainer = Trainer(fast_dev_run=True)
    else:
        trainer = instantiate(cfg.trainer)

    print("CPUs: ", os.cpu_count())
    print("Pytorch Threads: ", torch.get_num_threads())
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
