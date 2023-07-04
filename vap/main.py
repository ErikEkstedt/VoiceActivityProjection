import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything

log: logging.Logger = logging.getLogger(__name__)

# https://github.com/facebookresearch/recipes/blob/main/torchrecipes/audio/source_separation/main.py


@hydra.main(version_base=None, config_path="conf", config_name="default_config")
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", 0)
    seed_everything(seed, workers=True)
    log.info(OmegaConf.to_yaml(cfg))

    module = hydra.utils.instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
