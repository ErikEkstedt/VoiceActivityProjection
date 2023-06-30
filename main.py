import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

log: logging.Logger = logging.getLogger(__name__)

# https://github.com/facebookresearch/recipes/blob/main/torchrecipes/audio/source_separation/main.py


@hydra.main(config_path="conf", config_name="default_config")
def main(config: DictConfig) -> None:
    seed = config.get("seed", 0)
    seed_everything(seed, workers=True)
    log.info(OmegaConf.to_yaml(config))

    model = hydra.utils.instantiate(config.module)
    datamodule = hydra.utils.instantiate(config.datamodule)
    trainer = hydra.utils.instantiate(config.trainer)
    # print(datamodule)
    # print(trainer)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
