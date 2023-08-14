import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything

from vap.modules.lightning_module import VAPModule

log: logging.Logger = logging.getLogger(__name__)


# Inspired by:
# https://github.com/facebookresearch/recipes/blob/main/torchrecipes/audio/source_separation/main.py
@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", 0)
    seed_everything(seed, workers=True)
    log.info(OmegaConf.to_yaml(cfg))

    module = VAPModule.load_from_checkpoint(cfg.checkpoint)

    # When loading from the checkpoint the metrics provided by
    # the config (recursive initialization) are not loaded
    # Most likely because no test_metric was provided during training
    test_metric = instantiate(cfg.module.test_metric)
    module.test_metric = test_metric

    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    res = trainer.test(module, datamodule=datamodule)
    print(res)


if __name__ == "__main__":
    main()
