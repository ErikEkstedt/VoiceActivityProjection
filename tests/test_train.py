import pytest

import pytorch_lightning as pl
from vap.utils import load_hydra_conf
from vap.model import VAPModel
from datasets_turntaking import DialogAudioDM


@pytest.mark.train
@pytest.mark.parametrize(
    "config_name",
    [
        "model/vap_50hz",
        "model/vap_50hz_stereo",
    ],
)
def test_cpc_train(config_name):
    conf = load_hydra_conf()
    conf["model"] = load_hydra_conf(config_name=config_name)["model"]
    model = VAPModel(conf)

    conf["data"]["num_workers"] = 0
    conf["data"]["batch_size"] = 2
    conf["data"]["audio_duration"] = 10
    dm = DialogAudioDM(audio_mono=not model.stereo, **conf["data"])
    dm.prepare_data()

    cfg_dict = dict(conf)
    cfg_dict["trainer"]["fast_dev_run"] = True
    trainer = pl.Trainer(**cfg_dict["trainer"])
    trainer.fit(model, datamodule=dm)
