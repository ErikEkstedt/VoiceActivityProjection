import pytest

import pytorch_lightning as pl
from vap.utils import load_hydra_conf
from vap.model import VAPModel
from datasets_turntaking import DialogAudioDM


@pytest.mark.train
@pytest.mark.data
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
    conf["data"]["batch_size"] = 20
    conf["data"]["audio_duration"] = 10

    for k, v in conf["data"].items():
        print(f"{k}: {v}")

    dm = DialogAudioDM(audio_mono=not model.stereo, **conf["data"])
    print(dm)
    dm.prepare_data()

    cfg_dict = dict(conf)
    cfg_dict["trainer"]["fast_dev_run"] = True
    trainer = pl.Trainer(**cfg_dict["trainer"], logger=None)
    trainer.fit(model, datamodule=dm)
