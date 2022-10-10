import pytest

import torch
import pytorch_lightning as pl
from vap.utils import load_hydra_conf
from vap.model import VAPModel
from datasets_turntaking import DialogAudioDM


@pytest.mark.model
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
    opt = model.configure_optimizers()["optimizer"]

    conf["data"]["num_workers"] = 0
    conf["data"]["batch_size"] = 2
    conf["data"]["audio_duration"] = 10
    dm = DialogAudioDM(audio_mono=not model.stereo, **conf["data"])
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.val_dataloader()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    loss, out, batch = model.shared_step(batch)
    loss["loss"].backward()
    opt.step()
