import pytest
import torch
from vap.utils import load_hydra_conf
from vap.model import VAPModel


@pytest.fixture
def batch():
    data = torch.load("assets/vap_data.pt")
    if torch.cuda.is_available():
        data["shift"]["vad"] = data["shift"]["vad"].to("cuda")
        data["bc"]["vad"] = data["bc"]["vad"].to("cuda")
        data["only_hold"]["vad"] = data["only_hold"]["vad"].to("cuda")

        data["shift"]["waveform"] = data["shift"]["waveform"].to("cuda")
        data["bc"]["waveform"] = data["bc"]["waveform"].to("cuda")
        data["only_hold"]["waveform"] = data["only_hold"]["waveform"].to("cuda")

    batch = {
        "waveform": torch.cat(
            [
                data["shift"]["waveform"],
                data["bc"]["waveform"],
                data["only_hold"]["waveform"],
            ]
        ),
        "vad": torch.cat(
            [
                data["shift"]["vad"],
                data["bc"]["vad"],
                data["only_hold"]["vad"],
            ]
        ),
    }
    return batch


@pytest.mark.model
@pytest.mark.parametrize(
    ["dim", "channel_layers", "cross_layers", "num_heads"],
    [
        [256, 1, 1, 4],
        [128, 1, 1, 4],
        [64, 1, 1, 4],
        [128, 3, 1, 8],
        [128, 1, 3, 8],
    ],
)
def test_model(batch, dim, channel_layers, cross_layers, num_heads):
    conf = load_hydra_conf()
    conf["model"] = load_hydra_conf(config_name="model/vap_50hz_stereo")["model"]
    conf["model"]["ar"]["dim"] = dim
    conf["model"]["ar"]["channel_layers"] = channel_layers
    conf["model"]["ar"]["num_layers"] = cross_layers
    conf["model"]["ar"]["num_heads"] = num_heads
    model = VAPModel(conf)
    if torch.cuda.is_available():
        model = model.to("cuda")
    opt = model.configure_optimizers()["optimizer"]
    # Forward
    out = model.shared_step(batch)
    # Backward
    out["loss"].backward()
    # Update
    opt.step()
    opt.zero_grad()
    # Forward
    out = model.shared_step(batch)
    # Backward
    out["loss"].backward()
    # Update
    opt.step()
    opt.zero_grad()


@pytest.mark.model
@pytest.mark.parametrize(
    ["dim", "num_layers", "num_heads"],
    [
        [256, 1, 4],
        [256, 1, 8],
        [128, 1, 4],
        [64, 1, 4],
        [128, 2, 4],
        [128, 3, 4],
    ],
)
def test_model_mono(batch, dim, num_layers, num_heads):
    conf = load_hydra_conf()
    conf["model"] = load_hydra_conf(config_name="model/vap_50hz")["model"]
    conf["model"]["ar"]["dim"] = dim
    conf["model"]["ar"]["num_layers"] = num_layers
    conf["model"]["ar"]["num_heads"] = num_heads

    batch["waveform"] = batch["waveform"].mean(dim=1).unsqueeze(1)

    model = VAPModel(conf)
    if torch.cuda.is_available():
        model = model.to("cuda")
    opt = model.configure_optimizers()["optimizer"]

    # Forward
    out = model.shared_step(batch)
    # Backward
    out["loss"].backward()
    # Update
    opt.step()
    opt.zero_grad()

    # Forward
    out = model.shared_step(batch)
    # Backward
    out["loss"].backward()
    # Update
    opt.step()
    opt.zero_grad()


@pytest.mark.model
@pytest.mark.parametrize(
    "config_name",
    [
        "model/vap_50hz",
        "model/vap_50hz_stereo",
    ],
)
def test_model_directly_from_conf(config_name, batch):
    conf = load_hydra_conf()
    conf["model"] = load_hydra_conf(config_name=config_name)["model"]
    model = VAPModel(conf)
    if torch.cuda.is_available():
        model = model.to("cuda")

    if not model.stereo:
        batch["waveform"] = batch["waveform"].mean(dim=1).unsqueeze(1)

    opt = model.configure_optimizers()["optimizer"]
    out = model.shared_step(batch)
    out["loss"].backward()
    opt.step()
