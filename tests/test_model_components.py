import pytest

import torch
from vap.model import ProjectionModel, VAPHead, VAPModel, VACondition
from vap.utils import load_hydra_conf

B = 4
N = 500


@pytest.mark.model_components
@pytest.mark.parametrize(
    ["input_dim", "n_bins", "representation"],
    [
        (256, 4, "discrete"),
        (256, 4, "independent"),
        (256, 4, "comparative"),
    ],
)
def test_vap_head(input_dim, n_bins, representation):
    head = VAPHead(input_dim, n_bins, representation)

    x = torch.rand(B, N, input_dim)
    logits = head(x)

    output_shape = (B, N, head.output_dim)
    if representation == "comparative":
        output_shape = (B, N, 1)
    elif representation == "independent":
        output_shape = (B, N, *head.output_dim)

    assert logits.shape == output_shape, f"Wrong shape VAPHEAD {representation}"


@pytest.mark.model_components
@pytest.mark.parametrize(
    ["dim", "va_history", "va_history_bins"],
    [
        (256, True, 5),
        (256, False, 5),
    ],
)
def test_va_condition(dim, va_history, va_history_bins):
    va_cond = VACondition(dim, va_history, va_history_bins)
    va = torch.randint(0, 2, (B, N, 2)).float()
    output_shape = (B, N, dim)
    vah = None
    if va_history:
        vah = torch.rand((B, N, va_history_bins))

    z = va_cond(va, vah)
    assert z.shape == output_shape, f"VA cond wrong shape. {z.shape}!={output_shape}"


@pytest.mark.model_components
@pytest.mark.stereo
@pytest.mark.parametrize(
    "config_name",
    [
        "model/vap_50hz",
        "model/vap_50hz_stereo",
    ],
)
def test_projection_model(config_name):
    conf = load_hydra_conf(config_name=config_name)["model"]
    net = ProjectionModel(conf)

    waveform = torch.randn((B, 1, net.sample_rate))
    if net.stereo:
        waveform = torch.cat((waveform, torch.randn((B, 1, net.sample_rate))), dim=1)

    out_frames = int(net.sample_rate / net.encoder.downsample_ratio)

    va, vah = None, None
    if not net.stereo:
        va = torch.randint(0, 1, (B, out_frames, 2)).float()
        if conf["va_cond"]["history"]:
            vah = torch.rand((B, out_frames, conf["va_cond"]["history_bins"]))

    logits = net(waveform=waveform, va=va, va_history=vah)

    output_shape = (B, out_frames, net.vap_head.output_dim)
    if net.vap_head.representation == "comparative":
        output_shape = (B, out_frames, 1)
    elif net.vap_head.representation == "independent":
        output_shape = (B, out_frames, *net.vap_head.output_dim)

    assert (
        logits.shape == output_shape
    ), f"Output is wrong shape, {logits.shape} != {output_shape}"
