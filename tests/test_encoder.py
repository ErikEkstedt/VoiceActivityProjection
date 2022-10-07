import pytest
import torch

from vap.encoder import Encoder


B = 4
SR = 16000
downsample50hz = {
    "kernel": [5],
    "stride": [2],
    "dilation": [1],
    "activation": "GELU",
}

downsample20hz = {
    "kernel": [11],
    "stride": [5],
    "dilation": [1],
    "activation": "GELU",
}


@pytest.mark.encoder
@pytest.mark.parametrize(
    ["freeze", "downsample"],
    [
        (True, None),
        (True, downsample50hz),
        (True, downsample20hz),
    ],
)
def test_encoder(freeze, downsample):
    encoder = Encoder(freeze=freeze, downsample=downsample)
    x = torch.randn((B, 1, SR))
    z = encoder(x)
    out_frames = int(SR / encoder.downsample_ratio)
    output_shape = (B, out_frames, encoder.dim)
    assert z.shape == output_shape
