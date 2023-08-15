import pytest
import torch

from vap.modules.VAP import VAP, VAPMono
from vap.modules.encoder import EncoderCPC
from vap.modules.modules import TransformerStereo, GPT


SAMPLE_RATE = 16_000
FRAME_HZ = 50


@pytest.mark.modules
def test_vap():
    model = VAP(EncoderCPC(), TransformerStereo())
    x = torch.randn(4, 2, int(5 * SAMPLE_RATE))
    out = model(x)


@pytest.mark.modules
@pytest.mark.mono
def test_vap_mono():
    model = VAPMono(EncoderCPC(), GPT())

    x = torch.randn(4, 1, int(5 * SAMPLE_RATE))
    vad = torch.randint(0, 2, (4, int(5 * FRAME_HZ), 2)).float()
    out = model(x, vad)
