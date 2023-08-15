import pytest
import torch

from vap.modules.VAP import VAP, VAPMono
from vap.modules.encoder import EncoderCPC
from vap.modules.modules import TransformerStereo, GPT
from vap.modules.lightning_module import VAPModule, VAPMonoModule

BATCH_SIZE = 4
DURATION = 5
SAMPLE_RATE = 16_000
FRAME_HZ = 50
N_SAMPLES = int(DURATION * SAMPLE_RATE)


@pytest.mark.modules
@pytest.mark.lightning
def test_vap():
    model = VAP(EncoderCPC(), TransformerStereo())
    module = VAPModule(model)
    x = torch.randn(BATCH_SIZE, 2, N_SAMPLES)
    out = module(x)
    batch = {
        "waveform": x,
        "vad": torch.randint(0, 2, (BATCH_SIZE, int(5 * FRAME_HZ), 2)).float(),
    }
    out = module._step(batch, "train")


@pytest.mark.mono
@pytest.mark.modules
@pytest.mark.lightning
def test_vap_mono():
    model = VAPMono(EncoderCPC(), GPT())
    module = VAPMonoModule(model)

    x = torch.randn(BATCH_SIZE, 1, N_SAMPLES)

    vad = torch.randint(0, 2, (BATCH_SIZE, int(DURATION * FRAME_HZ), 2)).float()
    out = module(x, vad)

    frames_w_horizon = int((DURATION + model.horizon_time) * FRAME_HZ)
    batch = {
        "waveform": x,
        "vad": torch.randint(0, 2, (BATCH_SIZE, frames_w_horizon, 2)).float(),
    }
    out = module._step(batch, "train")
