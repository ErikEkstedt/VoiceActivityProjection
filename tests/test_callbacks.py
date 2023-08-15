import pytest
import torch
from vap.callbacks.flip_channels import flip_batch_channels
from vap.callbacks.vad_mask import vad_mask_batch
from vap.utils.utils import vad_list_to_onehot

DURATION = 10
FRAME_HZ = 50
SAMPLE_RATE = 16_000


def mask_batch():
    q = DURATION / 8

    # Silence at start
    vad_list = [
        [[0, 2 * q], [6 * q, DURATION]],
        [[3 * q, 4 * q], [5 * q, 6 * q]],
    ]
    v0 = vad_list_to_onehot(vad_list, duration=DURATION, frame_hz=FRAME_HZ)
    vad_list = [
        [[3 * q, 4 * q], [5 * q, 6 * q]],
        [[0, 2 * q], [6 * q, DURATION]],
    ]
    v1 = vad_list_to_onehot(vad_list, duration=DURATION, frame_hz=FRAME_HZ)
    v = torch.stack((v0, v1))
    q_samples = int(q * SAMPLE_RATE)
    w = torch.ones(2, 2, DURATION * SAMPLE_RATE)
    return {"waveform": w, "vad": v}


@pytest.mark.callbacks
def test_vad_mask_callback():

    batch = mask_batch()
    ch_sums = batch["waveform"].abs().sum(-1)
    print(ch_sums)
    w = vad_mask_batch(batch["waveform"], batch["vad"], scale=0)
    new_sums = w.abs().sum(-1)

    for b in range(len(batch["vad"])):
        assert new_sums[b, 0] != ch_sums[b, 0]
        assert new_sums[b, 1] != ch_sums[b, 1]

    assert new_sums[0, 0] == 80000
    assert new_sums[0, 1] == 119680
    assert new_sums[1, 0] == 119680
    assert new_sums[1, 1] == 80000


@pytest.mark.callbacks
def test_flip_channel_callback():

    batch = mask_batch()
    batch["waveform"][0, 1] = 0
    batch["waveform"][1, 0] = 0

    old_w_sums = batch["waveform"].sum(dim=-1)
    old_v = batch["vad"]
    new_batch = flip_batch_channels(batch)

    # Wavform flip
    new_w_sums = new_batch["waveform"].sum(dim=-1)
    assert all(new_w_sums[:, 0] == old_w_sums[:, 1])
    assert all(new_w_sums[:, 1] == old_w_sums[:, 0])

    # VAD flip
    for b in range(2):
        assert all(new_batch["vad"][b, :, 1] == old_v[b, :, 0])
        assert all(new_batch["vad"][b, :, 0] == old_v[b, :, 1])
