import pytest
from vap.modules.transformer import CausalTest, VapStereoTower


@pytest.mark.model
def test_self_attention():
    pass


@pytest.mark.model
@pytest.mark.parametrize(
    "dim,rotary_embeddings,cuda",
    [
        (256, False, False),
        (256, True, False),
        (256, False, True),
        (256, True, True),
        (512, False, False),
        (512, True, False),
    ],
)
def test_vap_stereo_tower(dim, rotary_embeddings, cuda):
    emb_type = "Rotary" if rotary_embeddings else "AliBI"
    vap_tower = VapStereoTower(rotary_embeddings=rotary_embeddings)
    if cuda:
        vap_tower = vap_tower.cuda()

    is_causal1, pre1, post1 = CausalTest.stereo_attn_causality(
        vap_tower, grad_channel=1, verbose=False
    )
    assert (
        is_causal1
    ), f"VapStereoTower({dim}, {emb_type}, cuda={cuda}) Ch 1.  Pre: {pre1}, Post: {post1}"

    is_causal2, pre2, post2 = CausalTest.stereo_attn_causality(
        vap_tower, grad_channel=2, verbose=False
    )
    assert (
        is_causal1
    ), f"VapStereoTower({dim}, {emb_type}, cuda={cuda}) Ch 1.  Pre: {pre1}, Post: {post1}"
