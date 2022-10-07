import pytest
import torch
from vap.transformer import (
    TransformerLayer,
    TransformerStereoLayer,
    GPT,
    GPTStereo,
)


@pytest.mark.transformer
def test_transformer_layer():
    layer = TransformerLayer(dim=256, ffn_dim=512, num_heads=8, cross_attention=False)
    x1 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    z1, self_attn_weights, cross_attn_weights = layer(x1)
    assert z1.shape == x1.shape, f"z1 {tuple(z1.shape)} != x1 {tuple(x1.shape)}"


@pytest.mark.transformer
def test_gpt():
    model = GPT(dim=256, dff_k=3, num_layers=4, num_heads=8)

    # Z2 -> X1 grads
    x1 = torch.rand((4, 20, model.dim)).requires_grad_(True)
    z = model(x1)
    z.sum().backward()
    n1 = x1.grad.abs().sum()
    assert z.shape == x1.shape, "Different shapes"
    assert n1 > 0, "No gradient from z -> x1"


@pytest.mark.transformer
@pytest.mark.stereo
def test_transformer_stereo_layer():
    layer = TransformerStereoLayer(
        dim=256, ffn_dim=512, num_heads=8, cross_attention=True
    )
    # Z2 -> X1 grads
    x1 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    x2 = torch.rand((4, 20, layer.dim))
    z1, z2, attn_list = layer(x1, x2)
    z2.sum().backward()
    n = x1.grad.abs().sum()
    assert n > 0, "(single) No gradient from z2 -> x1"
    assert z1.shape == x1.shape, f"z1 {tuple(z1.shape)} != x1 {tuple(x1.shape)}"
    assert z2.shape == x2.shape, f"z2 {tuple(z2.shape)} != x2 {tuple(x2.shape)}"

    # Z1 -> X2 grads
    x1 = torch.rand((4, 20, layer.dim))
    x2 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    z1, z2, attn_list = layer(x1, x2)
    z1.sum().backward()
    n = x2.grad.abs().sum()
    assert n > 0, "(single) No gradient from z1 -> x2"
    assert z1.shape == x1.shape, f"z1 {tuple(z1.shape)} != x1 {tuple(x1.shape)}"
    assert z2.shape == x2.shape, f"z2 {tuple(z2.shape)} != x2 {tuple(x2.shape)}"

    # Z1 -> X2 grads
    x1 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    x2 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    z1, z2, attn_list = layer(x1, x2)
    z1.sum().backward()
    n1 = x1.grad.abs().sum()
    n2 = x2.grad.abs().sum()
    assert n1 > 0, "(both) No gradient from z1 -> x1"
    assert n2 > 0, "(both) No gradient from z1 -> x2"
    assert z1.shape == x1.shape, f"z1 {tuple(z1.shape)} != x1 {tuple(x1.shape)}"
    assert z2.shape == x2.shape, f"z2 {tuple(z2.shape)} != x2 {tuple(x2.shape)}"


@pytest.mark.transformer
@pytest.mark.stereo
def test_gpt_stereo():
    model = GPTStereo(dim=256, dff_k=3, num_layers=4, num_heads=8)

    # Z2 -> X1 grads
    x1 = torch.rand((4, 20, model.dim)).requires_grad_(True)
    x2 = torch.rand((4, 20, model.dim)).requires_grad_(True)
    z = model(x1, x2)
    z.sum().backward()
    n1 = x1.grad.abs().sum()
    n2 = x1.grad.abs().sum()
    assert z.shape == x1.shape == x2.shape, "Different shapes"
    assert n1 > 0, "No gradient from z -> x1"
    assert n2 > 0, "No gradient from z -> x2"
