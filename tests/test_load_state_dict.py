import pytest
import torch
import zipfile
from pathlib import Path
from vap.modules.VAP import VAP
from vap.modules.encoder import EncoderCPC
from vap.modules.modules import TransformerStereo
from vap.modules.lightning_module import VAPModule

MODEL_STATE_DICT = "example/checkpoints/VAP_state_dict.pt"
MODULE_STATE_DICT = "example/checkpoints/VAPModule_state_dict.pt"
CHECKPOINT = "example/checkpoints/checkpoint.ckpt"


def test_model_state_dict():
    model = VAP(EncoderCPC(), TransformerStereo())
    l = model.load_state_dict(torch.load(MODEL_STATE_DICT))
    assert l.missing_keys == [], "Missing keys in state dict"
    assert l.unexpected_keys == [], "Unexpected keys in state dict"


def test_module_state_dict():
    model = VAP(EncoderCPC(), TransformerStereo())
    module = VAPModule(model)
    l = module.load_state_dict(torch.load(MODULE_STATE_DICT))
    assert l.missing_keys == [], "Missing keys in state dict"
    assert l.unexpected_keys == [], "Unexpected keys in state dict"


def test_checkpoint():
    p = Path(CHECKPOINT)
    if not p.exists():

        # unzip the checkpoint (replace extention with .zip)
        with zipfile.ZipFile(CHECKPOINT.replace(".ckpt", ".zip"), "r") as zip_ref:
            zip_ref.extractall(p.parent)

    VAPModule.load_from_checkpoint(CHECKPOINT, map_location="cpu")
