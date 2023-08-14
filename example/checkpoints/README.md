# Checkpoints


## Checkpoints

* Saved during training using [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint)
* Checkpoint from training was too large for git (>100mb) so I included a zip file
    - `example/checkpoints/checkpoint.zip`
    - Simply extract the file to get `example/checkpoints/checkpoint.ckpt`


```python 
from vap.modules.lightning_module import VAPModule

module = VAPModule.load_from_checkpoint("example/checkpoints/checkpoint.ckpt")
```

## State dict:

* Requires a model (`torch.nn.Module` or `LightningModule`)
* [what is a state_dict in pytorch?](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)
* A module contains a `model` so the parameter names are prefixed with `model.`
    * `encoder.encoder.gEncoder.conv0.weight` -> `model.encoder.encoder.gEncoder.conv0.weight`
    * `transformer.ar.combinator.ln.bias` -> `model.transformer.ar.combinator.ln.bias`
    * `objective.codebook.emb.weight` -> `model.objective.codebook.emb.weight` -> 
    * `va_classifier.weight` -> `model.va_classifier.weight` ->          

```python
from vap.modules.VAP import VAP
from vap.modules.encoder import EncoderCPC
from vap.modules.modules import TransformerStereo
from vap.modules.lightning_module import VAPModule
from vap.data.datamodule import VAPDataModule

# Module
encoder = EncoderCPC()
transformer = TransformerStereo()
model = VAP(encoder, transformer)  # the barebones model
module = VAPModule(model)  # the lightning module

# Load state dict
sd_model = torch.load("example/checkpoints/VAP_state_dict.pt")
sd_module = torch.load("example/checkpoints/VAPModule_state_dict.pt")

# Load only state dict
model.load_state_dict(sd_model)
module.load_state_dict(sd_module)
```

