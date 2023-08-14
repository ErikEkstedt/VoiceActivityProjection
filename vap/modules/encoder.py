from torch import Tensor
import torch.nn as nn
import einops

from vap.modules.encoder_components import load_CPC, get_cnn_layer


class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, load_pretrained=True, freeze=True):
        super().__init__()
        self.sample_rate = 16000
        self.encoder = load_CPC(load_pretrained)
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample_ratio = 160
        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[5],
            stride=[2],
            dilation=[1],
            activation="GELU",
        )
        self.downsample_ratio = 320

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...
        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")
        z = self.encoder.gAR(z)
        z = self.downsample(z)
        return z
