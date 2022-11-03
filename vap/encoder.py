import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from vap.encoder_components import load_CPC, get_cnn_layer


class Encoder(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, freeze=True, downsample=None):
        super().__init__()
        self.sample_rate = 16000
        self.encoder = load_CPC()
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample_ratio = 160
        if downsample is not None:
            self.downsample = get_cnn_layer(
                dim=self.output_dim,
                kernel=downsample["kernel"],
                stride=downsample["stride"],
                dilation=downsample["dilation"],
                activation=downsample["activation"],
            )
            if downsample["stride"] == [2]:
                if downsample["kernel"] == [5]:
                    self.downsample_ratio = 320
                else:
                    self.downsample_ratio = None
            elif downsample["stride"] == [5]:
                if downsample["kernel"] == [11]:
                    self.downsample_ratio = 800
                else:
                    self.downsample_ratio = None
            else:
                self.downsample_ratio = None
        else:
            self.downsample = nn.Identity()

        if freeze:
            self.freeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform):
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
