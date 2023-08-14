import torch
from transformers import Wav2Vec2ForPreTraining

from vap.utils.utils import everything_deterministic

everything_deterministic()

VALID_CHECKPOINTS = ["facebook/mms-1b", "facebook/mms-300m"]


class EncoderMMS(torch.nn.Module):
    """
    Using the early convolutional layers of the MMS model (model.wav2vec.feature_extractor.conv_layers).
    Both the large (mms-1b) and the small (mms-300m) uses 7 convolutional layers projection
    16kHz -> 50Hz with 512 channels.

    The feature_projection layer is optional and is used to project the 512 channels to relevant sizes
    for the transformer.

    Pretrained weights source:
        https://huggingface.co/facebook/mms-1b

    CITATION
    @article{pratap2023mms,
        title={Scaling Speech Technology to 1,000+ Languages},
        author={Vineel Pratap and Andros Tjandra and Bowen Shi
                and Paden Tomasello and Arun Babu
                and Sayani Kundu and Ali Elkahky and Zhaoheng Ni
                and Apoorv Vyas and Maryam Fazel-Zarandi
                and Alexei Baevski and Yossi Adi and Xiaohui Zhang
                and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
        journal={arXiv},
        year={2023}
    }
    """

    def __init__(
        self,
        checkpoint: str = "facebook/mms-300m",
        use_feature_projection: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.use_feature_projection = use_feature_projection

        assert (
            checkpoint in VALID_CHECKPOINTS
        ), f"Invalid checkpoint: {checkpoint}. Valid: {VALID_CHECKPOINTS}"

        # Load model
        model = Wav2Vec2ForPreTraining.from_pretrained(checkpoint)

        # Faster than looping over conv_layers
        self.conv1 = model.wav2vec2.feature_extractor.conv_layers[0]
        self.conv2 = model.wav2vec2.feature_extractor.conv_layers[1]
        self.conv3 = model.wav2vec2.feature_extractor.conv_layers[2]
        self.conv4 = model.wav2vec2.feature_extractor.conv_layers[3]
        self.conv5 = model.wav2vec2.feature_extractor.conv_layers[4]
        self.conv6 = model.wav2vec2.feature_extractor.conv_layers[5]
        self.conv7 = model.wav2vec2.feature_extractor.conv_layers[6]

        self.dim = 512
        if self.use_feature_projection:
            self.dim = 1024 if "300m" in checkpoint else 1280
            self.feature_projection = model.wav2vec2.feature_projection

        if freeze:
            self.freeze_params()

    def freeze_params(self):
        for p in self.parameters():
            p.requires_grad = False
        print(f"MMS {self.checkpoint} Frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Input must be (B, 1, n_samples), got {x.shape}"
        assert x.shape[1] == 1, f"Input must be (B, 1, n_samples), got {x.shape}"

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x).transpose(1, 2)

        if self.use_feature_projection:
            x = self.feature_projection(x)[0]
            # class Wav2Vec2FeatureProjection(nn.Module):
            # def forward(self, hidden_states):
            #     # non-projected hidden states are needed for quantization
            #     norm_hidden_states = self.layer_norm(hidden_states)
            #     hidden_states = self.projection(norm_hidden_states)
            #     hidden_states = self.dropout(hidden_states)
            #     return hidden_states, norm_hidden_states

        return x


if __name__ == "__main__":

    from vap.modules.causal_testing import test_causal_samples_to_frames
    import time

    enc = EncoderMMS(use_feature_projection=True)

    x = torch.randn(1, 1, 160000)
    y = enc(x)

    N = 10
    t = time.time()
    for i in range(N):
        _ = enc(x)
    t = time.time() - t
    print(f"Transpose Time: {t/N:.5f}")

    is_causal, pre_grad, post_grad = test_causal_samples_to_frames(
        enc, pad_samples=399, wav_channels=1
    )
    print(f"Is causal: {is_causal}")
    print(f"Pre grad:  {pre_grad.sum().item():.2f}")
    print(f"Post grad: {post_grad.sum().item():.2f}")
