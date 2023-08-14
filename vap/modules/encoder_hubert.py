import torch
from torch import Tensor
from transformers import HubertModel


CHECKPOINTS = {
    "english": {
        "base": "facebook/hubert-base-ls960",
        "large": "facebook/hubert-large-ll60k",
    },
    "japanese": {
        "base": "rinna/japanese-hubert-base",
    },
}


class EncoderHubert(torch.nn.Module):
    def __init__(
        self,
        pretrained_model: str = "facebook/hubert-base-ls960",
        output_dim: int = 256,
        causal: bool = True,
        only_feature_extractor: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.sample_rate = 16_000
        self.causal = causal
        self.output_dim = output_dim
        self.only_feature_extractor = only_feature_extractor
        self.pretrained_model = pretrained_model

        self.load_pretrained_model(pretrained_model)

        # features
        self.feature_dim = self.config.conv_dim[-1]
        self.transformer_dim = self.config.hidden_size

        self.projection = torch.nn.Identity()
        if self.only_feature_extractor:
            if self.feature_dim != self.output_dim:
                self.projection = self.create_projection(
                    self.feature_dim, self.output_dim, activation="GELU"
                )
        else:
            if self.transformer_dim != self.output_dim:
                self.projection = self.create_projection(
                    self.transformer_dim, self.output_dim, activation="GELU"
                )

        if freeze:
            self.freeze()

    def load_pretrained_model(self, pretrained_model: str):
        """
        Loading the pretrained Hubert model.

        Lightning uses 'deepcopy' to save the model's hyperparameters (self.save_hyperparameters)
        which does NOT work with `torch.nn.utils.weight_norm()`.

        ```bash
        RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
        ```

        See: https://github.com/pytorch/pytorch/issues/28594#issuecomment-1149882811

        In this model it is the convolutional positional embedding that is weight normalized.
        `hubert.encoder.pos_conv_embed.conv.weight` is a `torch.nn.utils.weight_norm()` module.

        The error occurs because `torch.nn.utils.weight_norm()` adds a `weight_g` and `weight_v` to the module
        and calculates the `weight` from these two parameters.
        Therefore the `weight` parameter is not a leaf in the graph and cannot be copied.

        SOLUTION:
            https://github.com/pytorch/pytorch/issues/28594#issuecomment-1149882811

        Set the weight to the `weight_v` parameter at initialization.
        """

        # Load model
        hubert = HubertModel.from_pretrained(pretrained_model)
        self.feature_extraction = hubert.feature_extractor
        self.feature_projection = hubert.feature_projection
        self.layer_norm = hubert.encoder.layer_norm
        self.layers = hubert.encoder.layers

        # Avoids 'deepcopy' errot on self.save_hyperparameters() in lightning_module
        # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
        # See: https://github.com/pytorch/pytorch/issues/28594
        self.pos_conv_embed = hubert.encoder.pos_conv_embed
        self.pos_conv_embed.conv.weight = self.pos_conv_embed.conv.weight_v.detach()
        self.config = hubert.config

    def create_projection(
        self, dim_in: int, dim_out: int, activation: str = "GELU"
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.LayerNorm(dim_out),
            getattr(torch.nn, activation)(),
        )

    def freeze(self):
        self.feature_extraction._freeze_parameters()
        for param in self.parameters():
            param.requires_grad = False

    def create_causal_mask(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3
        ), f"Expected x to b of [B, N_FRAMES, D] dimensions, got {x.shape}"
        b, n, _ = x.size()
        mask = torch.tril(torch.ones((n, n), device=x.device, dtype=torch.bool)).view(
            n, n
        )
        mask = mask.repeat(b, 1, 1).unsqueeze(1)
        mask.requires_grad_(False)
        return mask

    def extract_features(self, x: Tensor) -> Tensor:
        x = x.squeeze(1)
        return self.feature_extraction(x).transpose(1, 2)

    def transformer(self, x: Tensor) -> Tensor:
        """ """
        # Add positional embeddings
        hidden_states = self.feature_projection(x)
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings

        # Get causal mask
        mask = None
        if self.causal:
            mask = self.create_causal_mask(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3
        ), f"Expected x to be of [B, 1, N_SAMPLES] dimensions, got {x.shape}"
        assert (
            x.shape[1] == 1
        ), f"Expected x to be of [B, 1, N_SAMPLES] dimensions, got {x.shape}"
        extract_features = self.extract_features(x)

        if self.only_feature_extractor:
            return self.projection(extract_features)

        x = self.transformer(extract_features)
        return self.projection(x)


if __name__ == "__main__":

    from vap.modules.VAP import VAP
    from vap.modules.lightning_module import VAPModule
    from vap.modules.modules import TransformerStereo

    encoder = EncoderHubert()
    transformer = TransformerStereo()
    model = VAP(encoder, transformer)
    module = VAPModule(model)

    # for name, p in encoder.named_parameters():
    #     print(name, p.requires_grad)
    #     input()
    x = torch.randn(1, 2, 32000)
    out = model(x)
    out = module(x)
