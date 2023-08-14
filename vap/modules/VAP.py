import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import Tensor
from typing import Optional

from vap.objective import VAPObjective
from vap.utils.utils import (
    everything_deterministic,
    vad_fill_silences,
    vad_omit_spikes,
)

from vap.modules.modules import ProjectionLayer

OUT = dict[str, Tensor]

everything_deterministic()


def load_model_from_state_dict(path: str):
    from vap.modules.encoder import EncoderCPC
    from vap.modules.modules import TransformerStereo

    def load_encoder(sd):
        encoder = None
        if "encoder.encoder.gEncoder.conv0.weight" in sd:
            encoder = EncoderCPC()
        else:
            raise NotImplementedError("Only EncoderCPC is implemented")
        return encoder

    def load_transformer(sd):
        def get_transformer_layers(sd, layer_type="self"):
            layer_name = "ar_channel"
            if layer_type == "cross":
                layer_name = "ar"

            n = 0
            if f"transformer.{layer_name}.layers.1.ln_self_attn.weight" in sd.keys():
                while True:
                    m = n + 1
                    if (
                        f"transformer.{layer_name}.layers.{m}.ln_self_attn.weight"
                        in sd.keys()
                    ):
                        n = m
                    else:
                        break
            return n + 1

        dim = sd["transformer.ar_channel.layers.0.ln_self_attn.weight"].shape[0]
        dff = sd["transformer.ar_channel.layers.0.ffnetwork.0.weight"].shape[0]
        assert dff % dim == 0, "dff must be a multiple of dim"
        dff_k = int(dff / dim)
        num_heads = sd["transformer.ar_channel.layers.0.mha.m"].shape[0]
        self_layers = get_transformer_layers(sd, layer_type="self")
        cross_layers = get_transformer_layers(sd, layer_type="cross")

        return TransformerStereo(
            dim=dim,
            self_layers=self_layers,
            cross_layers=cross_layers,
            num_heads=num_heads,
            dff_k=dff_k,
        )

    p = Path(path)
    assert p.exists(), f"Path does not exist: {p}"

    sd = torch.load(p)
    E = load_encoder(sd)
    T = load_transformer(sd)
    model = VAP(E, T)
    model.load_state_dict(sd)
    return model


def step_extraction(
    waveform,
    model,
    chunk_time=20,
    step_time=5,
    pbar=True,
    verbose=False,
):
    """
    Takes a waveform, the model, and extracts probability output in chunks with
    a specific context and step time. Concatenates the output accordingly and returns full waveform output.
    """

    n_samples = waveform.shape[-1]
    duration = round(n_samples / model.sample_rate, 2)
    context_time = chunk_time - step_time

    # Samples
    # context_samples = int(context_time * model.sample_rate)
    step_samples = int(step_time * model.sample_rate)
    chunk_samples = int(chunk_time * model.sample_rate)

    # Frames
    # context_frames = int(context_time * model.frame_hz)
    chunk_frames = int(chunk_time * model.frame_hz)
    step_frames = int(step_time * model.frame_hz)

    # Fold the waveform to get total chunks
    folds = waveform.unfold(
        dimension=-1, size=chunk_samples, step=step_samples
    ).permute(2, 0, 1, 3)
    print("folds: ", tuple(folds.shape))

    expected_frames = round(duration * model.frame_hz)
    n_folds = int((n_samples - chunk_samples) / step_samples + 1.0)
    total = (n_folds - 1) * step_samples + chunk_samples

    # First chunk
    # Use all extracted data. Does not overlap with anything prior.
    out = model.probs(folds[0].to(model.device))
    # OUT:
    # {
    #   "probs": probs,
    #   "vad": vad,
    #   "p_now": p_now,
    #   "p_future": p_future,
    #   "H": H,
    # }

    if pbar:
        from tqdm import tqdm

        pbar = tqdm(folds[1:], desc=f"Context: {context_time}s, step: {step_time}")
    else:
        pbar = folds[1:]
    # Iterate over all other folds
    # and add simply the new processed step
    for w in pbar:
        o = model.probs(w.to(model.device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -step_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -step_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -step_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -step_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -step_frames:]], dim=1)
        # out["p_zero_shot"] = torch.cat([out["p_zero_shot"], o["p_zero_shot"][:, -step_frames:]], dim=1)

    processed_frames = out["p_now"].shape[1]

    ###################################################################
    # Handle LAST SEGMENT (not included in `unfold`)
    ###################################################################
    if expected_frames != processed_frames:
        omitted_frames = expected_frames - processed_frames

        omitted_samples = model.sample_rate * omitted_frames / model.frame_hz

        if verbose:
            print(f"Expected frames {expected_frames} != {processed_frames}")
            print(f"omitted frames: {omitted_frames}")
            print(f"omitted samples: {omitted_samples}")
            print(f"chunk_samples: {chunk_samples}")

        w = waveform[..., -chunk_samples:]
        o = model.probs(w.to(model.device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -omitted_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -omitted_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -omitted_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -omitted_frames:]], dim=1)
    return out


class VAP(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        bin_times: list[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
    ):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.objective = VAPObjective(bin_times=bin_times, frame_hz=frame_hz)
        self.frame_hz = frame_hz
        self.dim: int = getattr(self.transformer, "dim", 256)

        self.feature_projection = nn.Identity()
        if self.encoder.dim != self.transformer.dim:
            self.feature_projection = ProjectionLayer(
                self.encoder.dim, self.transformer.dim
            )

        # Outputs
        # Voice activity objective -> x1, x2 -> logits ->  BCE
        self.va_classifier = nn.Linear(self.dim, 1)
        self.vap_head = nn.Linear(self.dim, self.objective.n_classes)

    @property
    def horizon_time(self) -> float:
        return self.objective.horizon_time

    @property
    def sample_rate(self) -> int:
        return self.encoder.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def extract_labels(self, vad: Tensor) -> Tensor:
        return self.objective.get_labels(vad)

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    def encode_audio(self, audio: torch.Tensor) -> tuple[Tensor, Tensor]:
        assert (
            audio.shape[1] == 2
        ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
        x1 = self.encoder(audio[:, :1])  # speaker 1
        x2 = self.encoder(audio[:, 1:])  # speaker 2
        return x1, x2

    def head(self, x: Tensor, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        v1 = self.va_classifier(x1)
        v2 = self.va_classifier(x2)
        vad = torch.cat((v1, v2), dim=-1)
        logits = self.vap_head(x)
        return logits, vad

    def forward(self, waveform: Tensor, attention: bool = False) -> OUT:
        x1, x2 = self.encode_audio(waveform)
        x1 = self.feature_projection(x1)
        x2 = self.feature_projection(x2)
        out = self.transformer(x1, x2, attention=attention)
        logits, vad = self.head(out["x"], out["x1"], out["x2"])
        out["logits"] = logits
        out["vad"] = vad
        return out

    def entropy(self, probs: Tensor) -> Tensor:
        """
        Calculate entropy over each projection-window prediction (i.e. over
        frames/time) If we have C=256 possible states the maximum bit entropy
        is 8 (2^8 = 256) this means that the model have a one in 256 chance
        to randomly be right. The model can't do better than to uniformly
        guess each state, it has learned (less than) nothing. We want the
        model to have low entropy over the course of a dialog, "thinks it
        understands how the dialog is going", it's a measure of how close the
        information in the unseen data is to the knowledge encoded in the
        training data.
        """
        h = -probs * probs.log2()  # Entropy
        return h.sum(dim=-1).cpu()  # average entropy per frame

    def aggregate_probs(
        self,
        probs: Tensor,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ) -> dict[str, Tensor]:
        # first two bins
        p_now = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=now_lims[0], to_bin=now_lims[-1]
        ).cpu()
        p_future = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=future_lims[0], to_bin=future_lims[1]
        ).cpu()
        # P over all
        max_idx = self.objective.n_bins - 1
        pa = self.objective.probs_next_speaker_aggregate(probs, 0, max_idx).cpu()
        p = []
        for i in range(0, max_idx + 1):
            p.append(self.objective.probs_next_speaker_aggregate(probs, i, i).cpu())
        p = torch.stack(p)
        return {
            "p_now": p_now,
            "p_future": p_future,
            "p_all": pa,
            "p": p,
        }

    @torch.inference_mode()
    def get_shift_probability(
        self, out: OUT, start_time: float, end_time: float, speaker
    ) -> dict[str, list[float]]:
        """
        Get shift probabilities (for classification) over the region `[start_time, end_time]`

        The `speaker` is the speaker before the silence, i.e. the speaker of the target IPU

        Shapes:
        out['p']:           (4, n_batch, n_frames)
        out['p_now']:       (n_batch, n_frames)
        out['p_future']:    (n_batch, n_frames)
        """
        region_start = int(start_time * self.frame_hz)
        region_end = int(end_time * self.frame_hz)
        ps = out["p"][..., region_start:region_end].mean(-1).cpu()
        pn = out["p_now"][..., region_start:region_end].mean(-1).cpu()
        pf = out["p_future"][..., region_start:region_end].mean(-1).cpu()

        batch_size = pn.shape[0]

        # if batch size == 1
        if batch_size == 1:
            speaker = [speaker]

        # Make all values 'shift probabilities'
        # The speaker is the speaker of the target IPU
        # A shift is the probability of the other speaker
        # The predictions values are always for the first speaker
        # So if the current speaker is speaker 1 then the probability of the default
        # speaker is the same as the shift-probability
        # However, if the current speaker is speaker 0 then the default probabilities
        # are HOLD probabilities, so we need to invert them
        for ii, spk in enumerate(speaker):
            if spk == 0:
                ps[:, ii] = 1 - ps[:, ii]
                pn[ii] = 1 - pn[ii]
                pf[ii] = 1 - pf[ii]

        preds = {f"p{k+1}": v.tolist() for k, v in enumerate(ps)}
        preds["p_now"] = pn.tolist()
        preds["p_fut"] = pf.tolist()
        return preds

    @torch.inference_mode()
    def probs(
        self,
        waveform: Tensor,
        vad: Optional[Tensor] = None,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ) -> OUT:
        """"""
        out = self(waveform)
        probs = out["logits"].softmax(dim=-1)
        vap_vad = out["vad"].sigmoid()
        h = self.entropy(probs)
        ret = {
            "probs": probs,
            "vad": vap_vad,
            "H": h,
        }

        # Next speaker aggregate probs
        probs_agg = self.aggregate_probs(probs, now_lims, future_lims)
        ret.update(probs_agg)

        # If ground truth voice activity is known we can calculate the loss
        if vad is not None:
            labels = self.objective.get_labels(vad)
            ret["loss"] = self.objective.loss_vap(
                out["logits"], labels, reduction="none"
            ).cpu()
        return ret

    @torch.inference_mode()
    def vad(
        self,
        waveform: Tensor,
        max_fill_silence_time: float = 0.02,
        max_omit_spike_time: float = 0.02,
        vad_cutoff: float = 0.5,
    ) -> Tensor:
        """
        Extract (binary) Voice Activity Detection from model
        """
        vad = (self(waveform)["vad"].sigmoid() >= vad_cutoff).float()
        for b in range(vad.shape[0]):
            # TODO: which order is better?
            vad[b] = vad_fill_silences(
                vad[b], max_fill_time=max_fill_silence_time, frame_hz=self.frame_hz
            )
            vad[b] = vad_omit_spikes(
                vad[b], max_omit_time=max_omit_spike_time, frame_hz=self.frame_hz
            )
        return vad


if __name__ == "__main__":
    from vap.modules.encoder import EncoderCPC
    from vap.modules.modules import TransformerStereo

    encoder = EncoderCPC()
    transformer = TransformerStereo(dim=512)

    model = VAP(encoder, transformer)
    print(model)

    x = torch.randn(1, 2, 32000)
    out = model(x)
