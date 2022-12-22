import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from vap.encoder import EncoderCPC
from vap.objective import ObjectiveVAP
from vap.modules import GPT, GPTStereo
from vap.utils import everything_deterministic


everything_deterministic()

# TODO: @dataclass and CLI arguments or is hydra the way to go?
# TODO: Easy finetune task
# TODO: What to evaluate


def load_older_state_dict(
    path="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt",
):
    sd = torch.load(path)["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if "VAP.codebook" in k:
            continue
        if "vap_head" in k:
            k = k.replace("vap_head.projection_head", "vap_head")
        new_sd[k.replace("net.", "")] = v
    return new_sd


@dataclass
class VapConfig:
    sample_rate: int = 16_000
    frame_hz: int = 50
    bin_times = [0.2, 0.4, 0.6, 0.8]

    # Encoder
    freeze_encoder: bool = True

    # GPT
    dim: int = 256
    channel_layers: int = 1
    cross_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1

    @staticmethod
    def add_argparse_args(parser, fields_added=[]):
        for k, v in VapConfig.__dataclass_fields__.items():
            parser.add_argument(f"--vap_{k}", type=v.type, default=v.default)
            fields_added.append(k)
        return parser, fields_added

    @staticmethod
    def args_to_conf(args):
        return VapConfig(
            **{
                k.replace("vap_", ""): v
                for k, v in vars(args).items()
                if k.startswith("vap_")
            }
        )


class VapGPT(nn.Module):
    def __init__(self, conf: Optional[VapConfig] = None):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        # Audio Encoder
        self.encoder = EncoderCPC(freeze=conf.freeze_encoder)

        # Single channel
        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
        )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        # Outputs
        # Voice activity objective -> x1, x2 -> logits ->  BCE
        self.va_classifier = nn.Linear(conf.dim, 1)
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)

    def encode_audio(self, audio: torch.Tensor) -> Tuple[Tensor, Tensor]:
        assert (
            audio.shape[1] == 2
        ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
        x1 = self.encoder(audio[:, :1])  # speaker 1
        x2 = self.encoder(audio[:, 1:])  # speaker 2
        return x1, x2

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    @torch.no_grad()
    def probs(
        self,
        waveform: Tensor,
        now_lims: List[int] = [0, 1],
        future_lims: List[int] = [2, 3],
    ) -> Dict[str, Tensor]:
        out = self(waveform)
        probs = out["logits"].softmax(dim=-1)
        vad = out["vad"].sigmoid()

        # Calculate entropy over each projection-window prediction (i.e. over
        # frames/time) If we have C=256 possible states the maximum bit entropy
        # is 8 (2^8 = 256) this means that the model have a one in 256 chance
        # to randomly be right. The model can't do better than to uniformly
        # guess each state, it has learned (less than) nothing. We want the
        # model to have low entropy over the course of a dialog, "thinks it
        # understands how the dialog is going", it's a measure of how close the
        # information in the unseen data is to the knowledge encoded in the
        # training data.
        h = -probs * probs.log2()  # Entropy
        H = h.sum(dim=-1)  # average entropy per frame

        # first two bins
        p_now = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=now_lims[0], to_bin=now_lims[-1]
        )
        p_future = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=future_lims[0], to_bin=future_lims[1]
        )
        return {
            "probs": probs,
            "vad": vad,
            "p_now": p_now,
            "p_future": p_future,
            "H": H,
        }

    def forward(self, waveform: Tensor, attention: bool = False) -> Dict[str, Tensor]:
        x1, x2 = self.encode_audio(waveform)

        # Autoregressive
        o1 = self.ar_channel(x1, attention=attention)  # ["x"]
        o2 = self.ar_channel(x2, attention=attention)  # ["x"]
        out = self.ar(o1["x"], o2["x"], attention=attention)

        # Outputs
        v1 = self.va_classifier(out["x1"])
        v2 = self.va_classifier(out["x2"])
        vad = torch.cat((v1, v2), dim=-1)
        logits = self.vap_head(out["x"])

        ret = {"logits": logits, "vad": vad}
        if attention:
            ret["self_attn"] = torch.stack([o1["attn"], o2["attn"]], dim=1)
            ret["cross_attn"] = out["cross_attn"]
            ret["cross_self_attn"] = out["self_attn"]
        return ret


if __name__ == "__main__":
    from datasets_turntaking import DialogAudioDM
    from vap.utils import batch_to_device
    from vap.events import TurnTakingEvents
    from vap.zero_shot import ZeroShot

    dm = DialogAudioDM(
        datasets=["switchboard", "fisher"],
        audio_duration=20,
        batch_size=2,
        num_workers=1,
        flip_channels=True,
        flip_probability=0.5,
        mask_vad=False,
        mask_vad_probability=0.4,
    )
    dm.prepare_data()
    dm.setup()

    conf = VapConfig()
    model = VapGPT(conf)
    sd = load_older_state_dict()
    model.load_state_dict(sd, strict=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = model.half()

    batch = next(iter(dm.val_dataloader()))
    batch = batch_to_device(batch, "cuda")
    with torch.no_grad():
        labels = model.objective.get_labels(batch["vad"].half())
        out = model(waveform=batch["waveform"].half())
        out["vap_loss"] = model.objective.loss_vap(out["logits"], labels)
        out["vad_loss"] = model.objective.loss_vad(out["vad"], batch["vad"])
    print("vap_oss: ", out["vap_loss"])
    print("vad_Loss: ", out["vad_loss"])

    # from vap.plot_utils import plot_mel_spectrogram, plot_vad, plot_event
    #
    # eventer = TurnTakingEvents()
    # zs = ZeroShot()
    #
    # threshold = 0.5
    #
    # with torch.no_grad():
    #     for batch in dm.val_dataloader():
    #         out = model(batch["waveform"].to("cuda"))
    #         labels, ds_labels = model.objective.get_labels(
    #             batch["vad"].to("cuda"), ds_label=True
    #         )
    #         probs = zs.get_probs(out["logits"].cpu(), batch["vad"])
    #         nmax = labels.shape[1]
    #         events = eventer(batch["vad"][:, :nmax])
    #         preds, targets = zs.extract_prediction_and_targets(
    #             p=probs["p"], p_bc=probs["p_bc"], events=events
    #         )
    #         result = {}
    #         for k, v in preds.items():
    #             if v is not None:
    #                 pred_class = v.round()
    #                 correct = (pred_class == targets[k]).sum()
    #                 acc = correct / targets[k].nelement()
    #                 result[k] = acc
    #                 # print(f"{k}: {acc}")
    #         n_shift = sum([len(s) for s in events["shift"]])
    #         if n_shift > 0:
    #             print("hs: ", result["hs"])
    #             # for k, v in result.items():
    #             #     print(f"{k}: {v}")
    #             # print('#'*40)
    #
    #     for b in range(2):
    #         x = torch.arange(batch["vad"].shape[1] - 100) / 50
    #         plt.close("all")
    #         fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 4))
    #         plot_mel_spectrogram(y=batch["waveform"][b], ax=ax)
    #         plot_vad(x, batch["vad"][b, :nmax, 0], ax=ax[0], ypad=5)
    #         plot_vad(x, batch["vad"][b, :nmax, 1], ax=ax[1], ypad=5)
    #         plot_event(events["shift"][b], ax=ax, color="g")
    #         plot_event(events["hold"][b], ax=ax, color="b")
    #         plot_event(events["short"][b], ax=ax)
    #         ax[-1].plot(x, ds_labels[b], linewidth=2)
    #         ax[-1].set_ylim([0, 2])
    #         # ax[c].axvline(s/50, color='g', linewidth=2)
    #         # ax[c].axvline(e/50, color='r', linewidth=2)
    #         plt.tight_layout()
    #         plt.show()
    #         # plt.pause(0.1)
