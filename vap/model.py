import torch
import torch.nn as nn
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.precision_recall_curve import PrecisionRecallCurve
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from typing import Optional, Dict

from vap.encoder import Encoder
from vap.transformer import GPT, GPTStereo
from vap.utils import (
    everything_deterministic,
    batch_to_device,
)

from vap_turn_taking.vap_new import VAP
from vap_turn_taking.events import TurnTakingEventsNew

everything_deterministic()


class VAPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_bins: int = 4,
        representation: str = "discrete",
        bias_w_distribution: bool = True,
    ):
        super().__init__()
        self.representation = representation
        self.output_dim = 1
        if self.representation == "comparative":
            self.projection_head = nn.Linear(input_dim, 1)
        else:
            self.total_bins = 2 * n_bins
            if self.representation == "independent":
                self.projection_head = nn.Sequential(
                    nn.Linear(input_dim, self.total_bins),
                    Rearrange("... (c f) -> ... c f", c=2, f=self.total_bins // 2),
                )
                self.output_dim = (2, n_bins)
            else:
                self.n_classes = 2 ** self.total_bins
                self.projection_head = nn.Linear(input_dim, self.n_classes)
                self.output_dim = self.n_classes
                if bias_w_distribution:
                    self.projection_head.bias.data = torch.load(
                        "example/label_probs.pt"
                    ).log()

    def __repr__(self):
        s = "VAPHead\n"
        s += f"  representation: {self.representation}"
        s += f"  output: {self.output_dim}"
        return super().__repr__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(x)


class VACondition(nn.Module):
    def __init__(
        self, dim: int, va_history: bool = False, va_history_bins: int = 5
    ) -> None:
        super().__init__()
        self.dim = dim
        self.va_history = va_history
        self.va_history_bins = va_history_bins
        self.va_cond = nn.Linear(2, dim)  # va: 2 one-hot encodings -> dim
        self.ln = nn.LayerNorm(dim)
        if va_history:
            # vah: (N, vah_bins) -> dim
            self.va_hist_cond = nn.Linear(va_history_bins, dim)

    def forward(
        self, vad: torch.Tensor, va_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        v_cond = self.va_cond(vad)

        # Add vad-history information
        if self.va_history and va_history is not None:
            v_cond += self.va_hist_cond(va_history)

        return self.ln(v_cond)


class ProjectionModel(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.stereo = conf.get("stereo", False)
        self.frame_hz = conf["frame_hz"]
        self.sample_rate = conf["sample_rate"]

        # Audio Encoder
        self.encoder = Encoder(
            freeze=conf["encoder"].get("freeze", True),
            downsample=conf["encoder"].get("downsample", None),
        )

        if self.encoder.output_dim != conf["ar"]["dim"]:
            self.projection = nn.Linear(self.encoder.output_dim, conf["ar"]["dim"])
        else:
            self.projection = nn.Identity()

        # VAD Conditioning
        if self.stereo:
            self.ar_channel = GPT(
                dim=conf["ar"]["dim"],
                dff_k=conf["ar"]["dff_k"],
                num_layers=conf["ar"]["channel_layers"],
                num_heads=conf["ar"]["num_heads"],
                dropout=conf["ar"]["dropout"],
            )
        else:
            self.vad_condition = VACondition(
                dim=self.encoder.output_dim,
                va_history=conf["va_cond"]["history"],
                va_history_bins=conf["va_cond"]["history_bins"],
            )

        # Autoregressive
        AR = GPTStereo if self.stereo else GPT
        self.ar = AR(
            dim=conf["ar"]["dim"],
            dff_k=conf["ar"]["dff_k"],
            num_layers=conf["ar"]["num_layers"],
            num_heads=conf["ar"]["num_heads"],
            dropout=conf["ar"]["dropout"],
        )

        # Appropriate VAP-head
        self.vap_representation = conf["vap"]["type"]
        self.vap_head = VAPHead(
            input_dim=conf["ar"]["dim"],
            n_bins=len(conf["vap"]["bin_times"]),
            representation=self.vap_representation,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        va: Optional[torch.Tensor] = None,
        va_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.stereo:
            # Placeholder before defining architecture
            x1 = self.encoder(waveform[:, :1])  # speaker 1
            x2 = self.encoder(waveform[:, 1:])  # speaker 2
            x1 = self.projection(x1)
            x2 = self.projection(x2)
            # Autoregressive
            x1 = self.ar_channel(x1)
            x2 = self.ar_channel(x2)
            z = self.ar(x1, x2)
        else:
            assert va is not None, "Requires voice-activity input but va=None"
            z = self.encoder(waveform)
            z = self.projection(z)

            # Ugly: sometimes you may get an extra frame from waveform encoding
            z = z[:, : va.shape[1]]

            # Vad conditioning... extra frames... Also Ugly...
            vc = self.vad_condition(va, va_history)[:, : z.shape[1]]

            # Add vad-conditioning to audio features
            z = z + vc
            # Autoregressive
            z = self.ar(z)
        logits = self.vap_head(z)
        return logits


class VAPModel(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.stereo = conf["model"].get("stereo", False)
        self.frame_hz = conf["model"]["frame_hz"]
        self.sample_rate = conf["model"]["sample_rate"]
        self.audio_duration_training = conf["model"]["audio_duration"]

        # Model
        self.net = ProjectionModel(conf["model"])

        # VAP: labels, logits -> zero-shot probs
        sh_opts = conf["events"]["shift_hold"]
        bc_opts = conf["events"]["backchannel"]
        sl_opts = conf["events"]["long_short"]
        mt_opts = conf["events"]["metric"]
        self.event_extractor = TurnTakingEventsNew(
            sh_pre_cond_time=sh_opts["pre_cond_time"],
            sh_post_cond_time=sh_opts["post_cond_time"],
            sh_prediction_region_on_active=sh_opts["post_cond_time"],
            bc_pre_cond_time=bc_opts["pre_cond_time"],
            bc_post_cond_time=bc_opts["post_cond_time"],
            bc_max_duration=bc_opts["max_duration"],
            bc_negative_pad_left_time=bc_opts["negative_pad_left_time"],
            bc_negative_pad_right_time=bc_opts["negative_pad_right_time"],
            prediction_region_time=mt_opts["prediction_region_time"],
            long_onset_region_time=sl_opts["onset_region_time"],
            long_onset_condition_time=sl_opts["onset_condition_time"],
            min_context_time=mt_opts["min_context"],
            metric_time=mt_opts["pad_time"],
            metric_pad_time=mt_opts["pad_time"],
            max_time=self.audio_duration_training,
            frame_hz=self.frame_hz,
            equal_hold_shift=sh_opts["pre_cond_time"],
        )
        self.VAP = VAP(
            objective=conf["model"]["vap"]["type"],
            bin_times=conf["model"]["vap"]["bin_times"],
            frame_hz=conf["model"]["frame_hz"],
            pre_frames=conf["model"]["vap"]["pre_frames"],
            threshold_ratio=conf["model"]["vap"]["bin_threshold"],
        )
        self.vad_history_times = self.conf["data"]["vad_history_times"]
        self.horizon = self.VAP.horizon
        self.horizon_time = self.VAP.horizon_time

        # Training params
        self.learning_rate = conf["optimizer"]["learning_rate"]
        self.save_hyperparameters()

        # Metrics
        self.val_hs_metric = F1Score(num_classes=2, average="weighted", multiclass=True)
        self.val_ls_metric = F1Score(num_classes=2, average="weighted", multiclass=True)
        self.val_sp_metric = F1Score(num_classes=2, average="weighted", multiclass=True)
        self.val_bp_metric = F1Score(num_classes=2, average="weighted", multiclass=True)

    @property
    def run_name(self):
        s = "VAP"
        s += f"_{self.frame_hz}Hz"
        s += f"_ad{self.audio_duration_training}s"
        s += f"_{self.conf['model']['ar']['channel_layers']}"
        s += str(self.conf["model"]["ar"]["num_layers"])
        s += str(self.conf["model"]["ar"]["num_heads"])
        if not self.stereo:
            s += "_mono"
        return s

    def forward(
        self,
        waveform: torch.Tensor,
        va: Optional[torch.Tensor] = None,
        va_history: Optional[torch.Tensor] = None,
        return_events: bool = False,
    ):
        assert (
            waveform.ndim == 3
        ), f"Expects (B, N_CHANNEL, N_SAMPLES) got {waveform.shape}"

        if va is not None:
            assert va.ndim == 3, f"Expects (B, N_FRAMES, 2) got {va.shape}"

        if va_history is not None:
            assert (
                va_history.ndim == 3
            ), f"Expects (B, N_FRAMES, 5) got {va_history.shape}"

        logits = self.net(waveform, va=va, va_history=va_history)
        vap_out = self.VAP(logits=logits, va=va)
        vap_out["logits"] = logits

        n_max_frames = logits.shape[1] - self.horizon

        if return_events:
            events = self.event_extractor(va, max_frame=n_max_frames)
            vap_out.update(events)

        # Output keys
        #  ["probs", "logits"]
        #  if va is not None:
        #  ["probs", "p", "p_bc", "labels", "logits"]
        #  if return_events:
        #       ['shift', 'hold', 'short', 'long',
        #       'pred_shift', 'pred_shift_neg',
        #       'pred_bc', 'pred_bc_neg']
        return vap_out

    def configure_optimizers(self) -> Dict:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.conf["optimizer"]["betas"],
            weight_decay=self.conf["optimizer"]["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt,
            T_max=self.conf["optimizer"].get("lr_scheduler_tmax", 10),
            last_epoch=-1,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.conf["optimizer"].get("lr_scheduler_interval", "step"),
                "frequency": self.conf["optimizer"].get("lr_scheduler_freq", 1000),
            },
        }

    def shared_step(
        self, batch: Dict, reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            batch:      dict, containing 'waveform', va, va_history

        Returns:
            loss:       torch.Tensor
            out:        dict
            batch:      same as input arguments (fixed for differenct encoder Hz)
        """

        n_max_frames = batch["vad"].shape[1] - self.horizon

        ########################################
        # VA-history
        ########################################
        vah_input = None
        if "vad_history" in batch:
            vah_input = batch["vad_history"][:, :n_max_frames]

        ########################################
        # Forward pass -> logits: torch.Tensor
        ########################################
        logits = self.net(
            waveform=batch["waveform"],
            va=batch["vad"][:, :n_max_frames],
            va_history=vah_input,
        )

        ########################################
        # VAP-Head: Extract Probs and Labels
        vap_out = self.VAP(logits=logits, va=batch["vad"])
        vap_out["logits"] = logits
        ########################################

        loss = self.VAP.loss_fn(logits, vap_out["labels"])
        vap_out["loss"] = loss
        return vap_out

    def training_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", out["loss"], batch_size=batch_size, sync_dist=True)
        return {"loss": out["loss"]}

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""

        # Regular forward pass
        out = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        # log validation loss
        self.log("val_loss", out["loss"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        events = self.event_extractor(batch["vad"])

        preds, targets = self.VAP.extract_prediction_and_targets(
            p=out["p"], p_bc=out["p_bc"], events=events
        )

        # How to log torchmetrics
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        #   self.valid_acc(logits, y)
        #   self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        # Otherwise try manual

        # Update... I presume
        if preds["hs"] is not None:
            self.val_hs_metric(preds=preds["hs"], target=targets["hs"])
        if preds["ls"] is not None:
            self.val_ls_metric(preds=preds["ls"], target=targets["ls"])
        if preds["pred_shift"] is not None:
            self.val_sp_metric(preds=preds["pred_shift"], target=targets["pred_shift"])
        if preds["pred_backchannel"] is not None:
            self.val_bp_metric(
                preds=preds["pred_backchannel"], target=targets["pred_backchannel"]
            )
        # Log
        self.log("val_f1_hs", self.val_hs_metric, on_step=True, on_epoch=True)
        self.log("val_f1_ls", self.val_ls_metric, on_step=True, on_epoch=True)
        self.log("val_f1_pred_sh", self.val_sp_metric, on_step=True, on_epoch=True)
        self.log("val_f1_preed_bc", self.val_bp_metric, on_step=True, on_epoch=True)


if __name__ == "__main__":

    from os import cpu_count
    from datasets_turntaking import DialogAudioDM
    from vap.utils import load_hydra_conf

    conf = load_hydra_conf()
    config_name = "model/vap_50hz"  # "model/vap_50hz_stereo"
    conf["model"] = load_hydra_conf(config_name=config_name)["model"]
    model = VAPModel(conf)
    print(model.run_name)

    dm = DialogAudioDM(
        datasets=["switchboard", "fisher"],
        audio_duration=conf["data"]["audio_duration"],
        audio_mono=not model.stereo,
        batch_size=4,
        num_workers=cpu_count(),
    )
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(accelerator="gpu", devices=-1, fast_dev_run=1)
    trainer.fit(model, datamodule=dm)
