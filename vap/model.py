import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import einops
from einops.layers.torch import Rearrange
from typing import Optional, Dict

from vap.encoder import Encoder
from vap.transformer import GPT, GPTStereo
from vap.utils import (
    everything_deterministic,
    batch_to_device,
    get_audio_info,
    time_to_frames,
)
from vap_turn_taking import VAP, TurnTakingMetrics
from vap_turn_taking.utils import vad_list_to_onehot, get_activity_history
from datasets_turntaking.utils import load_waveform

everything_deterministic()


def loss_vad_projection(
    logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    # CrossEntropyLoss over discrete labels
    loss = F.cross_entropy(
        einops.rearrange(logits, "b n d -> (b n) d"),
        einops.rearrange(labels, "b n -> (b n)"),
        reduction=reduction,
    )
    if reduction == "none":
        n = logits.shape[1]
        loss = einops.rearrange(loss, "(b n) -> b n", n=n)
    return loss


class VAPHead(nn.Module):
    def __init__(
        self, input_dim: int, n_bins: int = 4, representation: str = "discrete"
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

    def __repr__(self):
        s = "VAPHead\n"
        s += f"  representation: {self.representation}"
        s += f"  output: {self.output_dim}"
        return super().__repr__()

    def forward(self, x) -> torch.Tensor:
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

    def init(self) -> None:
        nn.init.orthogonal_(self.va_condition.weight.data)

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

        self.net = ProjectionModel(conf["model"])

        # VAP: labels, logits -> zero-shot probs
        self.VAP = VAP(
            type=conf["model"]["vap"]["type"],
            bin_times=conf["model"]["vap"]["bin_times"],
            frame_hz=conf["model"]["frame_hz"],
            pre_frames=conf["model"]["vap"]["pre_frames"],
            threshold_ratio=conf["model"]["vap"]["bin_threshold"],
        )
        self.vad_history_times = self.conf["data"]["vad_history_times"]
        self.horizon_frames = self.VAP.horizon_frames
        self.horizon_time = self.VAP.horizon

        # Metrics
        self.val_metric = None  # self.init_metric()
        self.test_metric = None  # set in test if necessary

        # Training params
        self.learning_rate = conf["optimizer"]["learning_rate"]
        self.save_hyperparameters()

    @property
    def run_name(self):
        """
        -> 50hz_stereo_134_20s
        -> 50hz_44_20s
        -> 50hz_44_20s_ind_40
        -> 50hz_44_20s_cmp
        """
        conf = self.conf["model"]
        name = f"{conf['frame_hz']}hz"
        if self.stereo:
            name += "_stereo"
            name += f"_{conf['ar']['channel_layers']}{conf['ar']['num_layers']}{conf['ar']['num_heads']}"
        else:
            name += f"_{conf['ar']['num_layers']}{conf['ar']['num_heads']}"
        name += f'_{conf["audio_duration"]}s'
        if self.net.vap_head.representation == "comparative":
            name += "_cmp"
        elif self.net.vap_head.representation == "independent":
            n_bins = len(conf["vap"]["bin_times"])
            name += f"_ind_{n_bins}"
        return name

    def summary(self):
        s = "VAPModel\n"
        s += f"{self.net}"
        s += f"{self.VAP}"
        return s

    def init_metric(
        self,
        conf=None,
        threshold_pred_shift=None,
        threshold_short_long=None,
        threshold_bc_pred=None,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
    ):
        if conf is None:
            conf = self.conf

        if threshold_pred_shift is None:
            threshold_pred_shift = conf["events"]["threshold"]["S_pred"]

        if threshold_bc_pred is None:
            threshold_bc_pred = conf["events"]["threshold"]["BC_pred"]

        if threshold_short_long is None:
            threshold_short_long = conf["events"]["threshold"]["SL"]

        metric = TurnTakingMetrics(
            hs_kwargs=conf["events"]["SH"],
            bc_kwargs=conf["events"]["BC"],
            metric_kwargs=conf["events"]["metric"],
            threshold_pred_shift=threshold_pred_shift,
            threshold_short_long=threshold_short_long,
            threshold_bc_pred=threshold_bc_pred,
            shift_pred_pr_curve=shift_pred_pr_curve,
            bc_pred_pr_curve=bc_pred_pr_curve,
            long_short_pr_curve=long_short_pr_curve,
            frame_hz=self.frame_hz,
        )
        metric = metric.to(self.device)
        return metric

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.conf["optimizer"]["train_encoder_epoch"]:
            self.net.encoder.unfreeze()

    def on_test_epoch_start(self) -> None:
        if self.test_metric is None:
            self.test_metric = self.init_metric()
            self.test_metric.to(self.device)
        else:
            self.test_metric.reset()

    def on_validation_epoch_start(self) -> None:
        if self.val_metric is None:
            self.val_metric = self.init_metric()
            self.val_metric.to(self.device)
        else:
            self.val_metric.reset()

    def validation_epoch_end(self, outputs) -> None:
        r = self.val_metric.compute()
        self._log(r, split="val")

    def test_epoch_end(self, outputs) -> None:
        r = self.test_metric.compute()
        self._log(r, split="test")

    def _log(self, result, split="val"):
        for metric_name, values in result.items():
            if metric_name.startswith("pr_curve"):
                continue

            if metric_name.endswith("support"):
                continue

            if isinstance(values, dict):
                for val_name, val in values.items():
                    if val_name == "support":
                        continue
                    self.log(
                        f"{split}_{metric_name}_{val_name}", val.float(), sync_dist=True
                    )
            else:
                self.log(f"{split}_{metric_name}", values.float(), sync_dist=True)

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

    def forward(self, *args, **kwargs):
        logits = self.net(*args, **kwargs)
        return {"logits": logits}

    def calc_losses(self, logits, va_labels, reduction="mean"):
        if self.net.vap_head.representation == "comparative":
            loss = F.binary_cross_entropy_with_logits(logits, va_labels.unsqueeze(-1))
        elif self.net.vap_head.representation == "independent":
            loss = F.binary_cross_entropy_with_logits(
                logits, va_labels, reduction=reduction
            )
        else:
            loss = loss_vad_projection(
                logits=logits, labels=va_labels, reduction=reduction
            )
        return loss

    def shared_step(self, batch, reduction="mean"):
        """
        Arguments:
            batch:      dict, containing 'waveform', va, va_history

        Returns:
            loss:       torch.Tensor
            out:        dict
            batch:      same as input arguments (fixed for differenct encoder Hz)
        """

        va_input, vah_input, va_labels = None, None, None
        if "vad" in batch:
            va_labels = self.VAP.extract_label(va=batch["vad"])
            n_valid = va_labels.shape[1]  # Only keep the relevant vad information
            va_input = batch["vad"][:, :n_valid]
            if "vad_history" in batch and batch["vad_history"] is not None:
                vah_input = batch["vad_history"][:, :n_valid]

        # Forward pass -> {'logits': torch.Tensor}
        out = self(waveform=batch["waveform"], va=va_input, va_history=vah_input)

        loss = {}
        if va_labels is not None:
            out["va_labels"] = va_labels
            batch["vad"] = va_input
            batch["vad_history"] = vah_input
            logits = out["logits"][:, : va_labels.shape[1]]

            # Calculate Loss
            logit_loss = self.calc_losses(
                logits=logits,
                va_labels=va_labels,
                reduction=reduction,
            )
            loss["loss"] = logit_loss.mean()
            if reduction == "none":
                loss["loss_frames"] = logit_loss
        return loss, out, batch

    def load_sample(self, audio_path_or_waveform, vad_list=None):
        """
        Get the sample from the dialog

        Returns dict containing:
            waveform,
            vad,
            vad_history
        """

        # Loads the dialog waveform (stereo) and normalize/to-mono for each
        ret = {}
        if isinstance(audio_path_or_waveform, str):
            ret["waveform"] = load_waveform(
                audio_path_or_waveform,
                sample_rate=self.sample_rate,
                normalize=True,
                mono=not self.stereo,
            )[0].unsqueeze(0)
            duration = get_audio_info(audio_path_or_waveform)["duration"]
        else:
            ret["waveform"] = audio_path_or_waveform
            duration = audio_path_or_waveform.shape[-1] / self.sample_rate

        if self.stereo:
            if ret["waveform"].ndim == 2:
                ret["waveform"] = ret["waveform"].unsqueeze(0)

            if ret["waveform"].shape[1] == 1:
                z = torch.zeros_like(ret["waveform"])
                ret["waveform"] = torch.cat((ret["waveform"], z), dim=1)

        if vad_list is not None:
            vad_hop_time = 1.0 / self.frame_hz
            vad_history_frames = (
                (torch.tensor(self.vad_history_times) / vad_hop_time).long().tolist()
            )

            ##############################################
            # VAD-frame of relevant part
            ##############################################
            end_frame = time_to_frames(duration, vad_hop_time)
            all_vad_frames = vad_list_to_onehot(
                vad_list,
                hop_time=vad_hop_time,
                duration=duration,
                channel_last=True,
            )
            lookahead = torch.zeros((self.horizon_frames + 1, 2))
            all_vad_frames = torch.cat((all_vad_frames, lookahead))
            ret["vad"] = all_vad_frames[: end_frame + self.horizon_frames].unsqueeze(0)

            ##############################################
            # History
            ##############################################
            vad_history, _ = get_activity_history(
                all_vad_frames,
                bin_end_frames=vad_history_frames,
                channel_last=True,
            )
            # vad history is always defined as speaker 0 activity
            ret["vad_history"] = vad_history[:end_frame][..., 0].unsqueeze(0)
        return ret

    @torch.no_grad()
    def output(self, batch, reduction="none", out_device="cpu"):
        loss, out, batch = self.shared_step(
            batch_to_device(batch, str(self.device)), reduction=reduction
        )
        probs = self.VAP(logits=out["logits"], va=batch["vad"])
        batch = batch_to_device(batch, out_device)
        out = batch_to_device(out, out_device)
        loss = batch_to_device(loss, out_device)
        probs = batch_to_device(probs, out_device)
        return loss, out, probs, batch

    def get_event_max_frames(self, batch):
        total_frames = batch["vad"].shape[1]
        return total_frames - self.VAP.horizon_frames

    def training_step(self, batch, batch_idx, **kwargs):
        loss, _, _ = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]
        self.log("loss", loss["loss"], batch_size=batch_size, sync_dist=True)
        return {"loss": loss["loss"]}

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""

        # extract events for metrics (use full vad including horizon)
        max_event_frame = self.get_event_max_frames(batch)
        events = self.val_metric.extract_events(
            va=batch["vad"], max_frame=max_event_frame
        )

        # Regular forward pass
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        # log scores
        self.log("val_loss", loss["loss"], batch_size=batch_size, sync_dist=True)

        # Extract other metrics
        turn_taking_probs = self.VAP(logits=out["logits"], va=batch["vad"])
        self.val_metric.update(
            p=turn_taking_probs["p"],
            bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
            events=events,
        )

    def test_step(self, batch, batch_idx, **kwargs):
        max_event_frame = self.get_event_max_frames(batch)
        events = self.test_metric.extract_events(
            va=batch["vad"], max_frame=max_event_frame
        )

        # Regular forward pass
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        # log scores
        self.log("test_loss", loss["loss"], batch_size=batch_size, sync_dist=True)

        # Extract other metrics
        turn_taking_probs = self.VAP(logits=out["logits"], va=batch["vad"])
        self.test_metric.update(
            p=turn_taking_probs["p"],
            bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
            events=events,
        )
