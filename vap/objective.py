import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from typing import Dict, List, Tuple


def bin_times_to_frames(bin_times: List[float], frame_hz: int) -> List[int]:
    return (torch.tensor(bin_times) * frame_hz).long().tolist()


class ProjectionWindow:
    def __init__(
        self,
        bin_times: List = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.bin_times = bin_times
        self.frame_hz = frame_hz
        self.threshold_ratio = threshold_ratio

        self.bin_frames = bin_times_to_frames(bin_times, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.total_bins = self.n_bins * 2
        self.horizon = sum(self.bin_frames)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  bin_times: {self.bin_times}\n"
        s += f"  bin_frames: {self.bin_frames}\n"
        s += f"  frame_hz: {self.frame_hz}\n"
        s += f"  thresh: {self.threshold_ratio}\n"
        s += ")\n"
        return s

    def projection(self, va: Tensor) -> Tensor:
        """
        Extract projection (bins)
        (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames

        Arguments:
            va:         Tensor (B, N, C)

        Returns:
            vaps:       Tensor (B, m, C, M)

        """
        # Shift to get next frame projections
        return va[..., 1:, :].unfold(dimension=-2, size=sum(self.bin_frames), step=1)

    def projection_bins(self, projection_window: Tensor) -> Tensor:
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """

        start = 0
        v_bins = []
        for b in self.bin_frames:
            end = start + b
            m = projection_window[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        return torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)

    def __call__(self, va: Tensor) -> Tensor:
        projection_windows = self.projection(va)
        return self.projection_bins(projection_windows)


class Codebook(nn.Module):
    def __init__(self, bin_frames):
        super().__init__()
        self.bin_frames = bin_frames
        self.n_bins: int = len(self.bin_frames)
        self.total_bins: int = self.n_bins * 2
        self.n_classes: int = 2 ** self.total_bins

        self.emb = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.total_bins
        )
        self.emb.weight.data = self.create_code_vectors(self.total_bins)
        self.emb.weight.requires_grad_(False)

    def single_idx_to_onehot(self, idx: int, d: int = 8) -> Tensor:
        assert idx < 2 ** d, "must be possible with {d} binary digits"
        z = torch.zeros(d)
        b = bin(idx).replace("0b", "")
        for i, v in enumerate(b[::-1]):
            z[i] = float(v)
        return z

    def create_code_vectors(self, n_bins: int) -> Tensor:
        """
        Create a matrix of all one-hot encodings representing a binary sequence of `self.total_bins` places
        Useful for usage in `nn.Embedding` like module.
        """
        n_codes = 2 ** n_bins
        embs = torch.zeros((n_codes, n_bins))
        for i in range(2 ** n_bins):
            embs[i] = self.single_idx_to_onehot(i, d=n_bins)
        return embs

    def encode(self, x: Tensor) -> Tensor:
        """

        Encodes projection_windows x (*, 2, 4) to indices in codebook (..., 1)

        Arguments:
            x:          Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (
            2,
            self.n_bins,
        ), f"Codebook expects (..., 2, {self.n_bins}) got {x.shape}"

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = rearrange(x, "... c bpp -> (...) (c bpp)", c=2, bpp=self.n_bins)
        embed = self.emb.weight.T
        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-2])
        return embed_ind

    def decode(self, idx: Tensor):
        v = self.emb(idx)
        return rearrange(v, "... (c b) -> ... c b", c=2)

    def forward(self, projection_windows: Tensor):
        return self.encode(projection_windows)


class ObjectiveVAP(nn.Module):
    def __init__(
        self,
        bin_times: List[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_times = bin_times
        self.bin_frames: List[int] = bin_times_to_frames(bin_times, frame_hz)
        self.horizon = sum(self.bin_frames)
        self.horizon_time = sum(bin_times)

        self.codebook = Codebook(self.bin_frames)
        self.projection_window_extractor = ProjectionWindow(
            bin_times, frame_hz, threshold_ratio
        )
        self.requires_grad_(False)

    def __repr__(self):
        s = str(self.__class__.__name__)
        s += f"\n{self.codebook}"
        s += f"\n{self.projection_window_extractor}"
        s += "\n"
        return s

    @property
    def n_classes(self) -> int:
        return self.codebook.n_classes

    @property
    def n_bins(self) -> int:
        return self.codebook.n_bins

    def probs_next_speaker_aggregate(
        self,
        probs: Tensor,
        from_bin: int = 0,
        to_bin: int = 3,
        scale_with_bins: bool = False,
    ) -> Tensor:
        assert (
            probs.ndim == 3
        ), f"Expected probs of shape (B, n_frames, n_classes) but got {probs.shape}"
        idx = torch.arange(self.codebook.n_classes).to(probs.device)
        states = self.codebook.decode(idx)

        if scale_with_bins:
            states = states * torch.tensor(self.bin_frames)
        abp = states[:, :, from_bin : to_bin + 1].sum(-1)  # sum speaker activity bins
        # Dot product over all states
        p_all = torch.einsum("bid,dc->bic", probs, abp)
        # normalize
        p_all /= p_all.sum(-1, keepdim=True) + 1e-5
        return p_all

    def window_to_win_dialog_states(self, wins):
        return (wins.sum(-1) > 0).sum(-1)

    def get_labels(self, va: Tensor, ds_label=False) -> Tuple[Tensor, Tensor]:
        projection_windows = self.projection_window_extractor(va)
        idx = self.codebook(projection_windows)
        if ds_label:
            ds = self.window_to_win_dialog_states(projection_windows)
            return idx, ds
        return idx

    def loss_vap(
        self, logits: Tensor, labels: Tensor, reduction: str = "mean"
    ) -> Tensor:
        assert (
            logits.ndim == 3
        ), f"Exptected logits of shape (B, N_FRAMES, N_CLASSES) but got {logits.shape}"
        assert (
            labels.ndim == 2
        ), f"Exptected labels of shape (B, N_FRAMES) but got {labels.shape}"

        nmax = labels.shape[1]
        if logits.shape[1] > nmax:
            logits = logits[:, :nmax]

        # CrossEntropyLoss over discrete labels
        loss = F.cross_entropy(
            rearrange(logits, "b n d -> (b n) d"),
            rearrange(labels, "b n -> (b n)"),
            reduction=reduction,
        )
        # Shape back to original shape if reduction != 'none'
        if reduction == "none":
            loss = rearrange(loss, "(b n) -> b n", n=nmax)
        return loss

    def loss_vad(self, vad_output, vad):
        n = vad_output.shape[-2]
        return F.binary_cross_entropy_with_logits(vad_output, vad[:, :n])

    def get_probs(self, logits: Tensor) -> Dict[str, Tensor]:
        """
        Extracts labels from the voice-activity, va.
        The labels are based on projections of the future and so the valid
        frames with corresponding labels are strictly less then the original number of frams.

        Arguments:
        -----------
        logits:     torch.Tensor (B, N_FRAMES, N_CLASSES)
        va:         torch.Tensor (B, N_FRAMES, 2)

        Return:
        -----------
            Dict[probs, p, p_bc, labels]  which are all torch.Tensors
        """

        assert (
            logits.shape[-1] == self.n_classes
        ), f"Logits have wrong shape. {logits.shape} != (..., {self.n_classes}) that is (B, N_FRAMES, N_CLASSES)"

        probs = logits.softmax(dim=-1)
        return {
            "probs": probs,
            "p_now": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=0, to_bin=1
            ),
            "p_future": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=2, to_bin=3
            ),
            "p_tot": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=0, to_bin=3
            ),
        }

    @torch.no_grad()
    def extract_prediction_and_targets(
        self,
        p_now: Tensor,
        p_fut: Tensor,
        events: Dict[str, List[List[Tuple[int, int, int]]]],
        device=None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        batch_size = len(events["hold"])

        preds = {"hs": [], "pred_shift": [], "ls": [], "pred_backchannel": []}
        targets = {"hs": [], "pred_shift": [], "ls": [], "pred_backchannel": []}

        for b in range(batch_size):
            ###########################################
            # Hold vs Shift
            ###########################################
            # The metrics (i.e. shift/hold) are binary so we must decide
            # which 'class' corresponds to which numeric label
            # we use Holds=0, Shifts=1
            for start, end, speaker in events["shift"][b]:
                pshift = p_now[b, start:end, speaker]
                preds["hs"].append(pshift)
                targets["hs"].append(torch.ones_like(pshift))

            for start, end, speaker in events["hold"][b]:
                phold = 1 - p_now[b, start:end, speaker]
                preds["hs"].append(phold)
                targets["hs"].append(torch.zeros_like(phold))
            ###########################################
            # Shift-prediction
            ###########################################
            for start, end, speaker in events["pred_shift"][b]:
                # prob of next speaker -> the correct next speaker i.e. a SHIFT
                pshift = p_fut[b, start:end, speaker]
                preds["pred_shift"].append(pshift)
                targets["pred_shift"].append(torch.ones_like(pshift))
            for start, end, speaker in events["pred_shift_neg"][b]:
                # prob of next speaker -> the correct next speaker i.e. a HOLD
                phold = 1 - p_fut[b, start:end, speaker]  # 1-shift = Hold
                preds["pred_shift"].append(phold)
                # Negatives are zero -> hold predictions
                targets["pred_shift"].append(torch.zeros_like(phold))
            ###########################################
            # Backchannel-prediction
            ###########################################
            # TODO: Backchannel with p_now/p_fut???
            # for start, end, speaker in events["pred_backchannel"][b]:
            #     # prob of next speaker -> the correct next backchanneler i.e. a Backchannel
            #     pred_bc = p_bc[b, start:end, speaker]
            #     preds["pred_backchannel"].append(pred_bc)
            #     targets["pred_backchannel"].append(torch.ones_like(pred_bc))
            # for start, end, speaker in events["pred_backchannel_neg"][b]:
            #     # prob of 'speaker' making a 'backchannel' in the close future
            #     # over these negatives this probability should be low -> 0
            #     # so no change of probability have to be made (only the labels are now zero)
            #     pred_bc = p_bc[b, start:end, speaker]  # 1-shift = Hold
            #     preds["pred_backchannel"].append(
            #         pred_bc
            #     )  # Negatives are zero -> hold predictions
            #     targets["pred_backchannel"].append(torch.zeros_like(pred_bc))
            ###########################################
            # Long vs Shoft
            ###########################################
            # TODO: Should this be the same as backchannel
            # or simply next speaker probs?
            for start, end, speaker in events["long"][b]:
                # prob of next speaker -> the correct next speaker i.e. a LONG
                plong = p_fut[b, start:end, speaker]
                preds["ls"].append(plong)
                targets["ls"].append(torch.ones_like(plong))
            for start, end, speaker in events["short"][b]:
                # the speaker in the 'short' events is the speaker who
                # utters a short utterance: p[b, start:end, speaker] means:
                # the  speaker saying something short has this probability
                # of continue as a 'long'
                # Therefore to correctly predict a 'short' entry this probability
                # should be low -> 0
                # thus we do not have to subtract the prob from 1 (only the labels are now zero)
                # prob of next speaker -> the correct next speaker i.e. a SHORT
                pshort = p_fut[b, start:end, speaker]  # 1-shift = Hold
                preds["ls"].append(pshort)
                # Negatives are zero -> short predictions
                targets["ls"].append(torch.zeros_like(pshort))

        # cat/stack/flatten to single tensor
        device = device if device is not None else p_now.device
        out_preds = {}
        out_targets = {}
        for k, v in preds.items():
            if len(v) > 0:
                out_preds[k] = torch.cat(v).to(device)
            else:
                out_preds[k] = None
        for k, v in targets.items():
            if len(v) > 0:
                out_targets[k] = torch.cat(v).long().to(device)
            else:
                out_targets[k] = None
        return out_preds, out_targets


if __name__ == "__main__":
    ob = ObjectiveVAP()
    print(ob)
