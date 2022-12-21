import torch
from torch import Tensor
from typing import Dict, List, Tuple

from vap.objective import ObjectiveVAP
from vap.events import get_dialog_states


def end_of_segment_mono(n: int, max: int = 3) -> Tensor:
    """
    # 0, 0, 0, 0
    # 1, 0, 0, 0
    # 1, 1, 0, 0
    # 1, 1, 1, 0
    """
    v = torch.zeros((max + 1, n))
    for i in range(max):
        v[i + 1, : i + 1] = 1
    return v


def all_permutations_mono(n: int, start: int = 0) -> Tensor:
    vectors = []
    for i in range(start, 2 ** n):
        i = bin(i).replace("0b", "").zfill(n)
        tmp = torch.zeros(n)
        for j, val in enumerate(i):
            tmp[j] = float(val)
        vectors.append(tmp)
    return torch.stack(vectors)


def on_activity_change_mono(n: int = 4, min_active: int = 2) -> Tensor:
    """

    Used where a single speaker is active. This vector (single speaker) represents
    the classes we use to infer that the current speaker will end their activity
    and the other take over.

    the `min_active` variable corresponds to the minimum amount of frames that must
    be active AT THE END of the projection window (for the next active speaker).
    This used to not include classes where the activity may correspond to a short backchannel.
    e.g. if only the last bin is active it may be part of just a short backchannel, if we require 2 bins to
    be active we know that the model predicts that the continuation will be at least 2 bins long and thus
    removes the ambiguouty (to some extent) about the prediction.
    """

    base = torch.zeros(n)
    # force activity at the end
    if min_active > 0:
        base[-min_active:] = 1

    # get all permutations for the remaining bins
    permutable = n - min_active
    if permutable > 0:
        perms = all_permutations_mono(permutable)
        base = base.repeat(perms.shape[0], 1)
        base[:, :permutable] = perms
    return base


def combine_speakers(x1: Tensor, x2: Tensor, mirror: bool = False) -> Tensor:
    if x1.ndim == 1:
        x1 = x1.unsqueeze(0)
    if x2.ndim == 1:
        x2 = x2.unsqueeze(0)
    vad = []
    for a in x1:
        for b in x2:
            vad.append(torch.stack((a, b), dim=0))

    vad = torch.stack(vad)
    if mirror:
        vad = torch.stack((vad, torch.stack((vad[:, 1], vad[:, 0]), dim=1)))
    return vad


def sort_idx(x: Tensor) -> Tensor:
    if x.ndim == 1:
        x, _ = x.sort()
    elif x.ndim == 2:
        if x.shape[0] == 2:
            a, _ = x[0].sort()
            b, _ = x[1].sort()
            x = torch.stack((a, b))
        else:
            x, _ = x[0].sort()
            x = x.unsqueeze(0)
    return x


# TODO: make the zero-shot from paper accessible
class ZeroShot(ObjectiveVAP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset_silence, self.subset_silence_hold = self.init_subset_silence()
        self.subset_active, self.subset_active_hold = self.init_subset_active()
        self.bc_prediction = self.init_subset_backchannel()

    def init_subset_silence(self) -> Tuple[Tensor, Tensor]:
        """
        During mutual silences we wish to infer which speaker the model deems most likely.

        We focus on classes where only a single speaker is active in the projection window,
        renormalize the probabilities on this subset, and determine which speaker is the most
        likely next speaker.
        """

        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        # active = self._all_permutations_mono(n, start=1)  # at least 1 active
        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        active = on_activity_change_mono(self.codebook.n_bins, min_active=2)
        # non-active channel: zeros
        non_active = torch.zeros((1, active.shape[-1]))
        # combine
        shift_oh = combine_speakers(active, non_active, mirror=True)
        shift = self.codebook.encode(shift_oh)
        shift = sort_idx(shift)

        # symmetric, this is strictly unneccessary but done for convenience and to be similar
        # to 'get_on_activity_shift' where we actually have asymmetric classes for hold/shift
        hold = shift.flip(0)
        return shift, hold

    def init_subset_active(self) -> Tuple[Tensor, Tensor]:
        """On activity"""
        # Shift subset
        eos = end_of_segment_mono(self.codebook.n_bins, max=2)
        nav = on_activity_change_mono(self.codebook.n_bins, min_active=2)
        shift_oh = combine_speakers(nav, eos, mirror=True)
        shift = self.codebook.encode(shift_oh)
        shift = sort_idx(shift)

        # Don't shift subset
        eos = on_activity_change_mono(self.codebook.n_bins, min_active=2)
        zero = torch.zeros((1, self.codebook.n_bins))
        hold_oh = combine_speakers(zero, eos, mirror=True)
        hold = self.codebook.encode(hold_oh)
        hold = sort_idx(hold)
        return shift, hold

    def init_subset_backchannel(self, n: int = 4) -> Tensor:
        if n != 4:
            raise NotImplementedError("Not implemented for bin-size != 4")

        # at least 1 bin active over 3 bins
        bc_speaker = all_permutations_mono(n=3, start=1)
        bc_speaker = torch.cat(
            (bc_speaker, torch.zeros((bc_speaker.shape[0], 1))), dim=-1
        )

        # all permutations of 3 bins
        current = all_permutations_mono(n=3, start=0)
        current = torch.cat((current, torch.ones((current.shape[0], 1))), dim=-1)

        bc_both = combine_speakers(bc_speaker, current, mirror=True)
        return self.codebook.encode(bc_both)

    def marginal_probs(self, probs: Tensor, pos_idx: Tensor, neg_idx: Tensor) -> Tensor:
        p = []
        for next_speaker in [0, 1]:
            joint = torch.cat((pos_idx[next_speaker], neg_idx[next_speaker]), dim=-1)
            p_sum = probs[..., joint].sum(dim=-1)
            p.append(probs[..., pos_idx[next_speaker]].sum(dim=-1) / p_sum)
        return torch.stack(p, dim=-1)

    def probs_on_silence(self, probs: Tensor) -> Tensor:
        return self.marginal_probs(probs, self.subset_silence, self.subset_silence_hold)

    def probs_on_active(self, probs: Tensor) -> Tensor:
        return self.marginal_probs(probs, self.subset_active, self.subset_active_hold)

    def probs_backchannel(self, probs: Tensor) -> Tensor:
        ap = probs[..., self.bc_prediction[0]].sum(-1)
        bp = probs[..., self.bc_prediction[1]].sum(-1)
        return torch.stack((ap, bp), dim=-1)

    def silence_probs(
        self,
        p_a: Tensor,
        p_b: Tensor,
        sil_probs: Tensor,
        silence: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        w = torch.where(silence)
        p_a[w] = sil_probs[w][..., 0]
        p_b[w] = sil_probs[w][..., 1]
        return p_a, p_b

    def single_speaker_probs(
        self,
        p0: Tensor,
        p1: Tensor,
        act_probs: Tensor,
        current: Tensor,
        other_speaker: int,
    ) -> Tuple[Tensor, Tensor]:
        w = torch.where(current)
        p0[w] = 1 - act_probs[w][..., other_speaker]  # P_a = 1-P_b
        p1[w] = act_probs[w][..., other_speaker]  # P_b
        return p0, p1

    def overlap_probs(
        self, p_a: Tensor, p_b: Tensor, act_probs: Tensor, both: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        P_a_prior=A is next (active)
        P_b_prior=B is next (active)
        We the compare/renormalize given the two values of A/B is the next speaker
        sum = P_a_prior+P_b_prior
        P_a = P_a_prior / sum
        P_b = P_b_prior / sum
        """
        w = torch.where(both)
        # Re-Normalize and compare next-active
        sum = act_probs[w][..., 0] + act_probs[w][..., 1]

        p_a[w] = act_probs[w][..., 0] / sum
        p_b[w] = act_probs[w][..., 1] / sum
        return p_a, p_b

    def probs_next_speaker(self, probs: Tensor, va: Tensor) -> Tensor:
        """
        Extracts the probabilities for who the next speaker is dependent on what the current moment is.

        This means that on mutual silences we use the 'silence'-subset,
        where a single speaker is active we use the 'active'-subset and where on overlaps
        """
        sil_probs = self.probs_on_silence(probs)
        act_probs = self.probs_on_active(probs)

        # Start wit all zeros
        # p_a: probability of A being next speaker (channel: 0)
        # p_b: probability of B being next speaker (channel: 1)
        p_a = torch.zeros_like(va[..., 0])
        p_b = torch.zeros_like(va[..., 0])

        # dialog states
        ds = get_dialog_states(va)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        both = ds == 2

        # silence
        p_a, p_b = self.silence_probs(p_a, p_b, sil_probs, silence)

        # A current speaker
        # Given only A is speaking we use the 'active' probability of B being the next speaker
        p_a, p_b = self.single_speaker_probs(
            p_a, p_b, act_probs, a_current, other_speaker=1
        )

        # B current speaker
        # Given only B is speaking we use the 'active' probability of A being the next speaker
        p_b, p_a = self.single_speaker_probs(
            p_b, p_a, act_probs, b_current, other_speaker=0
        )

        # Both
        p_a, p_b = self.overlap_probs(p_a, p_b, act_probs, both)

        p_next_speaker = torch.stack((p_a, p_b), dim=-1)
        return p_next_speaker

    def get_probs(self, logits: Tensor, va: Tensor) -> Dict[str, Tensor]:
        probs = logits.softmax(-1)
        nmax = probs.shape[-2]
        p = self.probs_next_speaker(probs, va[:, :nmax])
        p_bc = self.probs_backchannel(probs)
        return {"p": p, "p_bc": p_bc}

    @torch.no_grad()
    def extract_prediction_and_targets(
        self,
        p: Tensor,
        p_bc: Tensor,
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
                pshift = p[b, start:end, speaker]
                preds["hs"].append(pshift)
                targets["hs"].append(torch.ones_like(pshift))
            for start, end, speaker in events["hold"][b]:
                phold = 1 - p[b, start:end, speaker]
                preds["hs"].append(phold)
                targets["hs"].append(torch.zeros_like(phold))
            ###########################################
            # Shift-prediction
            ###########################################
            for start, end, speaker in events["pred_shift"][b]:
                # prob of next speaker -> the correct next speaker i.e. a SHIFT
                pshift = p[b, start:end, speaker]
                preds["pred_shift"].append(pshift)
                targets["pred_shift"].append(torch.ones_like(pshift))
            for start, end, speaker in events["pred_shift_neg"][b]:
                # prob of next speaker -> the correct next speaker i.e. a HOLD
                phold = 1 - p[b, start:end, speaker]  # 1-shift = Hold
                preds["pred_shift"].append(phold)
                # Negatives are zero -> hold predictions
                targets["pred_shift"].append(torch.zeros_like(phold))
            ###########################################
            # Backchannel-prediction
            ###########################################
            for start, end, speaker in events["pred_backchannel"][b]:
                # prob of next speaker -> the correct next backchanneler i.e. a Backchannel
                pred_bc = p_bc[b, start:end, speaker]
                preds["pred_backchannel"].append(pred_bc)
                targets["pred_backchannel"].append(torch.ones_like(pred_bc))
            for start, end, speaker in events["pred_backchannel_neg"][b]:
                # prob of 'speaker' making a 'backchannel' in the close future
                # over these negatives this probability should be low -> 0
                # so no change of probability have to be made (only the labels are now zero)
                pred_bc = p_bc[b, start:end, speaker]  # 1-shift = Hold
                preds["pred_backchannel"].append(
                    pred_bc
                )  # Negatives are zero -> hold predictions
                targets["pred_backchannel"].append(torch.zeros_like(pred_bc))
            ###########################################
            # Long vs Shoft
            ###########################################
            # TODO: Should this be the same as backchannel
            # or simply next speaker probs?
            for start, end, speaker in events["long"][b]:
                # prob of next speaker -> the correct next speaker i.e. a LONG
                plong = p[b, start:end, speaker]
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
                pshort = p[b, start:end, speaker]  # 1-shift = Hold
                preds["ls"].append(pshort)
                # Negatives are zero -> short predictions
                targets["ls"].append(torch.zeros_like(pshort))

        # cat/stack/flatten to single tensor
        device = device if device is not None else p.device
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
    zs = ZeroShot()
