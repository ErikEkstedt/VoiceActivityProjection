import torch
from torch import Tensor
from dataclasses import dataclass
import random
from typing import Dict, Optional, Tuple, List
from vap.utils import find_island_idx_len

# Templates
TRIAD_SHIFT: Tensor = torch.tensor([[3, 1, 0], [0, 1, 3]])  # on Silence
TRIAD_SHIFT_OVERLAP: Tensor = torch.tensor([[3, 2, 0], [0, 2, 3]])
TRIAD_HOLD: Tensor = torch.tensor([[0, 1, 0], [3, 1, 3]])  # on silence
TRIAD_BC: Tensor = torch.tensor([0, 1, 0])

# Dialog states meaning
STATE_ONLY_A: int = 0
STATE_ONLY_B: int = 3
STATE_SILENCE: int = 1
STATE_BOTH: int = 2


@dataclass
class EventConfig:
    min_context_time: float = 3
    metric_time: float = 0.2
    metric_pad_time: float = 0.05
    max_time: int = 20
    frame_hz: int = 50
    equal_hold_shift: bool = True
    prediction_region_time: float = 0.5

    # Shift/Hold
    sh_pre_cond_time: float = 1.0
    sh_post_cond_time: float = 1.0
    sh_prediction_region_on_active: bool = True

    # Backchannel
    bc_pre_cond_time: float = 1.0
    bc_post_cond_time: float = 1.0
    bc_max_duration: float = 1.0
    bc_negative_pad_left_time: float = 1.0
    bc_negative_pad_right_time: float = 2.0

    # Long/Short
    long_onset_region_time: float = 0.2
    long_onset_condition_time: float = 1.0


def time_to_frames(time: float, frame_hz: int) -> int:
    frame = int(time * frame_hz)
    return frame


def get_dialog_states(vad: Tensor) -> Tensor:
    """Vad to the full state of a 2 person vad dialog
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    assert vad.ndim >= 1
    return (2 * vad[..., 1] - vad[..., 0]).long() + 1


def fill_pauses(
    vad: Tensor,
    ds: Tensor,
    islands: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tensor:
    assert vad.ndim == 2, "fill_pauses require ds=(n_frames, 2)"
    assert ds.ndim == 1, "fill_pauses require ds=(n_frames,)"

    filled_vad = vad.clone()

    if islands is None:
        s, d, v = find_island_idx_len(ds)
    else:
        s, d, v = islands

    # less than three entries means that there are no pauses
    # requires at least: activity-from-speaker  ->  Silence   --> activity-from-speaker
    if len(v) < 3:
        return vad

    triads = v.unfold(0, size=3, step=1)
    next_speaker, steps = torch.where(
        (triads == TRIAD_HOLD.unsqueeze(1).to(triads.device)).sum(-1) == 3
    )
    for ns, pre in zip(next_speaker, steps):
        cur = pre + 1
        # Fill the matching template
        filled_vad[s[cur] : s[cur] + d[cur], ns] = 1.0
    return filled_vad


def get_hs_regions(
    triads: Tensor,
    filled_vad: Tensor,
    triad_label: Tensor,
    start_of: Tensor,
    duration_of: Tensor,
    pre_cond_frames: int,
    post_cond_frames: int,
    prediction_region_frames: int,
    prediction_region_on_active: bool,
    long_onset_condition_frames: int,
    long_onset_region_frames: int,
    min_silence_frames: int,
    min_context_frames: int,
    max_frame: int,
) -> Tuple[
    List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]
]:
    """
    get regions defined by `triad_label`
    """

    region = []
    prediction_region = []
    long_onset_region = []

    # check if label is hold or shift
    # if the same speaker continues after silence -> hold
    hold_cond = triad_label[0, 0] == triad_label[0, -1]
    next_speakers, steps = torch.where(
        (triads == triad_label.unsqueeze(1)).sum(-1) == 3
    )
    # No matches -> return
    if len(next_speakers) == 0:
        return [], [], []

    for last_onset, next_speaker in zip(steps, next_speakers):
        not_next_speaker = int(not next_speaker)
        prev_speaker = next_speaker if hold_cond else not_next_speaker
        not_prev_speaker = 0 if prev_speaker == 1 else 1
        # All shift triads e.g. [3, 1, 0] centers on the silence segment
        # If we find a triad match at step 's' then the actual SILENCE segment
        # STARTS:           on the next step -> add 1
        # ENDS/next-onset:  on the two next step -> add 2
        silence = last_onset + 1
        next_onset = last_onset + 2
        ################################################
        # MINIMAL CONTEXT CONDITION
        ################################################
        if start_of[silence] < min_context_frames:
            continue
        ################################################
        # MAXIMAL FRAME CONDITION
        ################################################
        if start_of[silence] >= max_frame:
            continue
        ################################################
        # MINIMAL SILENCE CONDITION
        ################################################
        # Check silence duration
        if duration_of[silence] < min_silence_frames:
            continue
        ################################################
        # PRE CONDITION: ONLY A SINGLE PREVIOUS SPEAKER
        ################################################
        # Check `pre_cond_frames` before start of silence
        # to make sure only a single speaker was active
        sil_start = start_of[silence]
        pre_start = sil_start - pre_cond_frames
        pre_start = pre_start if pre_start > 0 else 0
        correct_is_active = (
            filled_vad[pre_start:sil_start, prev_speaker].sum() == pre_cond_frames
        )
        if not correct_is_active:
            continue
        other_is_silent = filled_vad[pre_start:sil_start, not_prev_speaker].sum() == 0
        if not other_is_silent:
            continue
        ################################################
        # POST CONDITION: ONLY A SINGLE PREVIOUS SPEAKER
        ################################################
        # Check `post_cond_frames` after start of onset
        # to make sure only a single speaker is to be active
        onset_start = start_of[next_onset]
        onset_region_end = onset_start + post_cond_frames
        correct_is_active = (
            filled_vad[onset_start:onset_region_end, next_speaker].sum()
            == post_cond_frames
        )
        if not correct_is_active:
            continue
        other_is_silent = (
            filled_vad[onset_start:onset_region_end, not_next_speaker].sum() == 0
        )
        if not other_is_silent:
            continue
        ################################################
        # ALL CONDITIONS MET
        ################################################
        region.append((sil_start.item(), onset_start.item(), next_speaker.item()))

        ################################################
        # LONG ONSET CONDITION
        ################################################
        # if we have a valid shift we check if the onset
        # of the next segment is longer than `long_onset_condition_frames`
        # and if true we add the region
        if not hold_cond and duration_of[next_onset] >= long_onset_condition_frames:
            # We add the 'long-onset' region defined by `long_onset_region_frames`
            # the condition is used to define "yea, this is an onset of a 'long' region"
            # whereas the `long_onset_region_frames` define the area in which we wish
            # to make predictions with the model.
            long_onset_region.append(
                (
                    onset_start.item(),
                    (onset_start + long_onset_region_frames).item(),
                    next_speaker.item(),
                )
            )

        ################################################
        # PREDICTION REGION CONDITION
        ################################################
        # The prediction region is defined at the end of the previous
        # activity, not inside the silences.

        # IF PREDICTION_REGION_ON_ACTIVE = FALSE
        # We don't care about the previous activity but only take
        # `prediction_region_frames` prior to the relevant hold/shift silence.
        # e.g. if prediction_region_frames=100 and the last segment prior to the
        # relevant hold/shift silence was 70 frames the prediction region would include
        # < 30 frames of silence (a pause or a shift (could be a quick back and forth limited by the condition variables...))
        if prediction_region_on_active:
            # We make sure that the last VAD segments
            # of the last speaker is longer than
            # `prediction_region_frames`
            if duration_of[last_onset] < prediction_region_frames:
                continue

        # that if the last activity
        prediction_start = sil_start - prediction_region_frames

        ################################################
        # MINIMAL CONTEXT CONDITION (PREDICTION)
        ################################################
        if prediction_start < min_context_frames:
            continue

        prediction_region.append(
            (prediction_start.item(), sil_start.item(), next_speaker.item())
        )

    return region, prediction_region, long_onset_region


def hold_shift_regions(
    vad: Tensor,
    ds: Tensor,
    pre_cond_frames: int,
    post_cond_frames: int,
    prediction_region_frames: int,
    prediction_region_on_active: bool,
    long_onset_condition_frames: int,
    long_onset_region_frames: int,
    min_silence_frames: int,
    min_context_frames: int,
    max_frame: int,
) -> Dict[str, List[Tuple[int, int, int]]]:
    assert vad.ndim == 2, f"expects vad of shape (n_frames, 2) but got {vad.shape}."

    start_of, duration_of, states = find_island_idx_len(ds)
    filled_vad = fill_pauses(vad, ds, islands=(start_of, duration_of, states))

    # If we have less than 3 unique dialog states
    # then we have no valid transitions
    if len(states) < 3:
        return {"shift": [], "hold": [], "long": [], "pred_shift": [], "pred_hold": []}

    triads = states.unfold(0, size=3, step=1)

    # SHIFTS
    shifts, pred_shifts, long_onset = get_hs_regions(
        triads=triads,
        filled_vad=filled_vad,
        triad_label=TRIAD_SHIFT.to(vad.device),
        start_of=start_of,
        duration_of=duration_of,
        pre_cond_frames=pre_cond_frames,
        post_cond_frames=post_cond_frames,
        prediction_region_frames=prediction_region_frames,
        prediction_region_on_active=prediction_region_on_active,
        long_onset_condition_frames=long_onset_condition_frames,
        long_onset_region_frames=long_onset_region_frames,
        min_silence_frames=min_silence_frames,
        min_context_frames=min_context_frames,
        max_frame=max_frame,
    )

    # HOLDS
    holds, pred_holds, _ = get_hs_regions(
        triads=triads,
        filled_vad=filled_vad,
        triad_label=TRIAD_HOLD.to(vad.device),
        start_of=start_of,
        duration_of=duration_of,
        pre_cond_frames=pre_cond_frames,
        post_cond_frames=post_cond_frames,
        prediction_region_frames=prediction_region_frames,
        prediction_region_on_active=prediction_region_on_active,
        long_onset_condition_frames=long_onset_condition_frames,
        long_onset_region_frames=long_onset_region_frames,
        min_silence_frames=min_silence_frames,
        min_context_frames=min_context_frames,
        max_frame=max_frame,
    )
    return {
        "shift": shifts,
        "hold": holds,
        "long": long_onset,
        "pred_shift": pred_shifts,
        "pred_hold": pred_holds,
    }


def backchannel_regions(
    vad: Tensor,
    ds: Tensor,
    pre_cond_frames: int,
    post_cond_frames: int,
    prediction_region_frames: int,
    min_context_frames: int,
    max_bc_frames: int,
    max_frame: int,
) -> Dict[str, List[Tuple[int, int, int]]]:
    assert vad.ndim == 2, f"expects vad of shape (n_frames, 2) but got {vad.shape}."

    filled_vad = fill_pauses(vad, ds)

    backchannel = []
    pred_backchannel = []
    for speaker in [0, 1]:
        start_of, duration_of, states = find_island_idx_len(filled_vad[..., speaker])
        if len(states) < 3:
            continue
        triads = states.unfold(0, size=3, step=1)
        steps = torch.where(
            (triads == TRIAD_BC.to(triads.device).unsqueeze(0)).sum(-1) == 3
        )[0]
        if len(steps) == 0:
            continue
        for pre_silence in steps:
            bc = pre_silence + 1
            post_silence = pre_silence + 2
            ################################################
            # MINIMAL CONTEXT CONDITION
            ################################################
            if start_of[bc] < min_context_frames:
                # print("Minimal context")
                continue
            ################################################
            # MAXIMAL FRAME CONDITION
            ################################################
            if start_of[bc] >= max_frame:
                # print("Max frame")
                continue
            ################################################
            # MINIMAL DURATION CONDITION
            ################################################
            # Check bc duration
            if duration_of[bc] > max_bc_frames:
                # print("Too Long")
                continue
            ################################################
            # PRE CONDITION: No previous activity from bc-speaker
            ################################################
            if duration_of[pre_silence] < pre_cond_frames:
                # print('not enough silence PRIOR to "bc"')
                continue
            ################################################
            # POST CONDITION: No post activity from bc-speaker
            ################################################
            if duration_of[post_silence] < post_cond_frames:
                # print('not enough silence POST to "bc"')
                continue
            ################################################
            # ALL CONDITIONS MET
            ################################################
            # Is the other speakr active before this segment?
            backchannel.append(
                (start_of[bc].item(), start_of[post_silence].item(), speaker)
            )

            pred_bc_start = start_of[bc] - prediction_region_frames
            if pred_bc_start < min_context_frames:
                continue

            pred_backchannel.append(
                (pred_bc_start.item(), start_of[bc].item(), speaker)
            )

    return {"backchannel": backchannel, "pred_backchannel": pred_backchannel}


def get_negative_sample_regions(
    vad: Tensor,
    ds: Tensor,
    min_pad_left_frames: int,
    min_pad_right_frames: int,
    min_region_frames: int,
    min_context_frames: int,
    max_frame: int,
) -> List[Tuple[int, int, int]]:
    min_dur_frames = min_pad_left_frames + min_pad_right_frames

    # fill pauses o recognize 'longer' segments of activity (including pauses)
    filled_vad = fill_pauses(vad, ds)
    ds_fill = get_dialog_states(filled_vad)
    index_of, duration_of, state_of = find_island_idx_len(ds_fill)

    neg_regions = []
    for current_speaker, current_speaker_state in enumerate(
        [STATE_ONLY_A, STATE_ONLY_B]
    ):
        next_potential_speaker = int(not current_speaker)
        dur = duration_of[state_of == current_speaker_state]
        idx = index_of[state_of == current_speaker_state]

        # iterate over all segments of longer activity
        for i, d in zip(idx, dur):
            ################################################
            # MINIMAL CONTEXT CONDITION
            ################################################
            # The total activity must allow for padding
            if d < min_dur_frames:
                continue

            # START of region after `min_active_frames`
            start = (i + min_pad_left_frames).item()
            ################################################
            # CONTEXT (global/model) CONDITION
            ################################################
            # START of region must be after `min_context_frames`
            if start < min_context_frames:
                start = min_context_frames

            # END of region prior to `min_pad_to_next_frames`
            end = (i + d - min_pad_right_frames).item()

            ################################################
            # MAXIMAL FRAME
            ################################################
            # end region can't span across last valid frame
            if end > max_frame:
                end = max_frame

            ################################################
            # REGION SIZE
            ################################################
            # Is the final potential region larger than
            # the minimal required frames?
            # Also handles if end < start  (i.e. min_region_frames > 0)
            if end - start < min_region_frames:
                continue

            neg_regions.append((start, end, next_potential_speaker))

    return neg_regions


class HoldShiftNew:
    def __init__(
        self,
        pre_cond_time: float,
        post_cond_time: float,
        prediction_region_time: float,
        prediction_region_on_active: bool,
        long_onset_condition_time: float,
        long_onset_region_time: float,
        min_silence_time: float,
        min_context_time: float,
        max_time: float,
        frame_hz: int,
    ):
        # Time
        self.pre_cond_time = pre_cond_time
        self.post_cond_time = post_cond_time
        self.min_silence_time = min_silence_time
        self.min_context_time = min_context_time
        self.max_time = max_time
        self.frame_hz = frame_hz

        # Frames
        self.pre_cond_frame = time_to_frames(pre_cond_time, frame_hz)
        self.post_cond_frame = time_to_frames(post_cond_time, frame_hz)
        self.prediction_region_frame = time_to_frames(prediction_region_time, frame_hz)
        self.prediction_region_on_active = prediction_region_on_active
        self.long_onset_condition_frames = time_to_frames(
            long_onset_condition_time, frame_hz
        )
        self.long_onset_region_frames = time_to_frames(long_onset_region_time, frame_hz)
        self.prediction_region_on_active = prediction_region_on_active
        self.min_silence_frame = time_to_frames(min_silence_time, frame_hz)
        self.min_context_frame = time_to_frames(min_context_time, frame_hz)
        self.max_frame = time_to_frames(max_time, frame_hz)

    def __repr__(self) -> str:
        s = "HoldShift"
        s += "\n---------"
        s += f"\n  Time:"
        s += f"\n\tpre_cond_time     = {self.pre_cond_time}s"
        s += f"\n\tpost_cond_time    = {self.post_cond_time}s"
        s += f"\n\tmin_silence_time  = {self.min_silence_time}s"
        s += f"\n\tmin_context_time  = {self.min_context_time}s"
        s += f"\n\tmax_time          = {self.max_time}s"
        s += f"\n  Frame:"
        s += f"\n\tpre_cond_frame    = {self.pre_cond_frame}"
        s += f"\n\tpost_cond_frame   = {self.post_cond_frame}"
        s += f"\n\tmin_silence_frame = {self.min_silence_frame}"
        s += f"\n\tmin_context_frame = {self.min_context_frame}"
        s += f"\n\tmax_frame         = {self.max_frame}"
        return s

    @torch.no_grad()
    def __call__(
        self,
        vad: Tensor,
        ds: Optional[Tensor] = None,
        max_time: Optional[float] = None,
    ) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        assert (
            vad.ndim == 3
        ), f"Expected vad.ndim=3 (B, N_FRAMES, 2) but got {vad.shape}"

        max_frame = self.max_frame
        if max_time is not None:
            max_frame = time_to_frames(max_time, self.frame_hz)

        batch_size = vad.shape[0]

        if ds is None:
            ds = get_dialog_states(vad)

        shift, hold, long = [], [], []
        pred_shift, pred_hold = [], []
        for b in range(batch_size):
            tmp_sh = hold_shift_regions(
                vad=vad[b],
                ds=ds[b],
                pre_cond_frames=self.pre_cond_frame,
                post_cond_frames=self.post_cond_frame,
                prediction_region_frames=self.prediction_region_frame,
                prediction_region_on_active=self.prediction_region_on_active,
                long_onset_region_frames=self.long_onset_region_frames,
                long_onset_condition_frames=self.long_onset_condition_frames,
                min_silence_frames=self.min_silence_frame,
                min_context_frames=self.min_context_frame,
                max_frame=max_frame,
            )
            shift.append(tmp_sh["shift"])
            hold.append(tmp_sh["hold"])
            long.append(tmp_sh["long"])
            pred_shift.append(tmp_sh["pred_shift"])
            pred_hold.append(tmp_sh["pred_hold"])
        return {
            "shift": shift,
            "hold": hold,
            "long": long,
            "pred_shift": pred_shift,
            "pred_hold": pred_hold,
            "hold": hold,
        }


class BackchannelNew:
    def __init__(
        self,
        pre_cond_time: float,
        post_cond_time: float,
        prediction_region_time: float,
        min_context_time: float,
        negative_pad_left_time: float,
        negative_pad_right_time: float,
        max_bc_duration: float,
        max_time: float,
        frame_hz: int,
    ):
        self.pre_cond_time = pre_cond_time
        self.post_cond_time = post_cond_time
        self.max_bc_time = max_bc_duration
        self.prediction_region_time = prediction_region_time
        self.negatives_min_pad_left_time = negative_pad_left_time
        self.negatives_min_pad_right_time = negative_pad_right_time
        self.min_context_time = min_context_time
        self.frame_hz = frame_hz
        self.max_time = max_time

        assert (
            prediction_region_time > 0
        ), f"Requires positive value for `prediction_region_time` got {prediction_region_time}"

        assert (
            negative_pad_left_time + negative_pad_right_time < max_time
        ), f"Negative padding duration exceeds `max_time` {max_time}"

        self.pre_cond_frame = time_to_frames(pre_cond_time, frame_hz)
        self.post_cond_frame = time_to_frames(post_cond_time, frame_hz)
        self.prediction_region_frames = time_to_frames(prediction_region_time, frame_hz)
        self.negatives_min_pad_left_frames = time_to_frames(
            negative_pad_left_time, frame_hz
        )
        self.negatives_min_pad_right_frames = time_to_frames(
            negative_pad_right_time, frame_hz
        )

        self.min_context_frame = time_to_frames(min_context_time, frame_hz)
        self.max_bc_frame = time_to_frames(max_bc_duration, frame_hz)
        self.max_frame = time_to_frames(max_time, frame_hz)

    def __repr__(self) -> str:
        s = "Backhannel"
        s += "\n----------"
        s += f"\n  Time:"
        s += f"\n\tpre_cond_time              = {self.pre_cond_time}s"
        s += f"\n\tpost_cond_time             = {self.post_cond_time}s"
        s += f"\n\tmax_bc_time                = {self.max_bc_time}s"
        s += f"\n\tnegatives_left_pad_time    = {self.negatives_min_pad_left_time}s"
        s += f"\n\tnegatives_right_pad_time   = {self.negatives_min_pad_right_time}s"
        s += f"\n\tmin_context_time           = {self.min_context_time}s"
        s += f"\n\tmax_time                   = {self.max_time}s"
        s += f"\n  Frame:"
        s += f"\n\tpre_cond_frame             = {self.pre_cond_frame}"
        s += f"\n\tpost_cond_frame            = {self.post_cond_frame}"
        s += f"\n\tnegatives_left_pad_frames  = {self.negatives_min_pad_left_frames}"
        s += f"\n\tnegatives_right_pad_frames = {self.negatives_min_pad_right_frames}"
        s += f"\n\tprediction_region_frames   = {self.prediction_region_frames}"
        s += f"\n\tmax_bc_frame               = {self.max_bc_frame}"
        s += f"\n\tmin_context_frame          = {self.min_context_frame}"
        s += f"\n\tmax_frame                  = {self.max_frame}"
        return s

    def sample_negative_segment(
        self, region: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        region_start, region_end, speaker = region
        max_end = region_end - self.prediction_region_frames
        segment_start = random.randint(region_start, max_end)
        segment_end = segment_start + self.prediction_region_frames
        return (segment_start, segment_end, speaker)

    def __call__(
        self,
        vad: Tensor,
        ds: Optional[Tensor] = None,
        max_time: Optional[float] = None,
    ):
        batch_size = vad.shape[0]

        max_frame = self.max_frame
        if max_time is not None:
            max_frame = time_to_frames(max_time, self.max_frame)

        if ds is None:
            ds = get_dialog_states(vad)

        backchannel, pred_backchannel = [], []
        pred_backchannel_neg = []
        for b in range(batch_size):
            bc_samples = backchannel_regions(
                vad[b],
                ds=ds[b],
                pre_cond_frames=self.pre_cond_frame,
                post_cond_frames=self.post_cond_frame,
                min_context_frames=self.min_context_frame,
                prediction_region_frames=self.prediction_region_frames,
                max_bc_frames=self.max_bc_frame,
                max_frame=max_frame,
            )

            bc_negative_regions = get_negative_sample_regions(
                vad=vad[b],
                ds=ds[b],
                min_pad_left_frames=self.negatives_min_pad_left_frames,
                min_pad_right_frames=self.negatives_min_pad_right_frames,
                min_region_frames=self.prediction_region_frames,
                min_context_frames=self.min_context_frame,
                max_frame=max_frame,
            )
            backchannel.append(bc_samples["backchannel"])
            pred_backchannel.append(bc_samples["pred_backchannel"])
            pred_backchannel_neg.append(bc_negative_regions)
        return {
            "backchannel": backchannel,
            "pred_backchannel": pred_backchannel,
            "pred_backchannel_neg": pred_backchannel_neg,
        }


class TurnTakingEvents:
    def __init__(self, conf: EventConfig):
        self.conf = conf
        # Memory to add extra event in upcomming batches
        # if there is a discrepancy between
        # `pred_shift` & `pred_shift_neg` and
        # `pred_bc` & `pred_bc_neg` and
        self.add_extra = {"shift": 0, "pred_shift": 0, "pred_backchannel": 0}
        self.min_silence_time = 1

        assert (
            conf.min_context_time < conf.max_time
        ), "`minimum_context_time` must be lower than `max_time`"

        self.HS = HoldShiftNew(
            pre_cond_time=conf.sh_pre_cond_time,
            post_cond_time=conf.sh_post_cond_time,
            prediction_region_time=conf.prediction_region_time,
            prediction_region_on_active=conf.sh_prediction_region_on_active,
            long_onset_condition_time=conf.long_onset_condition_time,
            long_onset_region_time=conf.long_onset_region_time,
            min_silence_time=self.min_silence_time,
            min_context_time=conf.min_context_time,
            max_time=conf.max_time,
            frame_hz=conf.frame_hz,
        )

        self.BC = BackchannelNew(
            pre_cond_time=conf.bc_pre_cond_time,
            post_cond_time=conf.bc_post_cond_time,
            prediction_region_time=conf.prediction_region_time,
            negative_pad_left_time=conf.bc_negative_pad_left_time,
            negative_pad_right_time=conf.bc_negative_pad_right_time,
            max_bc_duration=conf.bc_max_duration,
            min_context_time=conf.min_context_time,
            max_time=conf.max_time,
            frame_hz=conf.frame_hz,
        )

    def __repr__(self) -> str:
        s = "TurnTakingEvents\n\n"
        s += self.BC.__repr__() + "\n"
        s += self.HS.__repr__()
        return s

    def get_total_ranges(self, a):
        return sum([len(events) for events in a])

    def sample_equal_amounts(
        self, n_to_sample, b_set, event_type, is_backchannel=False
    ):
        """Sample a subset from `b_set` of size of `n_to_sample`"""

        batch_size = len(b_set)

        # Create empty set
        subset = [[] for _ in range(batch_size)]

        # Flatten all events in B
        b_set_flat, batch_idx = [], []
        for b in range(batch_size):
            b_set_flat += b_set[b]
            batch_idx += [b] * len(b_set[b])

        # The maximum number of samples to sample
        n_max = len(b_set_flat)

        if n_max < n_to_sample:
            diff = n_to_sample - n_max
            self.add_extra[event_type] += diff
            n_to_sample = n_max
        else:
            diff = n_max - n_to_sample
            add_extra = min(diff, self.add_extra[event_type])
            n_to_sample += add_extra  # add extra 'negatives'
            # subtract the number of extra events we now sample
            self.add_extra[event_type] -= add_extra

        # Choose random a random subset from b_set
        for idx in random.sample(list(range(len(b_set_flat))), k=n_to_sample):
            b = batch_idx[idx]
            entry = b_set_flat[idx]
            if is_backchannel:
                entry = self.BC.sample_negative_segment(entry)
            subset[b].append(entry)
        return subset

    @torch.no_grad()
    def __call__(
        self, vad: Tensor, max_time: Optional[float] = None
    ) -> Dict[str, List[List[Tuple[int, int, int]]]]:
        assert (
            vad.ndim == 3
        ), f"Expects vad of shape (B, N_FRAMES, 2) but got {vad.shape}"
        ret = {}

        ds = get_dialog_states(vad)
        bc = self.BC(vad, ds=ds, max_time=max_time)
        hs = self.HS(vad, ds=ds, max_time=max_time)

        ret.update(bc)
        ret.update(hs)

        # Sample equal amounts of "pre-hold" regions as "pre-shift"
        # ret["pred_shift_neg"] = self.sample_pred_shift_negatives(ret)
        n_pred_shift_negs_to_sample = self.get_total_ranges(ret["pred_shift"])
        ret["pred_shift_neg"] = self.sample_equal_amounts(
            n_pred_shift_negs_to_sample, ret["pred_hold"], event_type="pred_shift"
        )
        ret.pop("pred_hold")  # remove all pred_hold regions

        # Sample equal amounts of "pred_backchannel_neg" as "pred_backchannel"
        n_pred_bc_negs_to_sample = self.get_total_ranges(ret["pred_shift"])
        ret["pred_backchannel_neg"] = self.sample_equal_amounts(
            n_pred_bc_negs_to_sample,
            ret["pred_backchannel_neg"],
            event_type="pred_backchannel",
            is_backchannel=True,
        )

        if self.conf.equal_hold_shift:
            n_holds_to_sample = self.get_total_ranges(ret["shift"])
            ret["hold"] = self.sample_equal_amounts(
                n_holds_to_sample, ret["hold"], event_type="shift"
            )
        # renames
        ret["short"] = ret.pop("backchannel")
        return ret


if __name__ == "__main__":

    conf = EventConfig()
    eventer = TurnTakingEvents(conf)
