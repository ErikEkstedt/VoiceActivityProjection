import torch
from torch import Tensor
from typing import Optional

from vap.utils.utils import (
    find_island_idx_len,
    get_dialog_states,
    vad_list_to_onehot,
    vad_onehot_to_vad_list,
)


# Templates
TRIAD_SHIFT: Tensor = torch.tensor([[3, 1, 0], [0, 1, 3]])  # on Silence
TRIAD_SHIFT_OVERLAP: Tensor = torch.tensor([[3, 2, 0], [0, 2, 3]])
TRIAD_HOLD: Tensor = torch.tensor([[0, 1, 0], [3, 1, 3]])  # on silence
TRIAD_BC: Tensor = torch.tensor([0, 1, 0])

TRIADS = {
    "shift": TRIAD_SHIFT,
    "hold": TRIAD_HOLD,
    "backchannel": TRIAD_BC,
    "shift_overlap": TRIAD_SHIFT_OVERLAP,
}

# Dialog states meaning
STATE_ONLY_A: int = 0
STATE_ONLY_B: int = 3
STATE_SILENCE: int = 1
STATE_BOTH: int = 2

EVENTS = dict[str, list[Tensor]]


def example_vad():
    VL = {
        "shift": [[[0.0, 3.0], [6.5, 8]], [[4.0, 6.0], [9.5, 10]]],
        "shift_pause": [[[0.0, 2.0], [2.6, 3], [6.5, 8]], [[4.0, 6.0], [9.5, 10]]],
        "shift_pause_bad": [
            [[0.0, 2.0], [2.6, 3], [6.5, 8]],
            [[2.1, 2.3], [4.0, 6.0], [9, 10]],
        ],
        "hold": [[[0.0, 5.0], [6.0, 10.0]], []],
        "hold_pause": [[[0.0, 4.0], [4.4, 5], [6.0, 10.0]], []],
        "hold_pause_bad": [[[0.0, 4.0], [4.4, 5], [6.0, 10.0]], [[4.1, 4.3]]],
        "bc": [
            [[0.0, 3.0], [3.2, 7], [8.5, 9.0]],
            [[2.1, 2.5], [4.5, 4.8], [7.8, 10.0]],
        ],
        "shift_overlap": [[[0.0, 3.0], [6.5, 8]], [[2.0, 6.0], [7.5, 10]]],
        "shift_overlap_pause": [
            [[0.0, 1.2], [1.5, 3.0], [6.5, 8]],
            [[2.0, 6.0], [7.5, 10]],
        ],
    }
    vad_list = [v for _, v in VL.items()]
    vad = []
    for vl in vad_list:
        vad.append(vad_list_to_onehot(vl, duration=10, frame_hz=50))
    vad = torch.stack(vad)
    return {"vad": vad, "vad_list": vad_list}


@torch.no_grad()
def fill_pauses(
    vad: Tensor,
    ds: Optional[Tensor] = None,
    islands: Optional[tuple[Tensor, Tensor, Tensor]] = None,
) -> Tensor:
    assert vad.ndim == 2, "fill_pauses require vad=(n_frames, 2)"

    filled_vad = vad.clone()

    if islands is None:
        if ds is None:
            ds = get_dialog_states(vad)

        assert ds.ndim == 1, "fill_pauses require ds=(n_frames,)"
        s, d, v = find_island_idx_len(ds)
    else:
        s, d, v = islands

    # less than three entries means that there are no pauses
    # requires at least: activity-from-speaker  ->  Silence   --> activity-from-speaker
    if len(v) < 3:
        return vad

    # Find all holds/pauses and fill them
    # A hold is defined to not contain activity by the listener
    # inside the pause
    triads = v.unfold(0, size=3, step=1)
    next_speaker, steps = torch.where(
        (triads == TRIAD_HOLD.unsqueeze(1).to(triads.device)).sum(-1) == 3
    )
    for ns, pre in zip(next_speaker, steps):
        cur = pre + 1
        # Fill the matching template
        filled_vad[s[cur] : s[cur] + d[cur], ns] = 1.0
    return filled_vad


class EventCandidates:
    @staticmethod
    @torch.no_grad()
    def triad_matches(triads, start_of, duration_of, triad_label) -> torch.Tensor:
        """
        Extracts the timing (frames) for the event specified by the triad_label.

        Returns a (N, 5) tensor with the following columns:
            * start-of-silence
            * duration-of-silence
            * duration-of-pre-state
            * duration-of-post-state
            * next-speaker

        RETURNS:
            events:  Tensor, (N, 5) tensor
        """
        # Get Shift triads matches
        matches = (triads == triad_label.unsqueeze(1)).sum(-1) == 3
        if matches.any():  # if we have a match we continue
            next_speakers, steps = torch.where(matches)
            silence_start = start_of[steps + 1]
            silence_dur = duration_of[steps + 1]
            dur_pre_state = duration_of[steps]
            dur_post_state = duration_of[steps + 2]
            events = torch.stack(
                [
                    silence_start,
                    silence_dur,
                    dur_pre_state,
                    dur_post_state,
                    next_speakers,
                ],
                dim=-1,
            )
        else:
            events = torch.tensor([])
        return events

    @staticmethod
    @torch.no_grad()
    def hold_shift(vad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the timing (frames) for the events specified by the `event_names`.
        Returns a dict containing (N, 5) tensor (for each entry of `event_names`)
             or a dict containing (N, 6) with batch_idx at the end

        VALID event_names: ["shift", "hold", "shift_overlap"]
        with the following columns:
            * start-of-silence
            * duration-of-silence
            * duration-of-pre-state
            * duration-of-post-state
            * next-speaker
            * batch_idx (optional)

        RETURNS:
            events:  Tensor, (N, 5) tensor or (N, 6)
        """
        assert vad.ndim == 2 or vad.ndim == 3, f"vad.ndim={vad.ndim} is not 2D or 3D"
        assert (
            vad.shape[-1] == 2
        ), f"Expected 2 channels and channels last, got {vad.shape}"

        def append_batch_idx(batch_idx, sh):
            z = torch.ones((len(sh), 1), dtype=sh.dtype).fill_(batch_idx)  # (N, 1)
            return torch.cat((sh, z), dim=1)

        # If we don't have a batch  input
        if vad.ndim == 2:
            ds = get_dialog_states(vad)
            start_of, duration_of, states = find_island_idx_len(ds)
            triads = states.unfold(0, size=3, step=1)
            hold = EventCandidates.triad_matches(
                triads, start_of, duration_of, TRIADS["hold"]
            )
            shift = EventCandidates.triad_matches(
                triads, start_of, duration_of, TRIADS["shift"]
            )
        else:  # batch input
            hold, shift = [], []
            all_ds = get_dialog_states(vad)
            for batch_idx, ds in enumerate(all_ds):
                start_of, duration_of, states = find_island_idx_len(ds)
                triads = states.unfold(0, size=3, step=1)

                h = EventCandidates.triad_matches(
                    triads, start_of, duration_of, TRIADS["hold"]
                )
                if len(h) > 0:
                    h = append_batch_idx(batch_idx, h)
                    hold.append(h)

                s = EventCandidates.triad_matches(
                    triads, start_of, duration_of, TRIADS["shift"]
                )
                if len(s) > 0:
                    s = append_batch_idx(batch_idx, s)
                    shift.append(s)

            # stack tensors
            hold = torch.cat(hold, dim=0) if len(hold) > 0 else torch.tensor([])
            shift = torch.cat(shift, dim=0) if len(shift) > 0 else torch.tensor([])
        return hold, shift

    @staticmethod
    @torch.no_grad()
    def backchannel_matches(
        vad: Tensor,
        max_bc_frames: int,
        solo_pre_frames: int,
        solo_post_frames: int,
    ) -> Tensor:
        """
        1. Checks for Backchannel candidates which are isolated 'islands' of activity shorter than `max_bc_frames`
        2. Checks that there is activity from the other speaker in the `solo_pre_frames` before and `solo_post_frames` after

        with the following columns:
            * start-of-bc
            * duration-of-bc
            * duration-pre-bc-state
            * duration-post-bc-state
            * next-speaker
        """

        assert vad.ndim == 2, f"expects vad of shape (n_frames, 2) but got {vad.shape}."
        filled_vad = fill_pauses(vad)

        backchannel = []
        for speaker in [0, 1]:
            start_of, duration_of, states = find_island_idx_len(
                filled_vad[..., speaker]
            )
            if len(states) < 3:
                continue
            triads = states.unfold(0, size=3, step=1)
            steps = torch.where(
                (triads == TRIAD_BC.to(triads.device).unsqueeze(0)).sum(-1) == 3
            )[0]
            if len(steps) == 0:
                continue

            for pre_bc in steps:
                bc = pre_bc + 1
                post_bc = pre_bc + 2
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
                if duration_of[pre_bc] < solo_pre_frames:
                    # print('not enough silence PRIOR to "bc"')
                    continue

                ################################################
                # POST CONDITION: No post activity from bc-speaker
                ################################################
                if duration_of[post_bc] < solo_post_frames:
                    # print('not enough silence POST to "bc"')
                    continue

                ################################################
                # Other speaker active pre/post ?
                ################################################
                # Is the other speakr active before this segment?
                bc_start = start_of[bc]
                bc_end = bc_start + duration_of[bc]
                other_speaker = 1 - speaker

                pre_start = bc_start - solo_pre_frames
                pre_activity_other = filled_vad[pre_start:bc_start, other_speaker].sum()
                if pre_activity_other == 0:
                    continue

                post_end = bc_end + solo_post_frames
                post_activity_other = filled_vad[bc_end:post_end, other_speaker].sum()
                if post_activity_other == 0:
                    continue

                ################################################
                # ALL CONDITIONS MET
                ################################################

                backchannel.append(
                    (
                        start_of[bc],
                        duration_of[bc],
                        duration_of[pre_bc],
                        duration_of[post_bc],
                        other_speaker,
                    )
                )

        if len(backchannel) == 0:
            return torch.tensor([])
        return torch.tensor(backchannel)

    @staticmethod
    @torch.no_grad()
    def backchannel(
        vad: Tensor,
        max_bc_frames: int,
        solo_pre_frames: int,
        solo_post_frames: int,
    ) -> Tensor:
        """
        Extracts the timing (frames) for the events specified by the `event_names`.
        Returns a dict containing (N, 5) tensor (for each entry of `event_names`)
             or a dict containing (N, 6) with batch_idx at the end

        VALID event_names: ["shift", "hold", "shift_overlap"]
        with the following columns:
            * start-of-silence
            * duration-of-silence
            * duration-of-pre-state
            * duration-of-post-state
            * next-speaker
            * batch_idx (optional)

        RETURNS:
            events:  Tensor, (N, 5) tensor or (N, 6)
        """
        assert vad.ndim == 2 or vad.ndim == 3, f"vad.ndim={vad.ndim} is not 2D or 3D"
        assert (
            vad.shape[-1] == 2
        ), f"Expected 2 channels and channels last, got {vad.shape}"

        def append_batch_idx(batch_idx, sh):
            z = torch.ones((len(sh), 1), dtype=sh.dtype).fill_(batch_idx)  # (N, 1)
            return torch.cat((sh, z), dim=1)

        # If we don't have a batch  input
        if vad.ndim == 2:
            bc = EventCandidates.backchannel_matches(
                vad, max_bc_frames, solo_pre_frames, solo_post_frames
            )
        else:  # batch input
            bc = []
            for batch_idx, vad_single in enumerate(vad):
                bc_tmp = EventCandidates.backchannel_matches(
                    vad_single, max_bc_frames, solo_pre_frames, solo_post_frames
                )
                if len(bc_tmp) > 0:
                    bc_tmp = append_batch_idx(batch_idx, bc_tmp)
                    bc.append(bc_tmp)
            # stack tensors
            bc = torch.cat(bc, dim=0) if len(bc) > 0 else torch.tensor([])
        return bc


class EventConditions:
    @staticmethod
    @torch.no_grad()
    def filter_min_context(
        events: torch.Tensor, min_context_frames: int = 0
    ) -> torch.Tensor:
        """
        Filters out events that have less than `min_context` frames of context

        events: torch.Tensor (N, 5) or (N, 6) if batch_idx is included
        """

        if len(events) == 0:
            return events

        if min_context_frames == 0:
            return events

        # Minimum context
        valid_idx = torch.where(events[:, 0] >= min_context_frames)
        if len(valid_idx) == 0:
            return torch.tensor([])
        return events[valid_idx]

    @staticmethod
    @torch.no_grad()
    def filter_min_silence(
        events: torch.Tensor, min_silence_frames: int = 0
    ) -> torch.Tensor:
        """
        Filters out events that have less than `min_silence_frames` frames of silence

        events: torch.Tensor (N, 5) or (N, 6) if batch_idx is included
        """

        if len(events) == 0:
            return events

        if min_silence_frames == 0:
            return events

        valid_idx = torch.where(events[:, 1] >= min_silence_frames)
        if len(valid_idx) == 0:
            return torch.tensor([])
        return events[valid_idx]

    @staticmethod
    @torch.no_grad()
    def filter_pre_post_strict(
        events: torch.Tensor, solo_pre_frames: int = 0, solo_post_frames: int = 0
    ) -> torch.Tensor:
        """
        Filters out events that have less than `min_silence_frames` frames of silence

        events: torch.Tensor (N, 5) or (N, 6) if batch_idx is included
        """

        if len(events) == 0:
            return events

        # no op
        if solo_pre_frames == 0 and solo_post_frames == 0:
            return events

        # Solo pre frame condition
        valid_idx = torch.where(events[:, 2] >= solo_pre_frames)
        if len(valid_idx) == 0:
            return torch.tensor([])

        events = events[valid_idx]

        # Solo post frame condition
        valid_idx = torch.where(events[:, 3] >= solo_post_frames)
        if len(valid_idx) == 0:
            return torch.tensor([])
        return events[valid_idx]

    @staticmethod
    @torch.no_grad()
    def filter_pre_post_pause_inclusion(
        events: torch.Tensor,
        vad: torch.Tensor,
        solo_pre_frames: int = 0,
        solo_post_frames: int = 0,
        event_name: str = "hold",
    ) -> torch.Tensor:
        """
        Fill out all the pauses in the vad and perform pre-post condition
        here we only care about if the previous 'turn' was actually from one speaker
        however the previous state must not cover everything.

        * events is either (N, 5) or (N, 6) if batch_idx is included or not
        * batch_idx is always last

        Example
               start   dur pre-dur post-dur next-spkr  batch idx (optional)
        tensor([[300.,  25., 100.,  75.,      0.,       0.],
                [150.,  50., 150., 100.,      1.,       0.],
                [300.,  25., 100.,  75.,      0.,       1.],
                [300.,  25., 100.,  75.,      0.,       2.],
                [400.,  50.,  75.,  50.,      1.,       2.],
                [300.,  25., 150.,  50.,      0.,       7.],
                [300.,  25., 150.,  50.,      0.,       8.]])
        """
        assert events.ndim == 2, f"events must be 2D, got {events.ndim}D"

        if len(events) == 0:
            return events

        # no op
        if solo_pre_frames == 0 and solo_post_frames == 0:
            return events

        def filter_single(ev: torch.Tensor, vad_filled: torch.Tensor) -> torch.Tensor:
            silence_start = ev[0]
            next_speaker = ev[4]
            prev_speaker = next_speaker if event_name == "hold" else 1 - next_speaker
            not_next_speaker = 1 - next_speaker
            not_prev_speaker = 1 - prev_speaker

            pre_start = silence_start - solo_pre_frames
            if pre_start < 0:
                # print("pre_start < 0")
                return torch.tensor([])

            ######################################################
            # PRE
            ######################################################
            # Check if the filled vad contains sufficient activity from the previous speaker
            pre_activity = vad_filled[pre_start:silence_start, prev_speaker]
            if pre_activity.sum() != solo_pre_frames:
                # print("pre_activity.sum() != solo_pre_frames")
                return torch.tensor([])

            # The other speaker should not be active in pre-region
            pre_activity_other = vad_filled[pre_start:silence_start, not_prev_speaker]
            if pre_activity_other.sum() > 0:
                # print("pre_activity_other.sum() > 0")
                return torch.tensor([])

            ######################################################
            # POST
            ######################################################
            next_onset_start = silence_start + ev[1]  # add silence duration

            # if condition is outside of vad
            # post condition requires more future info
            post_cond_end = next_onset_start + solo_post_frames
            if post_cond_end > vad.shape[1]:
                # print("post_cond_end > vad.shape[0]")
                return torch.tensor([])

            # Check if the filled vad contains sufficient activity from the next speaker
            post_activity = vad_filled[next_onset_start:post_cond_end, next_speaker]
            if post_activity.sum() != solo_post_frames:
                # print("post_activity.sum() != solo_post_frames")
                return torch.tensor([])

            # The other speaker should not be active in post-region
            post_other_activity = vad_filled[
                next_onset_start:post_cond_end, not_next_speaker
            ]
            if post_other_activity.sum() > 0:
                # print("post_other_activity.sum() > 0")
                return torch.tensor([])

            return ev

        if events.shape[-1] == 5:
            if vad.ndim > 2:
                assert (
                    vad.shape[0] == 1
                ), f"If batch idx is not included in `events` we expect a single vad (N, 2) or (1, N, 2). Got: {vad.shpae}"
                vad_filled = fill_pauses(vad[0])
            else:
                vad_filled = fill_pauses(vad)
            new_events = filter_single(events, vad_filled)
        else:
            # batch idx are included
            new_events = []
            batch_idx_unique = events[:, -1].unique().long()
            for batch_idx in batch_idx_unique:
                vad_filled = fill_pauses(vad[batch_idx])
                evs = events[events[:, -1] == batch_idx]
                for ev in evs:
                    ev = filter_single(ev, vad_filled)
                    if len(ev) > 0:
                        new_events.append(ev)
            new_events = (
                torch.stack(new_events) if len(new_events) > 0 else torch.tensor([])
            )
        return new_events

    @staticmethod
    @torch.no_grad()
    def filter_hold_shift(
        events: torch.Tensor,
        vad: Optional[torch.Tensor] = None,
        min_context_frames: int = 0,
        min_silence_frames: int = 0,
        solo_pre_frames: int = 0,
        solo_post_frames: int = 0,
        condition="strict",
        event_name: Optional[str] = None,
    ) -> torch.Tensor:
        assert condition in [
            "strict",
            "pause_inclusion",
        ], f"Unknown condition: {condition}"

        # No-op
        if len(events) == 0:
            return events

        events = EventConditions.filter_min_context(events, min_context_frames)
        if len(events) == 0:
            return events

        events = EventConditions.filter_min_silence(events, min_silence_frames)
        if len(events) == 0:
            return events

        if condition == "strict":
            events = EventConditions.filter_pre_post_strict(
                events, solo_pre_frames, solo_post_frames
            )
        else:
            assert isinstance(
                vad, torch.Tensor
            ), f"VAD must be a torch.Tensor. got: {type(vad)}"
            assert isinstance(
                event_name, str
            ), f"event_name must be a str. got: {type(event_name)}"
            assert event_name in [
                "hold",
                "shift",
            ], f"event_name must be 'hold' or 'shift'. got: {event_name}"
            events = EventConditions.filter_pre_post_pause_inclusion(
                events=events,
                vad=vad,
                solo_pre_frames=solo_pre_frames,
                solo_post_frames=solo_post_frames,
                event_name=event_name,
            )
        return events


class HoldShiftEvents:
    def __init__(
        self,
        min_context_time: float = 0,
        min_silence_time: float = 0.2,
        solo_pre_time: float = 1.0,
        solo_post_time: float = 1.0,
        condition: str = "pause_inclusion",
        frame_hz: int = 50,
    ):
        self.min_context_frames = int(min_context_time * frame_hz)
        self.min_silence_frames = int(min_silence_time * frame_hz)
        self.solo_pre_frames = int(solo_pre_time * frame_hz)
        self.solo_post_frames = int(solo_post_time * frame_hz)
        self.condition = condition
        self.frame_hz = frame_hz

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  min_context_frames={self.min_context_frames},\n"
        s += f"  min_silence_frames={self.min_silence_frames},\n"
        s += f"  solo_pre_frames={self.solo_pre_frames},\n"
        s += f"  solo_post_frames={self.solo_post_frames},\n"
        s += f"  condition={self.condition},\n"
        s += ")"
        return s

    def __call__(self, vad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hold, shift = EventCandidates.hold_shift(vad)
        # Apply condition to filter candidates
        hold = EventConditions.filter_hold_shift(
            hold,
            vad=vad,
            min_context_frames=self.min_context_frames,
            min_silence_frames=self.min_silence_frames,
            solo_pre_frames=self.solo_pre_frames,
            solo_post_frames=self.solo_post_frames,
            condition=self.condition,
            event_name="hold",
        )
        shift = EventConditions.filter_hold_shift(
            shift,
            vad=vad,
            min_context_frames=self.min_context_frames,
            min_silence_frames=self.min_silence_frames,
            solo_pre_frames=self.solo_pre_frames,
            solo_post_frames=self.solo_post_frames,
            condition=self.condition,
            event_name="shift",
        )
        return hold, shift


class BackchannelEvents:
    def __init__(
        self,
        max_bc_time: float = 1.0,
        solo_pre_time: float = 1.0,
        solo_post_time: float = 1.0,
        min_context_time: float = 0,
        min_silence_time: float = 0.2,
        frame_hz: int = 50,
    ):
        self.max_bc_frames = int(max_bc_time * frame_hz)
        self.solo_pre_frames = int(solo_pre_time * frame_hz)
        self.solo_post_frames = int(solo_post_time * frame_hz)

        self.min_context_frames = int(min_context_time * frame_hz)
        self.min_silence_frames = int(min_silence_time * frame_hz)
        self.frame_hz = frame_hz

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  max_bc_frames={self.max_bc_frames},\n"
        s += f"  solo_pre_frames={self.solo_pre_frames},\n"
        s += f"  solo_post_frames={self.solo_post_frames},\n"
        s += f"  min_context_frames={self.min_context_frames},\n"
        s += f"  min_silence_frames={self.min_silence_frames},\n"
        s += ")"
        return s

    def __call__(self, vad: Tensor) -> Tensor:
        bc = EventCandidates.backchannel(
            vad, self.max_bc_frames, self.solo_pre_frames, self.solo_post_frames
        )

        if len(bc) == 0:
            return torch.tensor([])

        bc = EventConditions.filter_min_context(bc, self.min_context_frames)
        return bc


class PE:
    @staticmethod
    def plot_regions(xs, ws, ax, height=1, color="g", frame_hz=50):
        for x, w in zip(xs, ws):
            PE.plot_region(x, w, ax, height=height, color=color, frame_hz=frame_hz)

    @staticmethod
    def plot_region(x, w, ax, height=1, color="g", frame_hz=50):
        x = x / frame_hz
        w = w / frame_hz
        ax.add_patch(
            plt.Rectangle(
                (x, 0),
                width=w,
                height=height,
                edgecolor=color,
                facecolor=color,
                alpha=0.5,
            )
        )

    @staticmethod
    def plot_vad_oh(
        vad, ax, draw_style="horizontal", y=None, color="b", frame_hz=50, linewidth=3
    ):
        t = torch.arange(len(vad)) / frame_hz
        if y is None:
            y0, y1 = ax.get_ylim()
            y = (y0 + y1) / 2

        if draw_style == "horizontal":
            s, d, v = find_island_idx_len(vad)
            s = s[v == 1] - 1
            d = d[v == 1] - 1
            x0 = t[s]
            x1 = t[s + d]
            ax.hlines(
                y=[y] * len(x0), xmin=x0, xmax=x1, color=color, linewidth=linewidth
            )
        elif draw_style == "rectangle":
            s, d, v = find_island_idx_len(vad)
            s = s[v == 1] - 1
            d = d[v == 1] - 1
            x0 = t[s]
            x1 = t[s + d]
            for x0_, x1_ in zip(x0, x1):
                ax.add_patch(
                    plt.Rectangle(
                        (x0_, y),
                        width=x1_ - x0_,
                        height=0.1,
                        linewidth=0,
                        edgecolor=color,
                        facecolor=color,
                    )
                )
        else:
            ax.plot(t, vad, color=color)

    @staticmethod
    def plot_vad_oh_rect(vad, ax, y_center=0.5, height=0.45, alpha=0.6, frame_hz=50):
        vl = vad_onehot_to_vad_list(vad.unsqueeze(0))[0]
        PE.plot_vad_list_rect(vl, ax, y_center=y_center, height=height, alpha=alpha)

    @staticmethod
    def plot_vad_list_rect(vad_list, ax, y_center=0.5, height=0.45, alpha=0.6):
        for ii, ch in enumerate(vad_list):
            y = y_center if ii == 0 else y_center - height
            color = "b" if ii == 0 else "orange"
            for x0, x1 in ch:
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y),
                        width=x1 - x0,
                        height=height,
                        linewidth=1,
                        edgecolor=color,
                        facecolor=color,
                        alpha=alpha,
                    )
                )


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from vap.utils.utils import read_json
    from vap.utils.audio import get_audio_info

    # from vap.events.events import TurnTakingEvents

    audio_path = "/home/erik/projects/data/switchboard/audio/swb1_d1/data/sw02005.wav"
    duration = get_audio_info(audio_path)["duration"]
    vad_list = read_json(
        "/home/erik/projects/CCConv/vap_switchboard/data/vad_list/sw02005.json"
    )
    vad = vad_list_to_onehot(vad_list, duration, frame_hz=50)

    def plot_events_batch(vad, events):
        fig, axs = plt.subplots(
            vad.shape[0], 1, sharex=True, sharey=True, figsize=(9, vad.shape[0])
        )
        for batch_idx in range(len(vad)):
            PE.plot_vad_oh_rect(
                vad[batch_idx], axs[batch_idx], y_center=0.5, height=0.45
            )

        if "shift" in events:
            for shift in events["shift"]:
                x, w = shift[:2]
                batch_idx = shift[-1].long()
                PE.plot_region(x, w, axs[batch_idx], height=1, color="g")
        if "hold" in events:
            for hold in events["hold"]:
                x, w = hold[:2]
                batch_idx = hold[-1].long()
                PE.plot_region(x, w, axs[batch_idx], height=1, color="r")

        if "backchannel" in events:
            for hold in events["backchannel"]:
                x, w = hold[:2]
                batch_idx = hold[-1].long()
                PE.plot_region(x, w, axs[batch_idx], height=1, color="purple")
        axs[0].set_xlim(0, 11)
        plt.show(block=False)

    vad = example_vad()["vad"]

    bc = EventCandidates.backchannel(
        vad, max_bc_frames=50, solo_pre_frames=50, solo_post_frames=50
    )
    plot_events_batch(vad, {"backchannel": bc})
    bc = BackchannelEvents(min_context_time=2.5)(vad)
    plot_events_batch(vad, {"backchannel": bc})

    hold, shift = EventCandidates.hold_shift(vad)
    plot_events_batch(vad, {"shift": shift, "hold": hold})
    hold, shift = HoldShiftEvents(condition="strict")(vad)
    plot_events_batch(vad, {"shift": shift, "hold": hold})
    hold, shift = HoldShiftEvents(condition="pause_inclusion")(vad)
    plot_events_batch(vad, {"shift": shift, "hold": hold})
