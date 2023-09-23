import torch
from torch import Tensor

# from torchmetrics.functional.classification.accuracy import accuracy
# from torchmetrics.functional import f1_score


from vap.events.new_events import HoldShiftEvents, BackchannelEvents

METRIC = dict[str, Tensor]
P_METRIC = dict[str, METRIC]
PROBS = dict[str, Tensor]

SHIFT_LABEL: int = 1
HOLD_LABEL: int = 0
SHORT_LABEL: int = 1
LONG_LABEL: int = 1

TEVENT_NAMES = list[str]
EventNames: TEVENT_NAMES = ["shift", "hold", "backchannel", "shift_overlap"]


class MetricPrediction:
    SHIFT_LABEL: int = 1
    HOLD_LABEL: int = 0

    @staticmethod
    @torch.no_grad()
    def hold(
        probs: Tensor,
        events: Tensor,
        n_frames: int,
        silence_offset: int,
        max_frame=None,
    ) -> tuple[Tensor, Tensor]:
        assert probs.ndim == 2, f"Expected probs to be 2D: (B, N) != {probs.ndim}"
        assert events.ndim == 2, f"Expected events to be 2D: (B, 3) != {events.ndim}"
        assert (
            events.shape[-1] == 6
        ), f"Expected events to be 2D: (B, 6) (to contain batch-idx) != {events.shape[-1]}"

        min_silence_frames = silence_offset + n_frames

        predictions = []
        for ev in events:
            silence_start = ev[0]
            silence_dur = ev[1]
            next_speaker = ev[4]
            batch_idx = ev[-1]

            # check silence is long enough
            if silence_dur < min_silence_frames:
                continue

            # Part of classification region is beyond horizon
            # Model will not get access to full classification region
            if max_frame is not None:
                if silence_start + min_silence_frames > max_frame:
                    continue

            start = silence_start + silence_offset
            end = start + n_frames
            next_speaker_likelihood = probs[batch_idx, start:end].cpu().mean()

            # If the label corresponds to speaker B, then flip the prediction
            if next_speaker == 1:
                next_speaker_likelihood = 1 - next_speaker_likelihood

            # We treat the label == 0 has hold prediction
            # Therefore we must flip the predictions probabilities
            # High likelihood of the next speaker -> 0
            # HOLD event: the next speaker is the same as the previous speaker
            next_speaker_likelihood = 1 - next_speaker_likelihood
            predictions.append(next_speaker_likelihood)

        if len(predictions) == 0:
            return torch.tensor([]), torch.tensor([])

        predictions = torch.stack(predictions)
        return predictions, torch.zeros_like(predictions)  # MetricPrediction.HOLD_LABEL

    @staticmethod
    @torch.no_grad()
    def shift(
        probs: Tensor,
        events: Tensor,
        n_frames: int,
        silence_offset: int,
        max_frame=None,
    ) -> tuple[Tensor, Tensor]:
        assert probs.ndim == 2, f"Expected probs to be 2D: (B, N) != {probs.ndim}"
        assert events.ndim == 2, f"Expected events to be 2D: (B, 3) != {events.ndim}"
        assert (
            events.shape[-1] == 6
        ), f"Expected events to be 2D: (B, 6) (to contain batch-idx) != {events.shape[-1]}"

        predictions = []
        min_silence_frames = silence_offset + n_frames
        for ev in events:
            silence_start = ev[0]
            silence_dur = ev[1]
            next_speaker = ev[4]
            batch_idx = ev[-1]

            # check silence is long enough
            if silence_dur < min_silence_frames:
                continue

            # Part of classification region is beyond horizon
            # Model will not get access to full classification region
            if max_frame is not None:
                if silence_start + min_silence_frames > max_frame:
                    continue

            start = silence_start + silence_offset
            end = start + n_frames
            next_speaker_likelihood = probs[batch_idx, start:end].cpu().mean()

            # If the label corresponds to speaker B, then flip the prediction
            if next_speaker == 1:
                next_speaker_likelihood = 1 - next_speaker_likelihood

            predictions.append(next_speaker_likelihood)

        if len(predictions) == 0:
            return torch.tensor([]), torch.tensor([])

        predictions = torch.stack(predictions)
        return predictions, torch.ones_like(predictions)  # MetricPrediction.SHIFT_LABEL


class HoldVsShiftClassification:
    def __init__(
        self,
        duration: float = 0.2,
        silence_offset: float = 0.05,
        min_context_time: float = 0,
        min_silence_time: float = 0.2,
        solo_pre_time: float = 1,
        solo_post_time: float = 1,
        condition: str = "pause_inclusion",
        horizon: float = 2,
        frame_hz: int = 50,
        pred_policy: list[str] = ["p_now", "p_future"],
    ) -> None:
        self.n_frames = int(duration * frame_hz)
        self.silence_offset = int(silence_offset * frame_hz)
        self.min_silence_frames = self.silence_offset + self.n_frames
        self.horizon_frames = int(horizon * frame_hz)
        self.pred_policy = pred_policy

        # Event extractor
        self.condition = condition
        self.eventer = HoldShiftEvents(
            min_context_time=min_context_time,
            min_silence_time=min_silence_time,
            solo_pre_time=solo_pre_time,
            solo_post_time=solo_post_time,
            condition=condition,
        )

    def __repr__(self) -> str:
        s = "-----------HoldVsShift----------------\n"
        s += f"{self.__class__.__name__}(\n"
        s += f"  n_frames={self.n_frames},\n"
        s += f"  silence_offset={self.silence_offset},\n"
        s += f"  min_silence_frames={self.min_silence_frames},\n"
        s += f"  horizon_frames={self.horizon_frames},\n"
        s += f"  pred_policy={self.pred_policy},\n"
        s += f"  condition={self.condition},\n"
        s += f")\n\n"
        s += self.eventer.__repr__() + "\n"
        s += "---------------------------------------"
        return s

    def __call__(
        self,
        probs: dict[str, Tensor],
        vad: Tensor,
        omit_horizon: bool = True,
    ) -> tuple[P_METRIC, P_METRIC]:
        """ """
        assert (
            probs["p_now"].ndim == 2
        ), f"Expected probs to be 2D: (B, N) != {probs['p_now'].ndim}D"

        B, N = probs["p_now"].size()
        max_frame = N - self.horizon_frames if omit_horizon else N

        # Extract Events
        holds, shifts = self.eventer(vad)

        # Extract Predictions
        preds = {
            p_name: {en: torch.empty([]) for en in ["hold", "shift"]}
            for p_name in self.pred_policy
        }
        targets = {
            p_name: {en: torch.empty([]) for en in ["hold", "shift"]}
            for p_name in self.pred_policy
        }
        for p_name in self.pred_policy:
            hp, hlabel = MetricPrediction.hold(
                probs[p_name], holds, self.n_frames, self.silence_offset, max_frame
            )
            if len(hp) > 0:
                preds[p_name]["hold"] = hp
                targets[p_name]["hold"] = hlabel

            sp, slabel = MetricPrediction.shift(
                probs[p_name], shifts, self.n_frames, self.silence_offset, max_frame
            )
            if len(sp) > 0:
                preds[p_name]["shift"] = sp
                targets[p_name]["shift"] = slabel
        return preds, targets


class BackchannelClassification:
    """
    Backchannel prediction & Long vs Short (the short part)

    -----------------------------------------------------------

    Backchannel-prediction
    A: | ##################################################|
    B:                            | ### BC #### |
    Y:        |<-- pred. time --->|

    How to infer a backchannel prediction?
    a) Use the zero-shot approach
        - Store the average/max likelihood of bc-states
    b) Does p-now favor backchanneler while p-future favors speaker
        - inside prediction region

    -----------------------------------------------------------

    Short prediction
    A: | ##################################################|
    B:                  |<---------- BC --------->|
    Y:                  |<----n_frames  -->|
    X:                  |<-p->|<-n_frames->|

    How to determine a short classification?
    a) Use the zero-shot approach
    b) Does P-future favor the next speaker?
    """

    def __init__(
        self,
        metric_time: float = 0.1,
        onset_offset_time: float = 0.1,
        prediction_time: float = 0.5,
        short_policy: str = "avg",
        max_bc_time: float = 1.0,
        solo_pre_time: float = 1.0,
        solo_post_time: float = 1.0,
        min_context_time: float = 0,
        min_silence_time: float = 0.2,
        horizon: float = 2.0,
        frame_hz: int = 50,
    ):

        self.n_frames = int(metric_time * frame_hz)
        self.onset_offset = int(onset_offset_time * frame_hz)
        self.bc_min_frames = self.n_frames + self.onset_offset
        self.prediction_frames = int(prediction_time * frame_hz)

        self.short_policy = torch.mean if short_policy == "avg" else torch.max

        self.horizon_frames = int(horizon * frame_hz)
        self.min_context_frames = int(min_context_time * frame_hz)

        # Event extractor
        self.eventer = BackchannelEvents(
            max_bc_time=max_bc_time,
            solo_pre_time=solo_pre_time,
            solo_post_time=solo_post_time,
            min_context_time=min_context_time,
            min_silence_time=min_silence_time,
            frame_hz=frame_hz,
        )

    def __repr__(self) -> str:
        s = "-----------Backchannels---------------\n"
        s += f"{self.__class__.__name__}(\n"
        s += f"  n_frames={self.n_frames},\n"
        s += f"  horizon_frames={self.horizon_frames},\n"
        s += f")\n\n"
        s += self.eventer.__repr__() + "\n"
        s += "---------------------------------------"
        return s

    def prediction_region_min_context(self):
        pass

    def flip_prob(self, prob, next_speaker):
        """
        Likelihood is always for channel 0 (speaker A)
        so we flip the probability if the `next_speaker` is channel 1 (speaker B)
        Such that the probabilities are aligned with guessing the correct
        next speaker.
        """
        if next_speaker == 1:
            prob = 1 - prob
        return prob

    def __call__(
        self,
        probs: dict[str, Tensor],
        vad: Tensor,
        omit_horizon: bool = True,
    ):

        B, N, _ = vad.size()
        max_frame = N - self.horizon_frames if omit_horizon else N
        bcs = self.eventer(vad)

        prediction_probs = []
        short_preds = []
        for bc in bcs:
            bc_onset = bc[0]
            bc_dur = bc[1]
            next_speaker = bc[4]
            batch_idx = bc[5]

            # (long vs) SHORT classification
            if bc_dur >= self.bc_min_frames:
                start = bc_onset + self.onset_offset
                end = start + self.n_frames
                prob = probs["p_future"][batch_idx, start:end]
                prob = self.flip_prob(prob, next_speaker)
                short_preds.append(self.short_policy(prob))

            # Backchannel predictions
            pred_start = bc_onset - self.prediction_frames
            if pred_start >= self.min_context_frames:
                p_next_speaker = probs["p_future"][batch_idx, pred_start:bc_onset]
                p_next_speaker = self.flip_prob(p_next_speaker, next_speaker)

                not_next_speaker = 1 - next_speaker
                p_backchanneler = probs["p_now"][batch_idx, pred_start:bc_onset]
                p_backchanneler = self.flip_prob(p_backchanneler, not_next_speaker)

                p_bc = p_next_speaker + p_backchanneler
                # if p_bc is high that means that p_now favors the backchanneller
                # while p-future favors the ongoing speaker
                # -> 2 is the best case

        short_preds = (
            torch.stack(short_preds) if len(short_preds) > 0 else torch.empty([])
        )
        short_targets = torch.ones_like(short_preds)

        return {
            "short": {"preds": short_preds, "targets": short_targets},
            "bc_pred": {},
        }


class VAPMetric:
    def __init__(
        self,
        hs_metric: HoldVsShiftClassification,
        pred_policy: list[str] = ["p_now", "p_future"],
    ):
        self.pred_policy = pred_policy
        self.hs_metric = hs_metric

    @torch.no_grad()
    def __call__(self, probs: PROBS, vad: Tensor):
        sh_preds, sh_targets = self.hs_metric(probs, vad)


# # Concatenate the tensor (flatten)
# # so that my lsp don't make a fuss
# # new_preds = {p: {en: torch.empty([]) for en in ['hold', 'shift']} for p in self.pred_policy}
# # new_targets = {p: {en: torch.empty([]) for en in ['hold', 'shift']} for p in self.pred_policy}
# new_preds = {en: {p: torch.empty([]) for p in self.pred_policy} for en in ['hold', 'shift']}
# new_targets = {en: {p: torch.empty([]) for p in self.pred_policy} for en in ['hold', 'shift']}
# for event_name in ['hold', 'shift']:
#     for p_name in self.pred_policy:
#         new_preds[event_name][p_name] = torch.cat(preds[event_name][p_name])
#         new_targets[event_name][p_name] = torch.cat(targets[event_name][p_name])
# return new_preds, new_targets


def extract_backchannel_regions(
    backchannels,
    batch_idx: int,
    bc_pred_frames: int,
) -> torch.Tensor:
    regions = []
    for bc in backchannels:
        bc_start = bc[0]
        bc_dur = bc[1]
        next_speaker = bc[-1]
        # check if backchannel is long enough
        if bc_dur < bc_pred_frames:
            continue
        regions.append([batch_idx, bc_start, bc_start + bc_pred_frames, next_speaker])
    return torch.tensor(regions)


# TODO: this is not gonna work see lsp error
def get_event_regions(
    events, silence_pred_frames, silence_pred_offset_frames, bc_pred_frames
):
    batch_size = len(events["hold"])  # random entry. they should all have the same size
    min_silence_frames = silence_pred_offset_frames + silence_pred_frames
    prediction_regions = {name: [] for name in events.keys()}
    for batch_idx in range(batch_size):
        for event_name in events.keys():
            if event_name in ["hold", "shift"]:
                reg = extract_hold_shift_regions(
                    events[event_name][batch_idx],
                    batch_idx,
                    min_silence_frames,
                    silence_pred_offset_frames,
                    silence_pred_frames,
                )
            elif event_name == "backchannel":
                reg = extract_backchannel_regions(
                    events["backchannel"][batch_idx], batch_idx, bc_pred_frames
                )
            elif event_name == "shift_overlap":
                reg = extract_backchannel_regions(
                    events["backchannel"][batch_idx], batch_idx, bc_pred_frames
                )
            else:
                raise ValueError(f"Unknown event name: {event_name}")

            prediction_regions[event_name].extend(reg)
    return prediction_regions


def get_next_speaker_probs(
    probs: torch.Tensor, shift_regions: list[torch.Tensor], max_frame=None
) -> torch.Tensor:
    """
    Gets the probability for the next speaker

    by default all `probs` are associated with speaker A (channel=0)
    so we need to flip the probability if the next speaker is B (channel=1)
    """
    preds = []
    for batch_idx, start, end, next_speaker in shift_regions:
        if max_frame is not None and end > max_frame:
            continue
        pred_frames = probs[batch_idx, start:end]
        pred = pred_frames.mean().cpu()  # pred for A is the next speaker
        # If the label corresponds to speaker B, then flip the prediction
        if next_speaker == 1:
            pred = 1 - pred
        preds.append(pred)
    if len(preds) > 0:
        return torch.stack(preds)
    else:
        return torch.tensor([])


class VAPMetricNew:
    def __init__(
        self,
        silence_pred_time=0.2,  # region condition
        silence_pred_offset_time=0.2,  # region condition
        bc_pred_time=0.4,  # region condition
        solo_pre_time: float = 1.0,  # event condition
        solo_post_time: float = 1.0,  # event condition
        max_bc_time: float = 1.0,  # event condition
        min_context_time: float = 3.0,  # event condition
        condition: str = "pause_inclusion",  # event condition
        event_names: list[str] = ["shift", "hold", "backchannel", "shift_overlap"],
        shift_prob_type: str = "p_future",
        hold_prob_type: str = "p_future",
        horizon_time: float = 2,
        frame_hz: int = 50,
    ):
        self.event_names = event_names
        self.shift_prob_type = shift_prob_type
        self.hold_prob_type = hold_prob_type

        # Frame information
        self.horizon_frames = int(horizon_time * frame_hz)
        self.silent_pred_frames = int(silence_pred_time * frame_hz)
        self.silence_pred_offset_frames = int(silence_pred_offset_time * frame_hz)
        self.bc_pred_frames = int(bc_pred_time * frame_hz)
        self.silence_min_time = silence_pred_offset_time + silence_pred_time

        # Event extractor
        self.eventer = TurnTakingEvents(
            solo_pre_time=solo_pre_time,
            solo_post_time=solo_post_time,
            silence_min_time=self.silence_min_time,
            max_bc_time=max_bc_time,
            frame_hz=frame_hz,
            event_names=event_names,
            min_context_time=min_context_time,
            condition=condition,
        )

        # Scores
        self.acc = {k: [] for k in event_names + ["shift_vs_hold"]}
        self.n = {k: [] for k in event_names + ["shift_vs_hold"]}
        self.f1 = {"shift_vs_hold": []}

        # Store all shift/hold predictions and targets
        # for Bacc and F1
        self.shift_preds_fut = []
        self.hold_preds_fut = []
        self.shift_preds_now = []
        self.hold_preds_now = []

        self.bc_preds_now = []
        self.bc_preds_fut = []

    def __repr__(self):
        s = f"VAPMetricNew(\n"
        s += f"  event_names={self.event_names}, \n"
        s += f"  shift_prob_type={self.shift_prob_type}, \n"
        s += f"  hold_prob_type={self.hold_prob_type}, \n"
        s += f"  silent_pred_frames={self.silent_pred_frames}, \n"
        s += f"  silence_pred_offset_frames={self.silence_pred_offset_frames}, \n"
        s += f"  bc_pred_frames={self.bc_pred_frames}, \n"
        s += ")"
        s += "\n\n"
        s += self.eventer.__repr__()
        s += "\n\n"
        s += "Scores: \n"
        for name in self.acc.keys():
            s += f"{name} = {self.acc[name]}, {self.n[name]} \n"
        return s

    def store_shift_hold_preds(self, probs, regions):
        """
        Shift vs Holds
        """
        max_frame = probs["p_now"].shape[1] - self.horizon_frames
        shift_preds_now = get_next_speaker_probs(
            probs["p_now"], regions["shift"], max_frame
        ).to("cpu")
        shift_preds_fut = get_next_speaker_probs(
            probs["p_future"], regions["shift"], max_frame
        ).to("cpu")
        if len(shift_preds_now) > 0:
            self.shift_preds_now.append(shift_preds_now.to("cpu"))
        if len(shift_preds_fut) > 0:
            self.shift_preds_fut.append(shift_preds_fut.to("cpu"))

        hold_preds_now = get_next_speaker_probs(
            probs["p_now"], regions["hold"], max_frame
        ).to("cpu")
        hold_preds_fut = get_next_speaker_probs(
            probs["p_future"], regions["hold"], max_frame
        ).to("cpu")
        if len(hold_preds_now) > 0:
            hold_preds_now = 1 - hold_preds_now  # flip holds to be -> label 0
            self.hold_preds_now.append(hold_preds_now)
        if len(hold_preds_fut) > 0:
            hold_preds_fut = 1 - hold_preds_fut  # flip holds to be -> label 0
            self.hold_preds_fut.append(hold_preds_fut)

    def store_bc_preds(self, probs, regions):
        max_frame = probs["p_now"].shape[1] - self.horizon_frames
        bc_preds_now = get_next_speaker_probs(
            probs["p_now"], regions["backchannel"], max_frame
        ).to("cpu")
        bc_preds_fut = get_next_speaker_probs(
            probs["p_future"], regions["backchannel"], max_frame
        ).to("cpu")
        if len(bc_preds_now) > 0:
            self.bc_preds_now.append(bc_preds_now)
        if len(bc_preds_fut) > 0:
            self.bc_preds_fut.append(bc_preds_fut)

    def compute_shift_vs_hold_acc(self, shift_preds, hold_preds):
        shift_preds = torch.cat(shift_preds)
        hold_preds = torch.cat(hold_preds)
        acc = {
            "shift": (shift_preds > 0.5).sum() / len(shift_preds),
            "hold": (hold_preds < 0.5).sum() / len(hold_preds),
        }
        acc["shift_vs_hold"] = (acc["shift"] + acc["hold"]) / 2
        return acc

    def compute_shift_vs_hold_f1(self, shift_preds, hold_preds):
        shift_preds = torch.cat(shift_preds)
        hold_preds = torch.cat(hold_preds)
        shift_targets = torch.ones_like(shift_preds)
        hold_targets = torch.zeros_like(hold_preds)
        sh_preds = torch.cat([shift_preds, hold_preds])
        sh_targets = torch.cat([shift_targets, hold_targets])
        return f1_score(
            sh_preds > 0.5,
            sh_targets,
            task="multiclass",
            num_classes=2,
            average="weighted",
        )

    def compute_hold_vs_shift(self):
        acc = {"now": {}, "fut": {}}
        acc["now"] = self.compute_shift_vs_hold_acc(
            self.shift_preds_now, self.hold_preds_now
        )
        acc["fut"] = self.compute_shift_vs_hold_acc(
            self.shift_preds_fut, self.hold_preds_fut
        )
        f1 = {"now": {}, "fut": {}}
        f1["now"]["shift_vs_hold"] = self.compute_shift_vs_hold_f1(
            self.shift_preds_now, self.hold_preds_now
        )
        f1["fut"]["shift_vs_hold"] = self.compute_shift_vs_hold_f1(
            self.shift_preds_fut, self.hold_preds_fut
        )
        return {"acc": acc, "f1": f1}

    def compute_bc(self):
        bc_preds_now = torch.cat(self.bc_preds_now)
        bc_preds_fut = torch.cat(self.bc_preds_fut)
        acc = {
            "now": (bc_preds_now > 0.5).sum() / len(bc_preds_now),
            "fut": (bc_preds_fut > 0.5).sum() / len(bc_preds_fut),
        }
        return acc

    def compute(self):
        hs = self.compute_hold_vs_shift()
        bc = self.compute_bc()
        return {"hs": hs, "bc": bc}

    @torch.no_grad()
    def update_batch(self, probs: dict[str, torch.Tensor], vad: torch.Tensor):
        events = self.eventer(vad)
        regions = get_event_regions(
            events,
            silence_pred_frames=self.silent_pred_frames,
            silence_pred_offset_frames=self.silence_pred_offset_frames,
            bc_pred_frames=self.bc_pred_frames,
        )
        self.store_shift_hold_preds(probs, regions)
        self.store_bc_preds(probs, regions)


if __name__ == "__main__":

    from tqdm import tqdm
    from vap.utils.utils import vad_list_to_onehot
    from vap.modules.lightning_module import VAPModule
    from vap.data.datamodule import VAPDataModule
    from vap.events.new_events import HoldShiftEvents, BackchannelEvents, example_vad
    import matplotlib.pyplot as plt

    vad = example_vad()["vad"]

    # probs = torch.rand_like(vad[..., 0])
    probs = {
        "p_now": torch.rand_like(vad[..., 0]),
        "p_future": torch.rand_like(vad[..., 0]),
    }

    SH = HoldVsShiftClassification()
    BC = BackchannelClassification()

    bc = BC(probs, vad)

    # BC
    # tensor([[105,   9, 105,  86,   0,   2],
    #         [204,  11, 204, 285,   0,   5],
    #         [425,  25,  75,  50,   1,   6],
    #         [105,  20, 105, 100,   0,   6],
    #         [225,  15, 100, 150,   0,   6]])

    plot_events_batch(vad, {"backchannel": bc})

    preds, targets = SH(probs, vad)

    holds, shifts = HoldShiftEvents()(vad)
    preds, targets = SH(probs, holds=holds, shifts=shifts)

    # metric = VAPMetricNew()
    # print(metric)
    #
    # m = 9
    # probs = {
    #     "p_now": torch.ones_like(vad[:m, :, 0]) * 0.1,
    #     "p_future": torch.ones_like(vad[:m, :, 0]) * 0.2,
    # }
    # metric.update_batch(probs, vad[:m])
    # print(metric)
    #
    # score = metric.compute()
    # score = torch.stack(metric.acc["shift"])
    # weight = torch.tensor(metric.n["shift"])
    # N = weight.sum()
    # s = (score * weight).sum() / N
    # print(s)

    p = "/home/erik/projects/CCConv/VoiceActivityProjection/data/splits/swb/val_20s_5o.csv"
    dm = VAPDataModule(
        train_path=p, val_path=p, num_workers=0, batch_size=15, prefetch_factor=None
    )
    dm.setup()
    dloader = dm.val_dataloader()
    model = VAPModule.load_from_checkpoint("example/checkpoints/checkpoint.ckpt").model
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # diter = iter(dloader)

    metric = VAPMetricNew()
    N = 100
    n = 0
    # pbar = tqdm(total=N)
    with torch.inference_mode():
        for batch in tqdm(dloader):
            torch.cuda.empty_cache()
            probs = model.probs(batch["waveform"].to(model.device))
            metric.update_batch(probs, batch["vad"])
            # score = metric.compute()
            # pbar.update(1)
            # n += 1
            # if n == N:
            #     break
    # pbar.close()
    score = metric.compute()
    print(score)

    # for metric_type, scores in score.items():
    #     for p_type, ss in scores.items():
    #         print(metric_type.upper(), p_type.upper())
    #         for event_name, s in ss.items():
    #             print(f"{event_name}: {s:.2f}")
    #
    # ss = metric.compute_shift_hold_custom()
    # for event_name in ["shift", "hold", "shift_vs_hold"]:
    #     s0 = score["acc"][event_name]
    #     s1 = ss["acc"][event_name]
    #     print(f"{event_name}: {s0:.2f} vs {s1:.2f}")
    # print(
    #     f"F1 shift_vs_hold: {score['f1']['shift_vs_hold']:.2f} vs {ss['f1']['shift_vs_hold']:.2f}"
    # )
    #
    # print("SCORE")
    # print(score)
    # print("SCORE CUSTOM")
    # print("Acc: ", ss["acc"])
    # print("F1: ", ss["f1"])
