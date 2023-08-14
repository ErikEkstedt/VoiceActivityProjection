import torch
from torch import Tensor
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional import f1_score
from typing import Mapping, Iterable, Iterable

from vap.events.events import TurnTakingEvents, EventConfig


BATCH = Mapping[str, torch.Tensor]
EVENTS = dict[str, list[list[Iterable[int]]]]


class VAPMetric:
    EVENT_NAMES: list[str] = ["hs", "ls", "sp", "bp"]

    def __init__(self, event_config: EventConfig, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.event_config = event_config
        self.event_extractor = TurnTakingEvents(event_config)
        self.reset()

    def reset(self):
        self.preds = {k: [] for k in self.EVENT_NAMES}
        self.targets = {k: [] for k in self.EVENT_NAMES}

    def __repr__(self):
        s = f"{self.__class__.__name__}"
        s += f"\n{self.event_config})"
        for event_name in self.EVENT_NAMES:
            s += f"\n{event_name} Predicts: {len(self.preds[event_name])}"
            s += f"\n{event_name} Targets: {len(self.targets[event_name])}"
        return s

    @torch.no_grad()
    def extract_prediction_and_targets(
        self,
        p_now: torch.Tensor,
        p_fut: torch.Tensor,
        events: EVENTS,
        device=None,
    ) -> Iterable[dict[str, Tensor]]:
        """
        Iterates over all the events found in the batch and creates
        a prediction and target tensor for each event type.
        """
        batch_size = len(events["hold"])
        preds = {event_name: [] for event_name in self.EVENT_NAMES}
        targets = {event_name: [] for event_name in self.EVENT_NAMES}

        for b in range(batch_size):
            ###########################################
            # Hold vs Shift
            ###########################################
            # The metrics (i.e. shift/hold) are binary so we must decide
            # which 'class' corresponds to which numeric label
            # we use Holds=0, Shifts=1
            for start, end, speaker in events["shift"][b]:
                pshift = p_now[b, start:end]
                if speaker:
                    pshift = 1 - pshift
                preds["hs"].append(pshift)
                targets["hs"].append(torch.ones_like(pshift))

            for start, end, speaker in events["hold"][b]:
                phold = 1 - p_now[b, start:end]
                if speaker:
                    phold = 1 - phold
                preds["hs"].append(phold)
                targets["hs"].append(torch.zeros_like(phold))
            ###########################################
            # Shift-prediction
            ###########################################
            for start, end, speaker in events["pred_shift"][b]:
                # prob of next speaker -> the correct next speaker i.e. a SHIFT
                pshift = p_fut[b, start:end]
                if speaker:
                    pshift = 1 - pshift
                preds["sp"].append(pshift)
                targets["sp"].append(torch.ones_like(pshift))

            for start, end, speaker in events["pred_shift_neg"][b]:
                # prob of next speaker -> the correct next speaker i.e. a HOLD
                phold = 1 - p_fut[b, start:end]  # 1-shift = Hold
                if speaker:
                    phold = 1 - phold
                # Negatives are zero -> hold predictions
                preds["sp"].append(phold)
                targets["sp"].append(torch.zeros_like(phold))

            ###########################################
            # Long vs Shoft
            ###########################################
            # prob of next speaker -> the correct next speaker i.e. a LONG
            for start, end, speaker in events["long"][b]:
                plong = p_fut[b, start:end]
                if speaker:
                    plong = 1 - plong
                preds["ls"].append(plong)
                targets["ls"].append(torch.ones_like(plong))
            # the speaker in the 'short' events is the speaker who
            # utters a short utterance: p[b, start:end, speaker] means:
            # the  speaker saying something short has this probability
            # of continue as a 'long'
            # Therefore to correctly predict a 'short' entry this probability
            # should be low -> 0
            # thus we do not have to subtract the prob from 1 (only the labels are now zero)
            # prob of next speaker -> the correct next speaker i.e. a SHORT
            for start, end, speaker in events["short"][b]:
                pshort = p_fut[b, start:end]  # 1-shift = Hold
                if speaker:
                    pshort = 1 - pshort
                preds["ls"].append(pshort)
                # Negatives are zero -> short predictions
                targets["ls"].append(torch.zeros_like(pshort))

        # cat/stack/flatten to single tensor
        device = device if device else p_now.device
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

    def _flatten(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        preds: dict[str, Tensor] = {}
        targets: dict[str, Tensor] = {}
        for event_name in self.EVENT_NAMES:
            if self.preds[event_name] is None or len(self.preds[event_name]) == 0:
                continue
            preds[event_name] = torch.cat(self.preds[event_name])
            targets[event_name] = torch.cat(self.targets[event_name])
        return preds, targets

    def compute(self):
        preds, target = self._flatten()
        score = {}
        for event_name in preds.keys():
            p = (preds[event_name] >= self.threshold).float()
            score[event_name] = {
                "acc": accuracy(
                    p,
                    target[event_name],
                    task="multiclass",
                    num_classes=2,
                    average="none",
                ),
                "f1": f1_score(
                    p,
                    target[event_name].round(),
                    task="multiclass",
                    num_classes=2,
                    average="weighted",
                ),
            }
        return score

    def _update_metrics(self, preds, targets):
        for event_name in self.EVENT_NAMES:
            if preds[event_name] is None:
                continue
            self.preds[event_name] += [preds[event_name]]
            self.targets[event_name] += [targets[event_name]]

    @torch.no_grad()
    def update_batch(self, probs: dict[str, torch.Tensor], vad: torch.Tensor):
        events = self.event_extractor(vad)
        preds, targets = self.extract_prediction_and_targets(
            p_now=probs["p_now"], p_fut=probs["p_future"], events=events
        )
        self._update_metrics(preds, targets)


if __name__ == "__main__":

    from vap.modules.lightning_module import VAPModule
    from vap.modules.VAP import VAP
    from vap.modules.encoder import EncoderCPC
    from vap.modules.modules import TransformerStereo
    from vap.data.datamodule import VAPDataModule

    # Metric
    event_config = EventConfig()
    # Module
    encoder = EncoderCPC()
    transformer = TransformerStereo()
    model = VAP(encoder, transformer)
    module = VAPModule(model)
    dm = VAPDataModule(
        train_path="example/data/sliding_dev.csv",
        val_path="example/data/sliding_dev.csv",
        test_path="example/data/sliding_dev.csv",
        batch_size=4,
        num_workers=2,
    )
    dm.prepare_data()
    dm.setup("fit")

    dl = iter(dm.train_dataloader())

    metric = VAPMetric(event_config)

    batch = next(dl)
    out = module._step(batch)
    logits = out["logits"]
    probs = module.model.objective.get_probs(logits)

    metric.update_batch(probs, batch["vad"])
    print(metric)

    print(metric.preds["hs"])
    print(metric.targets["hs"])

    metric.preds["hs"] >= 0.5

    preds, targets = metric._flatten()

    psh = (preds["hs"] >= 0.5).float()

    targets["hs"]

    accuracy(psh, targets["hs"], task="multiclass", num_classes=2, average="none")

    score = metric.compute()
    score

    metric.reset()
