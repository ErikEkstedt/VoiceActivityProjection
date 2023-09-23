import pytest
import torch

from vap.utils.utils import vad_list_to_onehot
from vap.events.new_events import TurnTakingEvents

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

VL_LABEL_PAUSE = {
    "shift": [
        torch.Tensor([[300, 25, 100, 75, 0], [150, 50, 150, 100, 1]]),
        torch.Tensor([[300, 25, 100, 75, 0], [150, 50, 20, 100, 1]]),
        torch.Tensor([[300, 25, 100, 75, 0], [400, 50, 75, 50, 1]]),
        [],
        [],
        [],
        [],
        torch.Tensor([[300, 25, 150, 50, 0]]),
        torch.Tensor([[300, 25, 150, 50, 0]]),
    ],
    "hold": [
        [],
        [],
        [],
        torch.Tensor([[250, 50, 250, 200, 0]]),
        torch.Tensor([[200, 20, 200, 30, 0], [250, 50, 30, 200, 0]]),
        [],
        [],
        [],
        [],
    ],
    "backchannel": [
        [],
        [],
        torch.Tensor([[105, 9, 105, 86, 1]]),
        [],
        [],
        torch.Tensor([[204, 11, 204, 285, 1]]),
        torch.Tensor(
            [[425, 25, 75, 50, 0], [105, 20, 105, 100, 1], [225, 15, 100, 150, 1]]
        ),
        [],
        [],
    ],
    "shift_overlap": [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        torch.Tensor([[100, 50, 100, 150, 1], [375, 25, 50, 100, 1]]),
        torch.Tensor([[100, 50, 25, 150, 1], [375, 25, 50, 100, 1]]),
    ],
}


VL_LABEL_STRICT = {
    "shift": [
        torch.Tensor([[300, 25, 100, 75, 0], [150, 50, 150, 100, 1]]),
        torch.Tensor([[300, 25, 100, 75, 0]]),
        torch.Tensor([[300, 25, 100, 75, 0], [400, 50, 75, 50, 1]]),
        [],
        [],
        [],
        [],
        torch.Tensor([[300, 25, 150, 50, 0]]),
        torch.Tensor([[300, 25, 150, 50, 0]]),
    ],
    "hold": [[], [], [], torch.Tensor([[250, 50, 250, 200, 0]]), [], [], [], [], []],
    "backchannel": [
        [],
        [],
        torch.Tensor([[105, 9, 105, 86, 1]]),
        [],
        [],
        torch.Tensor([[204, 11, 204, 285, 1]]),
        torch.Tensor(
            [[425, 25, 75, 50, 0], [105, 20, 105, 100, 1], [225, 15, 100, 150, 1]]
        ),
        [],
        [],
    ],
    "shift_overlap": [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        torch.Tensor([[100, 50, 100, 150, 1], [375, 25, 50, 100, 1]]),
        torch.Tensor([[375, 25, 50, 100, 1]]),
    ],
}


@pytest.mark.events
def test_events_pause_inclusion():
    vad_list = [v for _, v in VL.items()]
    vad = []
    for vl in vad_list:
        vad.append(vad_list_to_onehot(vl, duration=10, frame_hz=50))
    vad = torch.stack(vad)

    eventer = TurnTakingEvents(
        solo_pre_time=1.0,
        solo_post_time=1.0,
        silence_min_time=0.2,
        max_bc_time=1.0,
        frame_hz=50,
        event_names=["shift", "hold", "backchannel", "shift_overlap"],
        condition="pause_inclusion",
    )

    events = eventer(vad, verbose=False)

    errors = []
    for event_name, batch_list in events.items():
        print(event_name.upper())
        for bidx, batch in enumerate(batch_list):
            if batch == []:
                valid = False
                if VL_LABEL_PAUSE[event_name][bidx] == []:
                    valid = True
                else:
                    errors.append(
                        (event_name, bidx, batch, VL_LABEL_PAUSE[event_name][bidx])
                    )
            else:
                valid = (VL_LABEL_PAUSE[event_name][bidx] == batch).all().item()
            if not valid:
                errors.append(
                    (event_name, bidx, batch, VL_LABEL_PAUSE[event_name][bidx])
                )

    if len(errors) > 0:
        for e in errors:
            print(e)
        assert False, "pause_inclusion events not as expected"


@pytest.mark.events
def test_events_strict():
    vad_list = [v for _, v in VL.items()]
    vad = []
    for vl in vad_list:
        vad.append(vad_list_to_onehot(vl, duration=10, frame_hz=50))
    vad = torch.stack(vad)

    eventer = TurnTakingEvents(
        solo_pre_time=1.0,
        solo_post_time=1.0,
        silence_min_time=0.2,
        max_bc_time=1.0,
        frame_hz=50,
        event_names=["shift", "hold", "backchannel", "shift_overlap"],
        condition="strict",
    )

    events = eventer(vad, verbose=False)

    errors = []
    for event_name, batch_list in events.items():
        print(event_name.upper())
        for bidx, batch in enumerate(batch_list):
            if batch == []:
                valid = False
                if VL_LABEL_STRICT[event_name][bidx] == []:
                    valid = True
                else:
                    errors.append(
                        (event_name, bidx, batch, VL_LABEL_STRICT[event_name][bidx])
                    )
            else:
                valid = (VL_LABEL_STRICT[event_name][bidx] == batch).all().item()
            if not valid:
                errors.append(
                    (event_name, bidx, batch, VL_LABEL_STRICT[event_name][bidx])
                )

    if len(errors) > 0:
        for e in errors:
            print(e)
        assert False, "STRICT events not as expected"


# @pytest.mark.events
# def test_events_pause_inclusion():
#     vad_list = [v for _, v in VL.items()]
#     vad = []
#     for vl in vad_list:
#         vad.append(vad_list_to_onehot(vl, duration=10, frame_hz=50))
#     vad = torch.stack(vad)
#     eventer = TurnTakingEvents(
#         solo_pre_time=1.0,
#         solo_post_time=1.0,
#         silence_min_time=0.2,
#         max_bc_time=1.0,
#         frame_hz=50,
#         event_names=["shift", "hold", "backchannel", "shift_overlap"],
#         condition="pause_inclusion",
#     )
#     events = eventer(vad, verbose=False)
#     assert all(events == VL_LABEL), "pause_inclusion events not as expected"
