# Events

Automatic extraction of turn-taking events.

* `streamlit run vap/events/streamlit_events.py`




## Information

The `TurnTakingEvents` module expects a VAD tensor of shape `(B, N_FRAMES, 2)` and outputs a dictionary with event frames.
See also [vap/metrics.py](vap/metrics.py) to get a sense of how it's used during training.

```python
conf = EventConfig()
eventer = TurnTakingEvents(conf)
events = eventer(vad)  # vad: Tensor (b, n_frames, 2) onehot /binary

# events: dict_keys([
#     'pred_backchannel',
#     'pred_backchannel_neg',
#     'shift',
#     'hold',
#     'long',
#     'pred_shift',
#     'pred_shift_neg',
#     'short'
#     ])
```

where an entry, e.g. 'shift', contains a list of shift events over the batch. containing `start_frame, end_frame, speaker` where the speaker represent the next-speaker, the speaker (channel) who is active after the silence.

```python
[[], # batch 0, empty: no shifts found in this sample
 [(947, 960, 1)], # batch 1
 [],
 [(500, 523, 0), (549, 554, 1), (869, 901, 1)], # batch 3: three shifts found in this sample
 [(387, 399, 0)],
 [(722, 733, 0), (758, 765, 1)],
 [(436, 451, 0),
  (529, 558, 0),
  (918, 939, 0),
  (378, 384, 1),
  (483, 490, 1),
  (780, 806, 1)],
 [],
 [(485, 514, 1)],
 [(848, 867, 1)]]
```

These can then be used to extract the probability output of the model.

```python
# iterate over the batch samples
for b in range(batch_size):
    # Hold vs Shift
    # We extract the hold/shift probabilities from the model output (p_now, p_future, p_all, etc)
    # Hold label = 0   |   Shift label = 1
    for start, end, speaker in events["shift"][b]:
        # Extract the probabilities over the shift-region
        pshift = p_now[b, start:end]

        # we want a shift prediction to be close to one
        # so we have to consider who the next-speaker is
        # to get appropriate values
        if speaker:
            pshift = 1 - pshift
        # e.g. store the prediction values
        # and the targets (label=1)
        preds["hs"].append(pshift)
        targets["hs"].append(torch.ones_like(pshift))

        # Similarly for HOLDS
        # However now the hold label=0
        # Given that the next-speaker is 'speaker' we must 'flip' the probability
        # i.e. a hold guess -> 1 - probability-of-next-speaker
        # therefore a high probability e.g. 0.9 for the next-speaker -> 0.1 hold prob
        for start, end, speaker in events["hold"][b]:
            phold = 1 - p_now[b, start:end]
            if speaker:
                phold = 1 - phold
            preds["hs"].append(phold)
            # Labels for hold are 0
            targets["hs"].append(torch.zeros_like(phold))

```

The `EventConfig`

```python
@dataclass
class EventConfig:
    min_context_time: float = 3
    metric_time: float = 0.2
    metric_pad_time: float = 0.05  # seconds into silence
    max_time: int = 20  # Max time for event (not extending passed model prediction) choose large number for entire dialogs e.g. 99999
    frame_hz: int = 50
    equal_hold_shift: bool = True  # extract equal amounts of hold/shift
    prediction_region_time: float = 0.5  # predict turn-shift

    # Shift/Hold
    sh_pre_cond_time: float = 1.0  # only one speaker prior to silence
    sh_post_cond_time: float = 1.0 # only one speaker post silence
    sh_prediction_region_on_active: bool = True  # predict shift while inside activity

    # Backchannel
    bc_pre_cond_time: float = 1.0  # only one speaker prior to silence
    bc_post_cond_time: float = 1.0 # only one speaker post silence
    bc_max_duration: float = 1.0 # maximum duration to be considered backchannel
    bc_negative_pad_left_time: float = 1.0
    bc_negative_pad_right_time: float = 2.0

    # Long/Short
    long_onset_region_time: float = 0.2  # how far into activity to predict
    long_onset_condition_time: float = 1.0  # long > n seconds 

```


