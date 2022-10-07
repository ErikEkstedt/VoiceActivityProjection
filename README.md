# VoiceActivityProjection

Voice Activity Projection is a Self-supervised objective for Turn-taking Events.

1. [DEMO page](https://erikekstedt.github.io/VAP/)
2. [Voice Activity Projection: Self-supervised Learning of Turn-taking Events](https://arxiv.org/abs/2205.09812)
    * Paper introducing the VAP model and comparing it to prior work [Skantze 2016]()
    * **Accepted at INTERSPEECH2022**
3. [How Much Does Prosody Help Turn-taking? Investigations using Voice Activity Projection Models](https://arxiv.org/abs/2209.05161)
    * Analysis inspired by Psycholinguistics Prosodic analysis
    * Winner of **Best Paper Award** at **SIGDIAL2022**

Content
* [Usage](#usage)
  * [Train](#train)
  * [Evaluation](#evaluation)
* [Installation](#installation)
* [Citation](#citation)


The model is a GPT-like transformer model, using [AliBI attention](https://ofir.io/train_short_test_long.pdf), which operates on pretrained speech representation (extract by submodule defined/trained/provided by [CPC facebookresearch](https://github.com/facebookresearch/CPC_audio)).

## Usage

The `run.py` script loads a pretrained model and evaluates on a sample (`waveform` + `text_grid_name.TextGrid` or `vad_list_name.json`).

* Using defaults: `python run.py`
* Custom run requires a audio file `sample.wav` and **either** a `vad_list_name.json` or `text_grid_name.TextGrid`
* See `examples/` folder for model-checkpoint, input data format etc.

```bash
python run.py \
  -c example/cpc_48_50hz_15gqq5s5.ckpt \
  -w example/student_long_female_en-US-Wavenet-G.wav \ # waveform
  -v example/vad_list.json \ # Required if model.mono=True else Optional
  -o VAP_OUTPUT.json  # output file

  # -tg example/student_long_female_en-US-Wavenet-G.TextGrid \ # OPTIONAL
```

### Train


**WARNING: Requires access to `Fisher` and/or `Switchboard` datasets.**
(DataModules, Datasets, etc are implemented in [datasets_turntaking](https://github.com/ErikEkstedt/datasets_turntaking))
```bash
python vap/train.py data.datasets=['switchboard','fisher']
```

### Evaluation

Evaluation over test-set.

**WARNING: Requires access to `Fisher` and/or `Switchboard` datasets.**
(DataModules, Datasets, etc are implemented in [datasets_turntaking](https://github.com/ErikEkstedt/datasets_turntaking))

** WARNING: Using hydra (notice the '+' flag or the absence of a flag).**
```bash
python vap/evaluation.py \
  +checkpoint_path=/full/path/checkpoint.ckpt \
  data.num_workers=4 \
  data.batch_size=16
```

----------------------------

## Installation

* Create conda env: `conda create -n voice_activity_projection python=3.9`
  - source env: `conda source voice_activity_projection`
* PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
* Dependencies:
  * **VAP**: Voice Activity Projection multi-purpose "head".
    * Install [`vap_turn_taking`](https://github.com/ErikEkstedt/vap_turn_taking)
      * `git clone https://github.com/ErikEkstedt/vap_turn_taking.git`
      * cd to repo, and install dependencies: `pip install -r requirements.txt`
      * Install: `pip install -e .`
  * **DATASET**
    * Install [datasets_turntaking](https://github.com/ErikEkstedt/datasets_turntaking)
      * `git clone https://github.com/ErikEkstedt/datasets_turntaking.git`
      * cd to repo, and install dependencies: `pip install -r requirements.txt`
      * Install repo: `pip install -e .`
    * **WARNING:** Requires [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) and/or [Fisher](https://catalog.ldc.upenn.edu/LDC2004S13) data!
* Install **`voice_activity_projection`** (this repo):
  * cd to root directory and run:
    * `pip install -r requirements.txt`
    * `pip install -e .`

-------------------------

### Paper

Event settings used in [Voice Activity Projection: Self-supervised Learning of Turn-taking Events](https://arxiv.org/abs/2205.09812).
* The event settings used in the paper are included in `vap/conf/events/events.json`.
  - See paper Section 3
* Events are extracted using [`vap_turn_taking`](https://github.com/ErikEkstedt/vap_turn_taking)

```python
from vap.utils import read_json

event_settings = read_json("vap/conf/events/events.json")
hs_kwargs = event_settings['hs']
bc_kwargs = event_settings['bc']
metric_kwargs = event_settings['metric']
```

```json
{
  "hs": {
    "post_onset_shift": 1,
    "pre_offset_shift": 1,
    "post_onset_hold": 1,
    "pre_offset_hold": 1,
    "non_shift_horizon": 2,
    "metric_pad": 0.05,
    "metric_dur": 0.1,
    "metric_pre_label_dur": 0.5,
    "metric_onset_dur": 0.2
  },
  "bc": {
    "max_duration_frames": 1.0,
    "pre_silence_frames": 1.0,
    "post_silence_frames": 2.0,
    "min_duration_frames": 0.2,
    "metric_dur_frames": 0.2,
    "metric_pre_label_dur": 0.5
  },
  "metric": {
    "pad": 0.05,
    "dur": 0.1,
    "pre_label_dur": 0.5,
    "onset_dur": 0.2,
    "min_context": 3.0
  }
}
```


## Citation

**TBD: citation from actual proceedings (not yet availabe)**

```latex
@article{ekstedtVapModel2022,
  title = {Voice Activity Projection: Self-supervised Learning of Turn-taking Events},
  url = {https://arxiv.org/abs/2205.09812},
  author = {Ekstedt, Erik and Skantze, Gabriel},
  journal = {arXiv},
  year = {2022},
}
```

```latex
@article{ekstedtVapProsody2022,
  title = {How Much Does Prosody Help Turn-taking? Investigations using Voice Activity Projection Models},
  url = {https://arxiv.org/abs/2209.05161},
  author = {Ekstedt, Erik and Skantze, Gabriel},
  journal = {arXiv},
  year = {2022},
}
```
