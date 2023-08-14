# Data


1. Access to **Audio** and **VAD** information
    - Resample audio to **16kHz**  (Encoder sample rate).
    - Change to **'.wav'** extension.
        - WARNING: '.mp3' files are slow when loading sub-segments directly from the whole audio.
        - Using [`torchaudio.load(...)`](https://pytorch.org/audio/stable/torchaudio.html#i-o-functionalities)
    - VAD information as `VAD_LIST` in `session_name.json`:
        ```python
        [
            [[0.1, 2.11], [2.5, 3.9], ...],
            [[2.2, 2.45], [4.8, 5.2], ...],
        ]
        ```
    - [start-time, end-time] for each IPU / VAD segment of activity
    - one list for each corresponding audio channel
2. Create csv with `audio_path` and `vad_path`
    * Folder with wav-files `/PATH/TO/WAV_FILES_DIR`
    * Folder with vad_list json-files `/PATH/TO/VAD_LIST_DIR`
    * Run [`vap/data/create_audio_vad_csv.py`](vap/data/create_audio_vad_csv.py):
    ```bash
    python vap/data/create_audio_vad_csv.py \
        --audio_dir /PATH/TO/WAV_FILES_DIR \  # recursive glob
        --vad_dir PATH/TO/VAD_LIST_DIR \  # matching the names of audio
        --output data/audio_vad.csv
    ```
    * Example `--audio_vad.csv`
    ```csv
    audio_path,vad_path
    /audio/007/fe_03_00785.wav,/vad_lists/fe_03_00785.json
    /audio/007/fe_03_00705.wav,/vad_lists/fe_03_00705.json
    ```
    * [Optional] Create splits (train/val/test)
        - Run [`vap/data/create_splits.py`](vap/data/create_splits.py):
        ```bash
        python vap/data/create_splits.py \
            --csv data/audio_vad.csv \
            --output_dir data/splits \
            --train_size 0.8 \
            --val_size 0.15 \
            --test_size 0.05
        ```
4. Create dataset csv (sliding-window)
    - Extract overlapping segments of
    - Run: [`vap/data/create_sliding_window`](vap/data/create_sliding_window.py)
    ```bash
    python vap/data/create_sliding_window.py \
        --audio_vad_csv data/audio_vad.csv \
        --output data/sliding_window_dset.csv \
        --duration 20 \
        --overlap 5 \
        --horizon 2 # the prediction horizon of VAP
    ```
5. Sanity check: `VAPDataset` and `VAPDataModule` ([`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#))
    - Training requires at least a training and a validation dataset
    - Run [`vap/data/datamodule.py`](vap/data/datamodule.py)
    - Iterate over batches
    ```bash
    python vap/data/datamodule.py \
        --csv /data/sliding_window_dset.csv \
        --batch_size 4 \
        --num_workers 8 \ # number of cpu cores
        --prefetch_factor 2 \
    ```
    - Visualize a single datapoint from the dataset:
    ```bash
    python vap/data/datamodule.py \
        --csv /data/sliding_window_dset.csv \
        --single
    ```
6. Event classification dataset
    - [`vap/data/dset_event.py`](vap/data/dset_event.py)
    - Run:
    ```bash
    python vap/data/dset_event.py \
        --audio_vad_csv /data/audio_vad.csv \
        --output data/classification/audio_vad_hs.csv \
        --pre_cond_time 1 \
        --post_cond_time 2 \
        --min_silence_time 0.1
    ```

## 3. Dataset CSV
To train the model using the included `LightningDataModule` we assume that we have csv-files where each row defines a sample.
* audio_path
    - `PATH/TO/AUDIO.wav`
* start: float, definining the start time of the sample
    - `0.0`
* end: float, definining the end time of the sample
    - `20.0`
* [Optional] session: str, the name of the sample session
    - `4637`
* [Optional] dataset: str, the name of the dataset
    - `switchboard`
* vad_list: a list containing the voice-activity start/end-times inside of the `start`/`end` times of the row-sample
    * WARNING: because the model train to predict the next 2s (by default) the VAD-list here actually spans 2s longer than the audio.
    * end-start = 20 -> vad list covers 22 seconds
    * Otherwise the last two seconds can't be trained on....

VAD-list example with relative start/end times grounded in the `start`/`end` time of the row sample audio. (Note the last times for the second speaker is over 20s as per the warning above).
```json
[
    [
        [1.16, 1.43],
        [1.73, 3.17],
        [3.27, 3.74],
        [3.94, 4.83],
        [5.41, 6.8]
    ],
    [
        [0.04, 0.28],
        [5.35, 5.83],
        [7.18, 9.3],
        [10.17, 15.12],
        [16.2, 17.17],
        [18.08, 19.03],
        [20.4, 20.75],
        [21.2, 22.0]
    ]
]
```

The top of the csv-file should look something like this (see `example/data/sliding_dev.csv`)
```csv
audio_path,start,end,vad_list,session,dataset
/PATH/AUDIO.wav,0.0,20.0,"[[[1.16, 1.43], [1.73, 3.17], [3.27, 3.74], [3.94, 4.83], [5.41, 6.8]], [[0.04, 0.28], [5.35, 5.83], [7.18, 9.3], [10.17, 15.12], [16.2, 17.17], [18.08, 19.03], [20.4, 20.75], [21.2, 22.0]]]",4637,switchboard
...
```

Or
```csv
session,audio_path,start,end,vad_list,vad_path
fe_03_00785,/audio/007/fe_03_00785.wav,1.86,21.86,"[[[0.95, 1.14], ..., [2.59, 3.44]], [[18.88, 19.97], ..., [20.33, 22.0]]]",/vad_lists/fe_03_00785.json
...
```

## 5. VAPDataset, VAPDataModule

### Dataset
```python

dset = VAPDataset(
    path:str ="example/data/sliding_dev.csv"
    horizon: float = 2,
    sample_rate: int = 16_000,
    frame_hz: int = 50,
    mono: bool = False,
    )

d = dset[0]
# {'session': 'fe_03_00785',
#  'waveform': tensor([[-7.3779e-04, -8.8292e-04, -2.3572e-04,  ..., -2.4323e-05,
#           -2.4002e-04, -1.8580e-04],
#          [-2.6986e-03, -4.8868e-03, -5.4737e-03,  ...,  5.6693e-04,
#            4.8612e-04,  2.3669e-04]]),
#  'vad': tensor([[0., 1.],
#          [0., 1.],
#          [0., 1.],
#          ...,
#          [0., 1.],
#          [0., 1.],
#          [0., 1.]]),
#  'dataset': ''}
```

### DataModule

* [LightningDataModule](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule)


```python
dm = VAPDataModule(
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        horizon: float = 2,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    )
dm.prepare_data()
dm.setup("fit")

print(dm)
print("Train: ", len(dm.train_dset))
print("Val: ", len(dm.val_dset))

dloader = dm.train_dataloader()
for batch in tqdm(dloader, total=len(dloader)):
    pass
```

## 6. Classification (event) Dataset

Run this script to extract events based on vad- and audio-paths. The code also contains `VAPClassificationDataset` used for evaluation.

```bash
python vap/data/dset_event.py \
    --audio_vad_csv /data/audio_vad.csv \
    --output data/classification/audio_vad_hs.csv \
    --pre_cond_time 1 \
    --post_cond_time 2 \
    --min_silence_time 0.1
```

Then run evaluation using the `vap/eval_events.py` code.

```bash
python vap/eval_events.py \
    --checkpoint example/checkpoints/checkpoint.ckpt \
    --csv example/classification_hs.csv \
    --output results/classification_hs_res.csv \
    --plot  # omit if on server
```
