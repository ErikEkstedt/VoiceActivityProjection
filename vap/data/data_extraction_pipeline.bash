#/bin/bash
#
# 1. Audio and VAD Directories
WAV_DIR=$1
VAD_DIR=$2
AUDIO_VAD_CSV="data/audio_vad.csv"

SPLIT_DIR="data/splits"
TRAIN="data/splits/sliding_window_train.csv"
VALIDATION="data/splits/sliding_window_val.csv"
CLASSIFICATION="data/classification/test_hs.csv"

# 2. a) Create csv file with audio and vad information
python vap/data/create_audio_vad_csv.py --audio_dir $WAV_DIR --vad_dir $VAD_DIR --output $AUDIO_VAD_CSV

# 2. b) Create Splits
python vap/data/create_splits.py --csv $AUDIO_VAD_CSV --output_dir $SPLIT_DIR

# 3. Create Data (TRAIN/VAL)
python vap/data/create_sliding_window_dset.py --audio_vad_csv $SPLIT_DIR/train.csv --output $TRAIN
python vap/data/create_sliding_window_dset.py --audio_vad_csv $SPLIT_DIR/val.csv --output $VALIDATION

# 5. Create Classification Data
python vap/data/dset_event.py --audio_vad_csv $SPLIT_DIR/test.csv --output $CLASSIFICATION
