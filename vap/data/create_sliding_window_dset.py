from pathlib import Path
from typing import Any
import torch
import pandas as pd
import tqdm

from vap.utils.utils import read_json, get_vad_list_subset, invalid_vad_list

VAD_LIST = list[list[list[float]]]


def get_vad_list_lims(vad_list: VAD_LIST) -> tuple[float, float]:
    start = max(vad_list[0][0][0], vad_list[1][0][0])
    end = max(vad_list[0][-1][-1], vad_list[1][-1][-1])
    return start, end


def get_sliding_windows(
    vad_list: list, window_duration: float = 20, overlap: float = 5
) -> list:
    """
    Sliding windows of a session
    """
    # Get boundaries from vad
    start, end = get_vad_list_lims(vad_list)

    # Define the step size
    step = window_duration - overlap

    # Calculate the number of windows
    n_windows = int((end - start - window_duration) / step) + 1

    # Get the starting times for each window
    starts = torch.arange(start, end, step)[:n_windows].tolist()
    # starts = [start + i * step for i in range(n_windows)]

    return starts


def sliding_window(
    vad_list: VAD_LIST,
    audio_path: str,
    duration: float = 20,
    overlap: float = 5,
    horizon: float = 2,
) -> list[dict[str, Any]]:
    """
    Get overlapping samples from a vad_list of a conversation
    """

    starts = get_sliding_windows(vad_list, duration, overlap)
    samples = []
    for start in starts:
        end = start + duration
        vad_list_subset = get_vad_list_subset(vad_list, start, end + horizon)
        samples.append(
            {
                "session": Path(audio_path).stem,
                "audio_path": audio_path,
                "start": start,
                "end": end,
                "vad_list": vad_list_subset,
            }
        )
    return samples


def main(args):
    df = pd.read_csv(args.audio_vad_csv)
    data = []
    skipped = []
    for _, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Create sliding window dataset"
    ):
        vad_list = read_json(row.vad_path)
        if invalid_vad_list(vad_list):
            skipped.append(row.vad_path)
            continue

        session_samples = sliding_window(
            vad_list=vad_list,
            audio_path=row.audio_path,
            duration=args.duration,
            overlap=args.overlap,
            horizon=args.horizon,
        )
        data.extend(session_samples)

    if len(skipped) > 0:
        print("Skipped: ", len(skipped))
        with open("/tmp/sliding_window_skipped_vad.txt", "w") as f:
            f.write("\n".join(skipped))
        print("See -> /tmp/sliding_window_skipped_vad.txt")
        print()

    print(f"Extracted {len(data)} segments from {len(df)} session.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--audio_vad_csv", type=str, default="data/audio_vad.csv")
    parser.add_argument("--output", type=str, default="data/sliding_window_dset.csv")
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--overlap", type=float, default=5)
    parser.add_argument("--horizon", type=float, default=2)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    main(args)
