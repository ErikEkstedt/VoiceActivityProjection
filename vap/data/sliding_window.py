import torch
from typing import Any
from vap.utils.utils import get_vad_list_subset

import pandas as pd
import json
import os
import csv
import subprocess
from typing import List, Any, Dict


VAD_LIST = list[list[list[float]]]


def get_vad_list_lims(vad_list: VAD_LIST) -> tuple[float, float]:
    start = max(vad_list[0][0][0], vad_list[1][0][0])
    end = max(vad_list[0][-1][-1], vad_list[1][-1][-1])
    return start, end


# This function uses 'duration' twice, so duration - duration always results in just 1 object
# def get_sliding_windows(
#     vad_list: VAD_LIST, duration: float = 20, overlap: float = 5
# ) -> list[float]:
#     """
#     Sliding windows of a session
#     """
#     # Get boundaries from vad
#     start, end = get_vad_list_lims(vad_list)

#     # get valid starting times for sliding window
#     duration = end - start
#     step = duration - overlap
#     n_clips = int((duration - duration) / step + 1)
# starts = torch.arange(start, end, step)[:n_clips].tolist()
#     return starts


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
    vad_list: VAD_LIST, duration: float = 20, overlap: float = 5, horizon: float = 2
) -> list[dict[str, Any]]:
    """
    Get overlapping samples from a vad_list of a conversation
    """
    samples = []
    starts = get_sliding_windows(vad_list, duration, overlap)
    for start in starts:
        end = start + duration
        vad_list_subset = get_vad_list_subset(vad_list, start, end + horizon)
        samples.append(
            {
                "start": start,
                "end": end,
                "vad_list": vad_list_subset,
            }
        )
    return samples


def create_csv(bucket_name: str):
    # Get a list of all files in the bucket
    files = (
        subprocess.check_output(["gsutil", "ls", "gs://{}".format(bucket_name)])
        .decode()
        .split()
    )

    # Define the output directory
    output_dir = "example/data/"

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write URLs to the CSV file
    with open(os.path.join(output_dir, f"{bucket_name}.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        for file_path in files:
            # Extract only the filename from the full path
            file_name = os.path.basename(file_path)
            writer.writerow(
                ["https://storage.googleapis.com/{}/{}".format(bucket_name, file_name)]
            )


def generate_vap_csv(bucket_name: str, json_path: str) -> None:
    create_csv(bucket_name)

    # Define the output directory
    output_dir = "example/data/"

    # Load the csv file into a pandas DataFrame
    df = pd.read_csv(os.path.join(output_dir, f"{bucket_name}.csv"), names=["url"])

    # Create a new list that will contain all new rows
    all_results = []

    # Iterate over the DataFrame
    for i, row in df.iterrows():
        # Get the file name from the url (without extension)
        file_name = os.path.splitext(os.path.basename(row["url"]))[0]
        print(file_name)

        # Construct the path to the corresponding json file
        json_file = os.path.join(json_path, file_name + ".json")

        # If the json file exists, read it and process its contents
        if os.path.isfile(json_file):
            with open(json_file) as f:
                vad_list = json.load(f)
                results = sliding_window(vad_list)

                print(f"Processing {json_file}...")
                print(f"{len(results)} results obtained from sliding_window function.")
                print(results)

                # Loop over the results and append each as a new row to the all_results list
                for result in results:
                    all_results.append(
                        {
                            "url": row["url"],
                            "start": result["start"],
                            "end": result["end"],
                            "vad_list": json.dumps(result["vad_list"]),
                        }
                    )

    # Create a DataFrame from all_results
    df_results = pd.DataFrame(all_results)

    # Write the DataFrame back to the csv
    df_results.to_csv(
        os.path.join(output_dir, f"{bucket_name}_with_json.csv"), index=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a CSV file with VAP information."
    )
    parser.add_argument(
        "bucket_name", type=str, help="Name of the Google Cloud Storage bucket"
    )
    parser.add_argument(
        "json_path", type=str, help="Path to the directory containing the JSON files"
    )

    args = parser.parse_args()

    generate_vap_csv(args.bucket_name, args.json_path)
