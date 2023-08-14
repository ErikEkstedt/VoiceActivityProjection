from os.path import join, isfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--audio_dir", type=str)
    parser.add_argument("--vad_dir", type=str)
    parser.add_argument("--output", type=str, default="data/audio_vad.csv")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    audio_paths = list(Path(args.audio_dir).rglob("*.wav"))
    data = []
    skipped = []
    for audio_path in tqdm(audio_paths):
        name = audio_path.stem
        vad_path = join(args.vad_dir, f"{name}.json")
        if not isfile(vad_path):
            # print(f"Missing {vad_path}")
            skipped.append(vad_path)
            continue
        data.append(
            {
                "audio_path": str(audio_path),
                "vad_path": vad_path,
            }
        )

    if len(skipped) > 0:
        print("Skipped: ", len(skipped))
        with open("/tmp/create_audio_vad_json_errors.txt", "w") as f:
            f.write("\n".join(skipped))
        print("See -> /tmp/create_audio_vad_json_errors.txt")
        print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print("Saved -> ", args.output)
