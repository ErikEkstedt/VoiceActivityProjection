from argparse import ArgumentParser
from pathlib import Path
from vap.data.datamodule import load_df
from vap.utils.utils import read_txt


def process_file(args, split):
    file_path = getattr(args, f"{split}_file")
    session_names = read_txt(file_path)
    split_df = df[df["session"].isin(session_names)]
    r = len(split_df) / N
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split_df.to_csv(f"{args.output_dir}/{split}.csv", index=False)
    print(f"Saved {split.capitalize()} file {len(session_names)}/{N} = {100*r:.1f}%")
    print(f"-> {args.output_dir}/{split}.csv")
    if len(split_df) != len(session_names):
        print(f"Warning: {len(split_df)} != {len(session_names)}")
        print(f"Missed {len(session_names) - len(split_df)} files")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/audio_vad.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)

    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    df = load_df(args.csv)
    N = len(df)

    # Add session column
    df["session"] = df["audio_path"].apply(lambda x: Path(x).stem)

    # If any file path were provided we simply extract those
    if args.train_file or args.val_file or args.test_file:
        if args.train_file:
            process_file(args, "train")

        if args.val_file:
            process_file(args, "val")

        if args.test_file:
            process_file(args, "test")
    else:
        train_size = int(N * args.train_size)
        val_size = int(N * args.val_size)
        test_size = N - train_size - val_size

        # Sample splits
        train_df = df.sample(n=train_size, random_state=0)
        df = df.drop(train_df.index)
        val_df = df.sample(n=val_size, random_state=0)
        test_df = df.drop(val_df.index)
        # Save splits
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
        val_df.to_csv(f"{args.output_dir}/val.csv", index=False)
        test_df.to_csv(f"{args.output_dir}/test.csv", index=False)
        print(f"Saved {len(train_df)} -> ", f"{args.output_dir}/train.csv")
        print(f"Saved {len(val_df)} -> ", f"{args.output_dir}/val.csv")
        print(f"Saved {len(test_df)} -> ", f"{args.output_dir}/test.csv")
