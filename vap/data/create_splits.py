from argparse import ArgumentParser
from pathlib import Path
from vap.data.datamodule import load_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/audio_vad.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.05)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    df = load_df(args.csv)
    N = len(df)
    train_size = int(N * args.train_size)
    val_size = int(N * args.val_size)
    test_size = N - train_size - val_size

    # Sample splits
    train_df = df.sample(n=train_size, random_state=0)
    df = df.drop(train_df.index)
    val_df = df.sample(n=val_size, random_state=0)
    test_df = df.drop(val_df.index)
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    # Save splits
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
    val_df.to_csv(f"{args.output_dir}/val.csv", index=False)
    test_df.to_csv(f"{args.output_dir}/test.csv", index=False)
