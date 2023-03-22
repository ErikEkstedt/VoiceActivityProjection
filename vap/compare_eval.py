import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    df = pd.read_csv(
        # "runs_evaluation/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56_switchboard_fisher/score.csv"
        # "runs_evaluation/VapGPT_50Hz_ad20s_134-epoch4-val_2.56_switchboard_fisher/score_nb-1600.csv"
        "runs_evaluation/VapGPT_50Hz_ad20s_134-epoch4-val_2.56_switchboard_fisher/score_fp16.csv"
    )

    df.plot()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ticks, labels = [], []
    ii = 0
    for c in df.columns:
        if "loss" in c:
            continue
        ax.bar(ii, df[c])
        labels.append(c.replace("test_", ""))
        ticks.append(ii)
        ii += 1
    ax.set_ylim([0, 1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    plt.tight_layout()
    plt.show()
