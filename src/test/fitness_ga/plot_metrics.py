import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(path="logs/recording.pkl"):
    m = pickle.load(open(path, "rb"))["meta"]
    t, gu, uv = m["times"], m["global_uncertainty"], m["unvisited"]
    total = m["map_width"] * m["map_height"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    ax1.plot(t, gu, color="tab:blue", lw=1.6)
    ax1.set_ylabel("global map uncertainty\n(sum over cells)")
    ax1.grid(True, color="0.92", lw=0.5)

    ax2.plot(t, uv, color="tab:red", lw=1.6)
    ax2.set_ylabel("cells never visited\n(by any drone)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(0, total)
    ax2.grid(True, color="0.92", lw=0.5)

    # right-hand axis: same data shown as % coverage, stays synced automatically
    sec = ax2.secondary_yaxis(
        "right",
        functions=(lambda u: 100 * (1 - u / total), lambda c: total * (1 - c / 100)),
    )
    sec.set_ylabel("coverage (%)")

    fig.suptitle("Global uncertainty and coverage over time")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metrics()