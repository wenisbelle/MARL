import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def view(path="logs/recording.pkl"):
    data = pickle.load(open(path, "rb"))
    N, W, H = data["meta"]["num_drones"], data["meta"]["map_width"], data["meta"]["map_height"]
    times, S = data["meta"]["times"], data["states"]
    T = len(times)

    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5.5), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(bottom=0.16)
    ims, selfs, bels, trues = [], [], [], []
    for i, ax in enumerate(axes):
        ims.append(ax.imshow(S[i]["maps"][0].T, origin="lower", extent=[0, W, 0, H],
                             cmap="gray_r", vmin=0, vmax=1))
        ax.set_title(f"Drone {i}"); ax.set_xlim(0, W); ax.set_ylim(0, H)
        # self — blue, so it never blends with the red beliefs
        selfs.append(ax.plot([], [], "o", color="tab:blue", ms=9, zorder=3, label="self")[0])
        # true positions of the OTHER drones — green
        trues.append(ax.plot([], [], "+", color="#2ca02c", ms=12, mew=2.5,
                             zorder=4, label="true others")[0])
        # where THIS drone believes the others are — strong, bold red, on top
        bels.append(ax.scatter([], [], color="#e00000", marker="X", s=180,
                               edgecolors="black", linewidths=1.2,
                               zorder=5, label="believed others"))
    axes[0].legend(loc="upper right", fontsize=8)

    def draw(val):
        f = int(val)
        truth = {i: S[i]["true_pos_cell"][f] for i in range(N)}
        for i in range(N):
            ims[i].set_data(S[i]["maps"][f].T)
            cx, cy = truth[i]; selfs[i].set_data([cx], [cy])
            others = [j for j in range(N) if j != i]
            b = S[i]["beliefs"][f]
            bels[i].set_offsets([[b[j, 0], b[j, 1]] for j in others])
            trues[i].set_data([truth[j][0] for j in others], [truth[j][1] for j in others])
        fig.suptitle(f"t = {times[f]:.0f}s   ({f + 1}/{T})")
        fig.canvas.draw_idle()

    sl = Slider(fig.add_axes([0.15, 0.05, 0.7, 0.03]), "time", 0, T - 1, valinit=0, valstep=1)
    sl.on_changed(draw); draw(0); plt.show()

if __name__ == "__main__":
    view()