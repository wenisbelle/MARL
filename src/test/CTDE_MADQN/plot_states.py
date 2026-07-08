import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import ListedColormap


def mask_to_grid(mask, cx, cy, A, W, H):
    """(W,H) bool grid of valid target cells — same idx->cell geometry as the protocol."""
    grid = np.zeros((W, H), dtype=bool)
    if mask is None:
        return grid
    cxi, cyi = int(cx), int(cy)   # int(_to_cell(...)) == protocol's get_current_cell()
    for idx in np.flatnonzero(mask):
        row, col = idx // A, idx % A
        x = (row + 0.5) / A
        y = (col + 0.5) / A
        tr = int(cxi + (x - 0.5) * A)
        tc = int(cyi + (y - 0.5) * A)
        if 0 <= tr < W and 0 <= tc < H:
            grid[tr, tc] = True
    return grid


def view(path="logs/recording.pkl"):
    data = pickle.load(open(path, "rb"))
    meta = data["meta"]
    N, W, H = meta["num_drones"], meta["map_width"], meta["map_height"]
    times, S = meta["times"], data["states"]
    A = int(meta.get("action_map_size", 10))
    T = len(times)

    glob = np.min(np.stack([S[i]["maps"] for i in range(N)]), axis=0)   # (T, W, H)
    masks = {i: S[i].get("masks") for i in range(N)}
    show_mask = [False]
    mask_cmap = ListedColormap(["#ff7f0e"])

    fig = plt.figure(figsize=(4.5 * N, 8))
    gs = fig.add_gridspec(2, N, height_ratios=[1.3, 1.0])
    ax_glob = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, i]) for i in range(N)]
    fig.subplots_adjust(bottom=0.13, hspace=0.25)

    # global panel (best-known uncertainty per cell), with each drone's self position
    im_glob = ax_glob.imshow(glob[0].T, origin="lower", extent=[0, W, 0, H],
                             cmap="gray_r", vmin=0, vmax=1)
    ax_glob.set_title("Global map (best-known per cell)")
    ax_glob.set_xlim(0, W); ax_glob.set_ylim(0, H)
    glob_dots = ax_glob.scatter([], [], color="tab:blue", marker="o", s=40,
                                edgecolors="black", linewidths=0.8, zorder=3)

    ims, selfs, bels, trues, ovs = [], [], [], [], []
    for i, ax in enumerate(axes):
        ims.append(ax.imshow(S[i]["maps"][0].T, origin="lower", extent=[0, W, 0, H],
                             cmap="gray_r", vmin=0, vmax=1))
        # mask overlay — orange, translucent, above the map but below the markers
        ovs.append(ax.imshow(np.ma.masked_all((H, W)), origin="lower", extent=[0, W, 0, H],
                             cmap=mask_cmap, vmin=0, vmax=1, alpha=0.30, zorder=2))
        ax.set_title(f"Drone {i}"); ax.set_xlim(0, W); ax.set_ylim(0, H)
        selfs.append(ax.plot([], [], "o", color="tab:blue", ms=9, zorder=4, label="self")[0])
        trues.append(ax.plot([], [], "+", color="#2ca02c", ms=12, mew=2.5,
                             zorder=5, label="true others")[0])
        bels.append(ax.scatter([], [], color="#e00000", marker="X", s=180,
                               edgecolors="black", linewidths=1.2,
                               zorder=6, label="believed others"))
    axes[0].legend(loc="upper right", fontsize=8)

    def draw(val):
        f = int(val)
        truth = {i: S[i]["true_pos_cell"][f] for i in range(N)}
        im_glob.set_data(glob[f].T)
        gp = [truth[i] for i in range(N) if np.isfinite(truth[i][0])]
        glob_dots.set_offsets(gp if gp else np.empty((0, 2)))
        for i in range(N):
            ims[i].set_data(S[i]["maps"][f].T)
            cx, cy = truth[i]; selfs[i].set_data([cx], [cy])
            others = [j for j in range(N) if j != i]
            b = S[i]["beliefs"][f]
            bels[i].set_offsets([[b[j, 0], b[j, 1]] for j in others])
            trues[i].set_data([truth[j][0] for j in others], [truth[j][1] for j in others])
            if show_mask[0] and np.isfinite(cx) and masks[i] is not None:
                g = mask_to_grid(masks[i][f], cx, cy, A, W, H)
                ovs[i].set_data(np.ma.masked_where(~g.T, np.ones_like(g.T, dtype=float)))
            else:
                ovs[i].set_data(np.ma.masked_all((H, W)))
        fig.suptitle(f"t = {times[f]:.0f}s   ({f + 1}/{T})")
        fig.canvas.draw_idle()

    sl = Slider(fig.add_axes([0.13, 0.05, 0.62, 0.03]), "time", 0, T - 1, valinit=0, valstep=1)
    sl.on_changed(draw)
    chk = CheckButtons(fig.add_axes([0.8, 0.035, 0.13, 0.05]), ["show masks"], [False])
    def toggle(_):
        show_mask[0] = not show_mask[0]
        draw(int(sl.val))
    chk.on_clicked(toggle)
    draw(0); plt.show()


if __name__ == "__main__":
    view()