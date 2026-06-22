import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_decisions(path="logs/recording.pkl", drone_id=0, fade_after=120.0):
    data = pickle.load(open(path, "rb"))
    W, H = data["meta"]["map_width"], data["meta"]["map_height"]
    decs = data["decisions"].get(drone_id, [])
    if not decs:
        print(f"No decisions recorded for drone {drone_id}"); return

    ts  = np.array([d["t"] for d in decs], dtype=float)
    cur = np.array([d["current_cell"] for d in decs], dtype=float)
    tgt = np.array([d["target_cell"]  for d in decs], dtype=float)
    n   = len(decs)

    # arrow tails at the current cell, components pointing to the target
    X, Y = cur[:, 0], cur[:, 1]
    U, V = tgt[:, 0] - cur[:, 0], tgt[:, 1] - cur[:, 1]
    idx  = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.5, 8))
    fig.subplots_adjust(bottom=0.16)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.grid(True, color="0.9", lw=0.5)

    norm = plt.Normalize(0, fade_after)
    cmap = plt.cm.RdYlGn_r            # recent = green, old = red

    # one quiver for all arrows; NaN U/V hides the ones not drawn yet
    q = ax.quiver(X, Y, np.full(n, np.nan), np.full(n, np.nan), np.zeros(n),
                  cmap=cmap, norm=norm, angles="xy", scale_units="xy", scale=1,
                  width=0.005, zorder=2)

    # highlight the current decision so the latest move is unmistakable
    cur_dot, = ax.plot([], [], "o", color="black", ms=6,  zorder=4, label="current cell")
    tgt_dot, = ax.plot([], [], "*", color="black", ms=15, zorder=4, label="latest target")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(q, ax=ax, label="decision age (s)")

    def draw(val):
        f = int(val)
        age = ts[f] - ts                          # how old each decision is *now*
        Uf = np.where(idx <= f, U, np.nan)        # blank everything after frame f
        Vf = np.where(idx <= f, V, np.nan)
        q.set_UVC(Uf, Vf, np.clip(age, 0, fade_after))
        cur_dot.set_data([cur[f, 0]], [cur[f, 1]])
        tgt_dot.set_data([tgt[f, 0]], [tgt[f, 1]])
        ax.set_title(f"Drone {drone_id}  —  decision {f + 1}/{n}   t = {ts[f]:.0f}s")
        fig.canvas.draw_idle()

    if n == 1:                                    # Slider needs valmin < valmax
        draw(0); plt.show(); return

    sl = Slider(fig.add_axes([0.15, 0.05, 0.7, 0.03]), "decision",
                0, n - 1, valinit=0, valstep=1, valfmt="%d")
    sl.on_changed(draw)
    draw(0)
    plt.show()


if __name__ == "__main__":
    plot_decisions(drone_id=0)