"""
plot_training_metrics.py
========================
Plots all training metrics from a CSV file with the format:
    iteration, loss, q_mean, td_error, action_entropy, reward,
    avg_reward, global_reward, avg_global_reward, eps

Usage:
    python plot_training_metrics.py                       # uses 'training_log.csv'
    python plot_training_metrics.py my_run.csv            # uses a specific file
    python plot_training_metrics.py my_run.csv --save     # saves figure to PNG
    python plot_training_metrics.py my_run.csv --show     # try to open a GUI window
"""

import sys
import os
import argparse

# ── backend selection ─────────────────────────────────────────────────────────
# Must happen BEFORE importing pyplot.
# If DISPLAY / WAYLAND_DISPLAY are unset (headless container), force Agg so
# matplotlib never tries to load Qt/xcb.  With --show the user can override.
_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if not _has_display:
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_tensor(value):
    """Convert 'tensor(0.9990)' strings (or plain floats) to float."""
    s = str(value).strip()
    if s.startswith("tensor("):
        inner = s[len("tensor("):-1]
        return float(inner)
    return float(s)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = df.columns.str.strip()
    # Parse eps column (may contain 'tensor(...)' strings)
    if "eps" in df.columns:
        df["eps"] = df["eps"].apply(parse_tensor)
    return df


# ── plot config ───────────────────────────────────────────────────────────────

# Each entry: (column_name, display_label, colour)
PANELS = [
    ("loss",             "Loss",               "#e05c5c"),
    ("q_mean",           "Q Mean",             "#5c9ee0"),
    ("td_error",         "TD Error",            "#e0a35c"),
    ("action_entropy",   "Action Entropy",      "#7ec87e"),
    ("reward",           "Reward (per step)",   "#c87eb8"),
    ("avg_reward",       "Avg Reward",          "#c8a87e"),
    ("global_reward",    "Global Reward",       "#5cc8c8"),
    ("avg_global_reward","Avg Global Reward",   "#8c8ce0"),
    ("eps",              "Epsilon (ε)",         "#aaaaaa"),
]


def build_figure(df: pd.DataFrame) -> plt.Figure:
    n_cols = 3
    available = [(col, lbl, clr) for col, lbl, clr in PANELS if col in df.columns]
    n_panels = len(available)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows), facecolor="#0f1117")
    fig.suptitle(
        "Training Metrics",
        fontsize=20, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
    )

    x = df["iteration"].values

    for idx, (col, label, color) in enumerate(available):
        row, col_idx = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col_idx])

        y = df[col].values

        # Shaded area under curve
        ax.fill_between(x, y, alpha=0.15, color=color)
        ax.plot(x, y, color=color, linewidth=1.8, zorder=3)

        # Zero reference line (only when data crosses zero)
        if y.min() < 0 < y.max():
            ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.3)

        # Moving average overlay (window = 10 % of data, min 3)
        window = max(3, len(y) // 10)
        if len(y) >= window:
            ma = pd.Series(y).rolling(window, min_periods=1).mean().values
            ax.plot(x, ma, color="white", linewidth=1.0,
                    linestyle="--", alpha=0.55, label=f"MA({window})")
            ax.legend(fontsize=7, loc="upper right",
                      framealpha=0.25, labelcolor="white")

        # Styling
        ax.set_facecolor("#1a1d27")
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Iteration", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(True, color="#333344", linewidth=0.5, linestyle=":")

    # Hide unused subplots
    for idx in range(n_panels, n_rows * n_cols):
        row, col_idx = divmod(idx, n_cols)
        fig.add_subplot(gs[row, col_idx]).set_visible(False)

    fig.patch.set_facecolor("#0f1117")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="training_log.csv",
        help="Path to the CSV file (default: training_log.csv)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the figure as a PNG instead of (or in addition to) showing it",
    )
    parser.add_argument(
        "--out",
        default="training_metrics.png",
        help="Output PNG filename when --save is used (default: training_metrics.png)",
    )
    args = parser.parse_args()

    print(f"Loading '{args.csv_file}' …")
    try:
        df = load_csv(args.csv_file)
    except FileNotFoundError:
        sys.exit(f"Error: file '{args.csv_file}' not found.")
    except Exception as exc:
        sys.exit(f"Error reading CSV: {exc}")

    print(f"  {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    fig = build_figure(df)

    if args.save:
        fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Figure saved to '{args.out}'")

    plt.show()


if __name__ == "__main__":
    main()