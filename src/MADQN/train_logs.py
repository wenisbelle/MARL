"""
training_logger.py — Reusable logger for DQN/MADQN training.

Tracks rolling metrics, prints one-line summaries per iteration, handles
checkpointing, and saves all metrics to a CSV file.
"""

from __future__ import annotations

import os
import csv  # <-- NEW: Imported for saving data
from collections import deque
from typing import Optional

import torch
import matplotlib

# Use a non-blocking backend so plt.show() doesn't pause training.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt


def _avg(xs) -> float:
    """Safe average — returns NaN on empty input so prints stay readable."""
    return sum(xs) / len(xs) if xs else float("nan")


class TrainingLogger:
    """
    Per-iteration logger with rolling reward window, checkpointing, and CSV saving.

    Args:
        checkpoint_dir: where to save model checkpoints and the CSV.
        reward_window:  rolling-average window (in iterations) for "is it improving?".
        checkpoint_every: how often to write a periodic checkpoint and check for "best".
        print_every:    only print summary lines every K iterations (1 = every iter).
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        reward_window: int = 10,
        checkpoint_every: int = 100,
        print_every: int = 1,
        live_plot: bool = False,
        plot_every: int = 1,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.live_plot = live_plot
        self.plot_every = plot_every

        os.makedirs(checkpoint_dir, exist_ok=True)

        # --- NEW: CSV Setup ---
        self.csv_path = os.path.join(checkpoint_dir, "training_metrics.csv")
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # Write the header row
        self.csv_writer.writerow([
            "iteration", "loss", "q_mean", "td_error", "action_entropy",
            "reward", "avg_reward", "global_reward", "avg_global_reward", "eps"
        ])
        # ----------------------

        # Per-iteration buffers
        self._losses: list[float] = []
        self._q_means: list[float] = []
        self._td_errors: list[float] = []
        self._action_entropies: list[float] = []

        # Rolling histories across iterations.
        self._reward_history = deque(maxlen=reward_window)
        self._global_reward_history = deque(maxlen=reward_window)

        # Full-history series for plotting.
        self._iters: list[int] = []
        self._reward_series: list[float] = []
        self._avg_reward_series: list[float] = []
        self._loss_series: list[float] = []
        self._q_series: list[float] = []
        self._ent_series: list[float] = []

        # Best-so-far tracking for checkpoint selection.
        self._best_avg_reward = float("-inf")

        # --- Live plot setup ---
        if self.live_plot:
            plt.ion()
            self._fig, self._axes = plt.subplots(2, 2, figsize=(11, 7))
            self._fig.suptitle("DQN training progress")

            self._line_reward, = self._axes[0, 0].plot([], [], "-", alpha=0.35, label="per-iter")
            self._line_avg,    = self._axes[0, 0].plot([], [], "-", linewidth=2, label=f"rolling avg (window={reward_window})")
            self._axes[0, 0].set_title("Reward")
            self._axes[0, 0].set_xlabel("iteration"); self._axes[0, 0].legend(loc="lower right")

            self._line_loss, = self._axes[0, 1].plot([], [], "-")
            self._axes[0, 1].set_title("TD loss"); self._axes[0, 1].set_xlabel("iteration")

            self._line_q, = self._axes[1, 0].plot([], [], "-")
            self._axes[1, 0].set_title("|Q| mean"); self._axes[1, 0].set_xlabel("iteration")

            self._line_ent, = self._axes[1, 1].plot([], [], "-")
            self._axes[1, 1].set_title("Action entropy"); self._axes[1, 1].set_xlabel("iteration")

            self._fig.tight_layout()
            self._fig.canvas.draw()
            plt.show(block=False)

    def log_update(
        self,
        *,
        loss: float,
        q_values_all: torch.Tensor,
        td_error: float,
    ) -> None:
        """Record metrics from one gradient step."""
        self._losses.append(loss)
        with torch.no_grad():
            self._q_means.append(q_values_all.abs().mean().item())
            self._td_errors.append(td_error)

            action_dim = q_values_all.shape[-1]
            probs = torch.bincount(
                q_values_all.argmax(-1), minlength=action_dim
            ).float()
            probs = probs / probs.sum().clamp(min=1)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            self._action_entropies.append(entropy)

    def log_iteration(
        self,
        *,
        it: int,
        new_count: int,
        buffer_size: int,
        eps: float,
        reward_sample: Optional[torch.Tensor] = None,
        global_reward_sample: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Print a one-line summary, update plots, and log to CSV.
        """
        mean_reward = (
            reward_sample.mean().item() if reward_sample is not None and reward_sample.numel() > 0
            else float("nan")
        )
        global_reward = (
            global_reward_sample.mean().item()
            if global_reward_sample is not None and global_reward_sample.numel() > 0
            else float("nan")
        )

        if mean_reward == mean_reward:
            self._reward_history.append(mean_reward)
        if global_reward == global_reward:
            self._global_reward_history.append(global_reward)

        # Calculate averages for the current iteration
        iter_loss = _avg(self._losses)
        iter_q_mean = _avg(self._q_means)
        iter_td_error = _avg(self._td_errors)
        iter_entropy = _avg(self._action_entropies)

        # --- NEW: Write to CSV ---
        self.csv_writer.writerow([
            it, 
            iter_loss, 
            iter_q_mean, 
            iter_td_error, 
            iter_entropy,
            mean_reward, 
            self.avg_reward, 
            global_reward, 
            self.avg_global_reward, 
            eps
        ])
        self.csv_file.flush()  # Force write to disk immediately so data isn't lost on crash
        # -------------------------

        if it % self.print_every == 0:
            print(
                f"[iter {it:4d}] "
                f"loss={iter_loss:.4f} | "
                f"|Q|={iter_q_mean:.3f} | "
                f"TD_err={iter_td_error:+.3f} | "
                f"ent={iter_entropy:.2f} | "
                f"r={mean_reward:+.3f} (avg={self.avg_reward:+.3f}) | "
                f"R_team={global_reward:+.3f} (avg={self.avg_global_reward:+.3f}) | "
                f"ε={eps:.3f} | "
                f"buf={buffer_size} | "
                f"new={new_count}"
            )

        self._iters.append(it)
        self._reward_series.append(mean_reward)
        self._avg_reward_series.append(self.avg_reward)
        self._loss_series.append(iter_loss)
        self._q_series.append(iter_q_mean)
        self._ent_series.append(iter_entropy)

        if self.live_plot and (it % self.plot_every == 0):
            self._refresh_plot()

        self._losses.clear()
        self._q_means.clear()
        self._td_errors.clear()
        self._action_entropies.clear()

    def _refresh_plot(self) -> None:
        """Update line data in place and rescale axes."""
        self._line_reward.set_data(self._iters, self._reward_series)
        self._line_avg.set_data(self._iters, self._avg_reward_series)
        self._line_loss.set_data(self._iters, self._loss_series)
        self._line_q.set_data(self._iters, self._q_series)
        self._line_ent.set_data(self._iters, self._ent_series)

        for ax in self._axes.ravel():
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        """Save the final plot to disk and close the CSV."""
        # --- NEW: Close the CSV file ---
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()
        # -------------------------------

        if self.live_plot:
            self._fig.savefig(os.path.join(self.checkpoint_dir, "training_curves.png"),
                              dpi=120, bbox_inches="tight")
            plt.ioff()

    def maybe_checkpoint(
        self,
        it: int,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        eps: float,
    ) -> None:
        """Periodic and best-so-far checkpointing."""
        if (it + 1) % self.checkpoint_every != 0:
            return

        periodic_path = os.path.join(self.checkpoint_dir, f"iter_{it + 1:05d}.pt")
        torch.save(
            {
                "iteration": it,
                "actor_state_dict": actor.state_dict(),
                "eps": eps,
            },
            periodic_path,
        )

        avg = self.avg_reward
        if avg > self._best_avg_reward and avg == avg:
            self._best_avg_reward = avg
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(
                {
                    "iteration": it,
                    "actor_state_dict": actor.state_dict(),
                    "target_state_dict": target_actor.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_avg_reward": self._best_avg_reward,
                    "eps": eps,
                },
                best_path,
            )
            print(f"  >>> saved new best to {best_path}  "
                  f"(avg{len(self._reward_history)} reward={avg:+.3f})")

    @property
    def avg_reward(self) -> float:
        return _avg(self._reward_history)

    @property
    def avg_global_reward(self) -> float:
        return _avg(self._global_reward_history)

    @property
    def best_avg_reward(self) -> float:
        return self._best_avg_reward