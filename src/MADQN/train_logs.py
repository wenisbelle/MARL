"""
training_logger.py — Reusable logger for DQN/MADQN training.

Tracks rolling metrics, prints one-line summaries per iteration, and handles
checkpointing (both periodic and best-so-far).

Usage in train.py:

    logger = TrainingLogger(
        checkpoint_dir="checkpoints",
        reward_window=10,
        checkpoint_every=100,
    )

    for it in range(NUM_ITERATIONS):
        ...collect, train...

        logger.log_update(loss=..., q_mean=..., td_error=..., action_entropy=...)
        logger.log_iteration(
            it=it,
            new_count=new_count,
            buffer_size=len(replay_buffer),
            eps=trainer_policy.eps,
            reward_sample=batch["reward"],            # tensor
            global_reward_sample=batch.get("global_reward"),  # optional
        )
        logger.maybe_checkpoint(it, trainer_policy.actor, target_actor, optimizer,
                                 eps=trainer_policy.eps)

The logger is stateless from the trainer's POV — call `log_update` once per
gradient step, `log_iteration` once per iteration, `maybe_checkpoint` once per
iteration. It handles averaging, formatting, file I/O, and "best so far" logic.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

import torch
import matplotlib

# Use a non-blocking backend so plt.show() doesn't pause training.
# "TkAgg" works on most Linux/Mac/Windows setups; if you're on a headless
# server, set live_plot=False and just read the CSV/console output instead.
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
    Per-iteration logger with rolling reward window and checkpointing.

    Args:
        checkpoint_dir: where to save model checkpoints.
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
        live_plot: bool = True,
        plot_every: int = 1,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.live_plot = live_plot
        self.plot_every = plot_every

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Per-iteration buffers — reset each iteration via `reset_iteration`.
        self._losses: list[float] = []
        self._q_means: list[float] = []
        self._td_errors: list[float] = []
        self._action_entropies: list[float] = []

        # Rolling histories across iterations.
        self._reward_history = deque(maxlen=reward_window)
        self._global_reward_history = deque(maxlen=reward_window)

        # Full-history series for plotting (no maxlen — grows with training).
        # These are what feed the live plot, so trends over the full run stay visible.
        self._iters: list[int] = []
        self._reward_series: list[float] = []
        self._avg_reward_series: list[float] = []
        self._loss_series: list[float] = []
        self._q_series: list[float] = []

        # Best-so-far tracking for checkpoint selection.
        self._best_avg_reward = float("-inf")

        # --- Live plot setup ---
        # Four panels: per-iter reward + rolling avg, loss, |Q|, action entropy.
        # Lines are created empty here and updated in-place each iteration —
        # much faster than redrawing the whole figure.
        if self.live_plot:
            plt.ion()                          # interactive mode = non-blocking
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
            self._ent_series: list[float] = []

            self._fig.tight_layout()
            self._fig.canvas.draw()
            plt.show(block=False)

    # ------------------------------------------------------------------
    # Per-update logging — called inside the inner gradient-step loop.
    # ------------------------------------------------------------------
    def log_update(
        self,
        *,
        loss: float,
        q_values_all: torch.Tensor,   # (B, action_dim) — full Q output
        td_error: float,
    ) -> None:
        """
        Record metrics from one gradient step.

        `q_values_all` is the full (B, action_dim) Q-values tensor — we use it
        both for the |Q| magnitude metric and to compute action-distribution
        entropy across the batch (cheap signal for policy collapse).
        """
        self._losses.append(loss)
        with torch.no_grad():
            self._q_means.append(q_values_all.abs().mean().item())
            self._td_errors.append(td_error)

            # Entropy of the empirical greedy-action distribution over the batch.
            # Near log(action_dim) = uniform; near 0 = policy collapsed onto one action.
            action_dim = q_values_all.shape[-1]
            probs = torch.bincount(
                q_values_all.argmax(-1), minlength=action_dim
            ).float()
            probs = probs / probs.sum().clamp(min=1)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            self._action_entropies.append(entropy)

    # ------------------------------------------------------------------
    # Per-iteration logging — called once after the inner training loop.
    # ------------------------------------------------------------------
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
        Print a one-line summary for this iteration. Updates the rolling
        reward history. Safe to call even if no updates happened (e.g. buffer
        not yet full) — empty fields show as `nan`.
        """
        # Reward stats from the last sampled batch — proxy for "how is policy doing".
        mean_reward = (
            reward_sample.mean().item() if reward_sample is not None and reward_sample.numel() > 0
            else float("nan")
        )
        global_reward = (
            global_reward_sample.mean().item()
            if global_reward_sample is not None and global_reward_sample.numel() > 0
            else float("nan")
        )

        # Only record finite values in the rolling window; NaN early on would
        # poison the running average.
        if mean_reward == mean_reward:        # x == x is False only for NaN
            self._reward_history.append(mean_reward)
        if global_reward == global_reward:
            self._global_reward_history.append(global_reward)

        if it % self.print_every == 0:
            print(
                f"[iter {it:4d}] "
                f"loss={_avg(self._losses):.4f} | "
                f"|Q|={_avg(self._q_means):.3f} | "
                f"TD_err={_avg(self._td_errors):+.3f} | "
                f"ent={_avg(self._action_entropies):.2f} | "
                f"r={mean_reward:+.3f} (avg={self.avg_reward:+.3f}) | "
                f"R_team={global_reward:+.3f} (avg={self.avg_global_reward:+.3f}) | "
                f"ε={eps:.3f} | "
                f"buf={buffer_size} | "
                f"new={new_count}"
            )

        # --- Update the plot series with this iteration's values ---
        # We push to the full-history lists BEFORE clearing the per-iter buffers.
        self._iters.append(it)
        self._reward_series.append(mean_reward)
        self._avg_reward_series.append(self.avg_reward)
        self._loss_series.append(_avg(self._losses))
        self._q_series.append(_avg(self._q_means))
        if self.live_plot:
            self._ent_series.append(_avg(self._action_entropies))
            if it % self.plot_every == 0:
                self._refresh_plot()

        # Clear per-iteration buffers for the next round.
        self._losses.clear()
        self._q_means.clear()
        self._td_errors.clear()
        self._action_entropies.clear()

    def _refresh_plot(self) -> None:
        """
        Update line data in place and rescale axes. Much cheaper than
        replotting from scratch — for long runs this keeps the live plot
        responsive without slowing training perceptibly.
        """
        self._line_reward.set_data(self._iters, self._reward_series)
        self._line_avg.set_data(self._iters, self._avg_reward_series)
        self._line_loss.set_data(self._iters, self._loss_series)
        self._line_q.set_data(self._iters, self._q_series)
        self._line_ent.set_data(self._iters, self._ent_series)

        for ax in self._axes.ravel():
            ax.relim()                  # recompute data limits from new line data
            ax.autoscale_view()         # apply them to the visible range

        # `flush_events` instead of `pause()` because pause() blocks training
        # for a small interval; flush_events just yields to the GUI loop.
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        """Save the final plot to disk so you have a record after closing."""
        if self.live_plot:
            self._fig.savefig(os.path.join(self.checkpoint_dir, "training_curves.png"),
                              dpi=120, bbox_inches="tight")
            plt.ioff()

    # ------------------------------------------------------------------
    # Checkpointing.
    # ------------------------------------------------------------------
    def maybe_checkpoint(
        self,
        it: int,
        actor: torch.nn.Module,
        target_actor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        eps: float,
    ) -> None:
        """
        Two-pronged checkpointing:
          * Periodic: every `checkpoint_every` iterations, save the latest
            model unconditionally — protection against crashes.
          * Best-so-far: at the same cadence, overwrite `best.pt` only when
            the rolling-average reward improves.

        Both checks fire on the same iterations so the periodic save acts as
        a fallback if "best" never updates.
        """
        if (it + 1) % self.checkpoint_every != 0:
            return

        # Periodic — always save.
        periodic_path = os.path.join(self.checkpoint_dir, f"iter_{it + 1:05d}.pt")
        torch.save(
            {
                "iteration": it,
                "actor_state_dict": actor.state_dict(),
                "eps": eps,
            },
            periodic_path,
        )

        # Best-so-far — only if rolling avg improved.
        avg = self.avg_reward
        if avg > self._best_avg_reward and avg == avg:    # ignore NaN
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

    # ------------------------------------------------------------------
    # Read-only views — handy if the trainer wants to log elsewhere too.
    # ------------------------------------------------------------------
    @property
    def avg_reward(self) -> float:
        return _avg(self._reward_history)

    @property
    def avg_global_reward(self) -> float:
        return _avg(self._global_reward_history)

    @property
    def best_avg_reward(self) -> float:
        return self._best_avg_reward