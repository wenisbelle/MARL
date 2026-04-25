"""
train_sample_factory.py
-----------------------
Sample Factory 2.x training script for GradySim drone coverage.

Install dependencies
--------------------
    pip install sample-factory gymnasium torch numpy

Quickstart (same interface as your existing GA — just run this instead)
-----------------------------------------------------------------------
    python -m your_package.train_sample_factory \
        --algo APPO \
        --env gradysim-coverage \
        --experiment coverage_v1 \
        --num_workers 12 \
        --num_envs_per_worker 1 \
        --map_width 10 \
        --map_height 10 \
        --num_drones 3 \
        --sim_duration 1000

    # Resume a run:
    python -m your_package.train_sample_factory \
        --algo APPO --env gradysim-coverage \
        --experiment coverage_v1 --restart_behavior resume

    # Watch a trained agent (no GPU needed):
    python -m your_package.enjoy_sample_factory \
        --algo APPO --env gradysim-coverage \
        --experiment coverage_v1

Parallelism model
-----------------
Sample Factory (APPO) uses asynchronous PPO:

  ┌──────────────────────────────────────────────────────┐
  │  N worker processes  (num_workers)                   │
  │  Each runs one GradySimEnv  (num_envs_per_worker=1)  │
  │  Each env runs GradySim in a daemon thread           │
  │  → No GIL contention; pure multiprocessing           │
  └──────────────────────────────────────────────────────┘
              │  observations (shared memory)
              ▼
  ┌──────────────────────────────────────────────────────┐
  │  Inference process  (GPU / CPU)                      │
  │  Batches obs from all workers → forward pass         │
  │  Returns actions to workers asynchronously           │
  └──────────────────────────────────────────────────────┘
              │  gradient updates
              ▼
  ┌──────────────────────────────────────────────────────┐
  │  Learner process  (GPU)                              │
  │  Samples from rollout buffer → PPO update            │
  │  Pushes new weights to inference process             │
  └──────────────────────────────────────────────────────┘

This is the same parallelism pattern your GA already uses (multiprocessing.Pool),
but APPO allows the GPU learner to update *continuously* while workers collect
experience — GPU utilisation stays high even with slow CPU environments.

Tuning guidance
---------------
- map_width/map_height : start small (10×10), scale up once the agent learns.
- num_workers          : set to physical core count minus 2.
- rollout              : 128–256 works well; increase if episodes are very long.
- batch_size           : 2048 is a safe default; scale with num_workers.
- entropy_coeff        : 0.01 encourages exploration of the map.
- After your GA converges, pass --fuzzy_params_path to seed background drones
  with the best individual, creating a stronger cooperative baseline.
"""

import os
import sys
from typing import Optional

import numpy as np

from sample_factory.cfg.arguments import parse_full_cfg, prepare_arg_parser
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from .gradysim_env import GradySimEnv, _DEFAULT_FUZZY_PARAMS

# ──────────────────────────────────────────────────────────────────────────────
# Environment name (used everywhere in Sample Factory CLI)
# ──────────────────────────────────────────────────────────────────────────────
ENV_NAME = "gradysim-coverage"


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# Called independently in each Sample Factory worker process.
# No shared state — each call creates a completely isolated GradySimEnv.
# ──────────────────────────────────────────────────────────────────────────────

def make_gradysim_env(full_env_name: str, cfg, env_config: Optional[dict] = None):
    """
    Factory function registered with Sample Factory.

    Parameters forwarded from the parsed CLI config:
      cfg.map_width, cfg.map_height, cfg.num_drones, cfg.sim_duration,
      cfg.fuzzy_params_path  (optional path to .npy file with GA-optimised params)
    """
    fuzzy_params = _load_fuzzy_params(getattr(cfg, "fuzzy_params_path", None))

    return GradySimEnv(
        map_width=cfg.map_width,
        map_height=cfg.map_height,
        number_of_drones=cfg.num_drones,
        simulation_duration=cfg.sim_duration,
        fuzzy_params=fuzzy_params,
        communication_range=getattr(cfg, "comm_range", 200.0),
    )


def _load_fuzzy_params(path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load fuzzy parameters from a .npy file (e.g. saved best GA individual).
    Returns None if no path is given, which falls back to _DEFAULT_FUZZY_PARAMS.
    """
    if path is None or not os.path.isfile(path):
        return None
    params = np.load(path)
    print(f"[GradySim] Loaded fuzzy params from {path}  shape={params.shape}")
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Custom CLI parameters
# ──────────────────────────────────────────────────────────────────────────────

def add_extra_params(parser) -> None:
    """Register GradySim-specific command-line arguments."""
    g = parser.add_argument_group("GradySim environment")
    g.add_argument(
        "--map_width", default=10, type=int,
        help="Map grid width (cells). Start with 10, scale to 50 after convergence.",
    )
    g.add_argument(
        "--map_height", default=10, type=int,
        help="Map grid height (cells).",
    )
    g.add_argument(
        "--num_drones", default=3, type=int,
        help="Total drones per simulation (1 RL-controlled + rest fuzzy).",
    )
    g.add_argument(
        "--sim_duration", default=1000, type=int,
        help="Simulated seconds per episode. Matches your GA setup.",
    )
    g.add_argument(
        "--comm_range", default=200.0, type=float,
        help="Inter-drone communication range (metres).",
    )
    g.add_argument(
        "--fuzzy_params_path", default=None, type=str,
        help=(
            "Path to a .npy file containing fuzzy parameters for background drones. "
            "Use your best GA individual here: np.save('best_ga.npy', hof[0])"
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter overrides
# These are sensible starting points.  Profile a few hundred episodes, then
# tune rollout / batch_size / entropy_coeff based on your GPU memory and the
# observed reward variance.
# ──────────────────────────────────────────────────────────────────────────────

def override_default_params(parser) -> None:
    parser.set_defaults(
        # ── Network architecture ──────────────────────────────────────────────
        # MLP suits the flat observation vector (map + position).
        # Switch to "conv" if you later include the 2-D map as a separate channel.
        encoder_type="mlp",
        encoder_subtype="mlp_mujoco",   # 2-layer MLP, ReLU activations
        hidden_size=256,                # increase to 512 for 50×50 maps

        # ── Rollout & batch ───────────────────────────────────────────────────
        # Each RL step ≈ one waypoint decision (a few simulated seconds).
        # An episode of 1000 s typically yields 100–300 steps.
        rollout=128,
        batch_size=2048,
        num_epochs=4,               # PPO epochs per batch
        num_batches_per_epoch=1,

        # ── Parallelism ───────────────────────────────────────────────────────
        # One env per worker: GradySim episodes are CPU-heavy; packing multiple
        # envs per worker would starve them.
        num_envs_per_worker=1,
        # worker_num_splits controls APPO pipeline depth — 2 is the sweet spot.
        worker_num_splits=2,

        # ── PPO hyperparams ───────────────────────────────────────────────────
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        lr_schedule="constant",      # switch to "kl_adaptive_minibatch" if unstable
        clip_ratio=0.2,
        value_loss_coeff=0.5,
        entropy_coeff=0.01,          # encourages map exploration

        # ── Reward & observation normalisation ────────────────────────────────
        # Normalising returns is important here: uncertainty rewards are tiny
        # fractions and vary with map size.
        normalize_returns=True,
        normalize_input=True,
        obs_subtract_mean=0.0,
        obs_scale=1.0,

        # ── Logging & checkpointing ───────────────────────────────────────────
        with_wandb=False,           # set True and add --wandb_project to enable
        save_every_sec=300,
        keep_checkpoints=3,
        stats_avg=100,              # average reward over N episodes in logs

        # ── Misc ──────────────────────────────────────────────────────────────
        # async_rl=True is the default for APPO and is what gives us GPU/CPU overlap.
        # Do not set this to False unless debugging.
        serial_mode=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    # Register the environment *before* parsing args so Sample Factory
    # can validate --env against the registry.
    register_env(ENV_NAME, make_gradysim_env)

    parser = prepare_arg_parser()
    add_extra_params(parser)
    override_default_params(parser)

    cfg = parse_full_cfg(parser)
    status = run_rl(cfg)
    return status


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation / enjoyment helper  (separate from training)
# ──────────────────────────────────────────────────────────────────────────────
# To watch a trained policy, run:
#
#   python -m your_package.train_sample_factory \
#       --algo APPO --env gradysim-coverage \
#       --experiment coverage_v1 --enjoy
#
# Or create a separate enjoy_sample_factory.py that calls enjoy() instead of
# run_rl().  The environment and parameter registration is identical.

def enjoy_main() -> int:
    """Entry point for policy evaluation (no training)."""
    from sample_factory.enjoy import enjoy  # imported here to avoid circular deps

    register_env(ENV_NAME, make_gradysim_env)

    parser = prepare_arg_parser()
    add_extra_params(parser)
    override_default_params(parser)

    cfg = parse_full_cfg(parser)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    # Allows both:
    #   python train_sample_factory.py ...          (training)
    #   python train_sample_factory.py --enjoy ...  (evaluation)
    args = sys.argv[1:]
    if "--enjoy" in args:
        sys.argv.remove("--enjoy")
        sys.exit(enjoy_main())
    else:
        sys.exit(main())
