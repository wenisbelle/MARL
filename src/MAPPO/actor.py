"""
actor.py — Per-agent Actor network for the cooperative drone-mapping MARL setup.

This module implements the *decentralized execution* side of the CTDE
(Centralized Training, Decentralized Execution) paradigm: the actor sees only
ONE agent's observation at a time and produces that agent's action. A
centralized critic (separate file, to be written later) will see the full
joint state and all agents' actions during training.

============================================================================
 INTEGRATION CONTRACT
============================================================================
The orchestrator / worker calls the policy as:

        policy(per_agent_obs_td) -> action_tensor of shape (action_dim,)

See `simulation_orchestrator.AsyncMARLOrchestrator._handle_flags` and the
`policy_callable` defined inside `worker._worker_loop`. So our `Actor.forward`
must accept a per-agent TensorDict and return just an action tensor.

For training (SAC/PPO/MAPPO style) we also need a stochastic sample with its
log-prob, so we expose a separate `Actor.sample(obs_td) -> (action, log_prob)`.

============================================================================
 OBSERVATION FIELDS CONSUMED (taken directly from MappingEnvironment specs)
============================================================================
For a single agent (the orchestrator slices these out per-agent):

  - map_patch                   : float32, shape (M, M)            in [0, 2.0]
  - individual_map_uncertainty  : float32, shape (1,)               in [0, 1]
  - position                    : float32, shape (2,)               in [0, 1]
  - estimated_positions         : float32, shape (max_num_agents, 2) in [0, 1]
  - encounter_flag              : bool,   shape (1,)

For BATCHED inputs (during training), each tensor has an extra leading dim.
We detect this automatically below.

============================================================================
 ACTION SPACE
============================================================================
Continuous, shape (action_dim,) — default 2 (target xy), or 3 if you turn on
`speed_action` in the env. Each component is in [0, 1] (matches the Bounded
action spec). We squash with `(tanh(u) + 1) / 2` so the output is bounded by
construction and gradients flow well at the interior.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


# ============================================================================
# Sub-modules: one encoder per modality, then a fusion trunk
# ============================================================================

class MapCNN(nn.Module):
    """
    Small CNN that ingests the local map patch (`map_patch`) and returns a
    flat feature vector.

    Architecture:
        Two conv blocks (each: Conv -> ReLU) with padding=1 keep the spatial
        size. A MaxPool halves the resolution. A third conv increases channel
        capacity, and an `AdaptiveAvgPool2d(1)` collapses to a single vector.
        That last pool also makes the encoder size-agnostic: if you change
        `observation_map_size` later, this module still works without code
        changes.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 20x20 -> 10x10 (default)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),               # always (B, 64, 1, 1)
        )
        self.proj = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W). Caller is responsible for adding the channel
               dimension if the env emits (B, H, W).
        Returns:
            (B, feature_dim) feature vector.
        """
        h = self.conv(x)
        h = h.flatten(start_dim=1)                 # (B, 64)
        return F.relu(self.proj(h))                # (B, feature_dim)


class VectorEncoder(nn.Module):
    """
    MLP that encodes the non-spatial fields (`position`,
    `individual_map_uncertainty`, `estimated_positions`, `encounter_flag`).

    We concatenate everything into a single vector and let a couple of dense
    layers do the rest. Nothing fancy: a wider trunk + ReLU is enough because
    the vector dim is small.
    """

    def __init__(self, in_dim: int, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# The Actor itself
# ============================================================================

class Actor(nn.Module):
    """
    Maps a per-agent observation TensorDict to an action in [0, 1]^action_dim.

    Two ways to use it
    ------------------
    1. Decentralized rollout (what `worker.py` does):

           actor = Actor(max_num_agents=3)
           action = actor(per_agent_obs_td)
           # action has shape (action_dim,) — drop-in replacement for RandomPolicy

       By default `forward(...)` returns the DETERMINISTIC mean action
       (squashed). This is exactly what `policy_callable` in the worker wants:
       a single tensor, no log-prob. If you want exploration during rollouts
       (e.g. SAC/PPO), call `forward(obs, deterministic=False)` or just
       `actor.sample(obs)` and discard the log-prob.

    2. Training (SAC / PPO / MAPPO):

           action, log_prob = actor.sample(batched_obs_td)
           # both have a leading batch dim matching the batch of obs_td

    Why a squashed Gaussian (and not, say, plain sigmoid + Bernoulli noise)?
        * Reparameterizable -> SAC / MAPPO need this.
        * Bounded output by construction -> matches the env's Bounded spec.
        * The change-of-variables correction for tanh is a one-liner and is
          numerically stable using `softplus`.

    Batched vs unbatched
    --------------------
    During rollouts the orchestrator slices a single agent's obs (no batch
    dim). During training you'll batch many transitions together (batch dim
    present). `_features` detects which is which by inspecting `position`,
    so callers don't have to worry about it.
    """

    # Clamping range for log_std prevents the policy from collapsing to a
    # delta (log_std -> -inf) or blowing up (log_std -> +inf). The chosen
    # bounds are the de facto SAC defaults.
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0

    def __init__(
        self,
        *,
        max_num_agents: int = 3,        # affects estimated_positions size
        action_dim: int = 2,            # 2 for xy target, 3 if speed_action=True in env
        map_channels: int = 1,          # map_patch is single-channel uncertainty
        map_feature_dim: int = 128,
        vector_feature_dim: int = 64,
        hidden_dim: int = 256,
        # Keys inside per_agent_obs_td — change here if you rename them in the env.
        map_key: str = "map_patch",
        position_key: str = "position",
        flag_key: str = "encounter_flag",
        uncertainty_key: str = "individual_map_uncertainty",
        estimated_positions_key: str = "estimated_positions",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_num_agents = max_num_agents

        # Cache the keys so we don't sprinkle string literals everywhere.
        self.map_key = map_key
        self.position_key = position_key
        self.flag_key = flag_key
        self.uncertainty_key = uncertainty_key
        self.estimated_positions_key = estimated_positions_key

        # Vector input dim = position(2) + uncertainty(1) + flag(1) + estimated(N*2).
        # If you later add or remove obs fields, update this number and the
        # corresponding concatenation inside `_features`.
        self.vector_in_dim = 2 + 1 + 1 + (max_num_agents * 2)

        # --- Encoders ---
        self.map_encoder = MapCNN(
            in_channels=map_channels, feature_dim=map_feature_dim
        )
        self.vec_encoder = VectorEncoder(
            in_dim=self.vector_in_dim, feature_dim=vector_feature_dim
        )

        # --- Fusion trunk: combines spatial + vector embeddings ---
        fused_dim = map_feature_dim + vector_feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Heads: output Gaussian (mean, log_std) over a pre-squash latent u ---
        # Final action a = (tanh(u) + 1) / 2  -> lives in [0, 1]^action_dim.
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    # ------------------------------------------------------------------
    # Internal: extract tensors from the TensorDict, normalize shapes,
    # run encoders, and return the fused features.
    # ------------------------------------------------------------------
    def _features(self, obs_td: TensorDict) -> Tuple[torch.Tensor, bool]:
        """
        Returns:
            features:    (B, hidden_dim) fused embedding
            is_unbatched: True if caller passed a single sample, so we know
                          to squeeze the leading dim out of the output later.
        """
        pos = obs_td[self.position_key]            # unbatched (2,)   | batched (B, 2)
        flag = obs_td[self.flag_key]               # unbatched (1,)   | batched (B, 1)
        unc = obs_td[self.uncertainty_key]         # unbatched (1,)   | batched (B, 1)
        ep = obs_td[self.estimated_positions_key]  # unbatched (N, 2) | batched (B, N, 2)
        mp = obs_td[self.map_key]                  # unbatched (H, W) | batched (B, H, W)

        # Detect batching from `position` (1-D = single sample, 2-D = batched).
        is_unbatched = pos.dim() == 1
        if is_unbatched:
            pos = pos.unsqueeze(0)
            flag = flag.unsqueeze(0)
            unc = unc.unsqueeze(0)
            ep = ep.unsqueeze(0)
            mp = mp.unsqueeze(0)

        # The env emits map_patch as (B, H, W) — no channel dim. The CNN
        # expects (B, C, H, W), so insert a singleton channel.
        if mp.dim() == 3:
            mp = mp.unsqueeze(1)                   # (B, 1, H, W)

        # `encounter_flag` is bool in the spec — must be float for the MLP.
        # Cast everything else to float too, just in case the env returns
        # something different (e.g. on GPU with mixed dtypes).
        pos = pos.float()
        flag = flag.float()
        unc = unc.float()
        ep = ep.float()
        mp = mp.float()

        # Flatten `estimated_positions` from (B, N, 2) to (B, N*2) so it can
        # be concatenated with the other 1-D vectors.
        ep_flat = ep.flatten(start_dim=1)          # (B, N*2)

        # --- Encode each modality and concatenate ---
        map_feat = self.map_encoder(mp)            # (B, map_feature_dim)
        vec_in = torch.cat([pos, unc, flag, ep_flat], dim=-1)  # (B, vector_in_dim)
        vec_feat = self.vec_encoder(vec_in)        # (B, vector_feature_dim)

        fused = torch.cat([map_feat, vec_feat], dim=-1)
        return self.trunk(fused), is_unbatched     # (B, hidden_dim), bool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        obs_td: TensorDict,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Run the actor forward and return ONLY the action tensor.

        Args:
            obs_td: per-agent observation TensorDict (batched or unbatched).
            deterministic:
                * True  (default) -> return the squashed mean action. This is
                  what you want during evaluation, or when you're going to add
                  external exploration noise (DDPG / TD3 style).
                * False -> return a reparameterized sample from the policy
                  (useful as the "rollout action" for SAC/PPO if you want
                  exploration to come from the policy itself).

        Returns:
            action tensor in [0, 1]^action_dim. Shape (action_dim,) when the
            input was unbatched, or (B, action_dim) when batched.
        """
        h, is_unbatched = self._features(obs_td)
        u_mean = self.mean_head(h)                 # (B, action_dim) — pre-squash mean

        if deterministic:
            u = u_mean
        else:
            log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = log_std.exp()
            # rsample-equivalent: u = mu + sigma * eps, with eps ~ N(0, I).
            # This is reparameterized — gradients flow through mu and sigma.
            eps = torch.randn_like(u_mean)
            u = u_mean + std * eps

        # Squash: (tanh(u) + 1) / 2 maps R -> (0, 1).
        action = 0.5 * (torch.tanh(u) + 1.0)

        if is_unbatched:
            action = action.squeeze(0)             # (action_dim,)
        return action

    def sample(self, obs_td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample from the squashed Gaussian policy + its log-prob.

        Returns:
            action:   (..., action_dim) in [0, 1]
            log_prob: (..., 1) — log π(action | obs), summed over action dims.

        Use this in the actor update of SAC / PPO / MAPPO. You typically need
        log_prob to compute the entropy term (SAC) or the importance ratio (PPO).

        Math (per action dim, then summed):

            u ~ N(μ(s), σ(s))                            # pre-squash Gaussian
            a = 0.5 * (tanh(u) + 1)                      # squash + rescale to [0, 1]

            log π(a | s) = log N(u; μ, σ)
                           - log |det da/du|
                         = log N(u; μ, σ)
                           - Σ_i [ log(0.5) + log(1 - tanh(u_i)^2) ]

            The log(1 - tanh(u)^2) term is computed in a numerically stable
            way as 2 * (log 2 - u - softplus(-2u)).
        """
        h, is_unbatched = self._features(obs_td)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterized sample.
        eps = torch.randn_like(mean)
        u = mean + std * eps                       # pre-squash latent action

        # log N(u; mean, std), summed over action dims -> shape (B, 1).
        log_prob_u = (
            -0.5 * ((u - mean) / std).pow(2)
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)

        # Squash + rescale to [0, 1].
        action = 0.5 * (torch.tanh(u) + 1.0)

        # Change of variables for the tanh squash. Stable form:
        # log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
        log_det_tanh = (
            2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        ).sum(dim=-1, keepdim=True)

        # Change of variables for the (* 0.5) rescaling. This is just a
        # constant per dim, so it doesn't affect gradients, but we include
        # it so log_prob is correct in absolute terms.
        log_det_rescale = self.action_dim * math.log(0.5)

        log_prob = log_prob_u - log_det_tanh - log_det_rescale

        if is_unbatched:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob


# ============================================================================
# Smoke test — run `python actor.py` to see the actor produce sensible output.
# This mirrors how the worker would call the actor during a rollout.
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # Build a dummy per-agent observation that matches the env's spec.
    M = 20
    N = 3
    per_agent_obs = TensorDict(
        {
            "map_patch":                   torch.rand(M, M) * 0.8,
            "individual_map_uncertainty":  torch.rand(1000),
            "position":                    torch.rand(2),
            "estimated_positions":         torch.rand(N, 2),
            "encounter_flag":              torch.zeros(1, dtype=torch.bool),
        },
        batch_size=[],
    )

    actor = Actor(max_num_agents=N, action_dim=2)

    # 1) Deterministic forward — what the worker will call.
    a_det = actor(per_agent_obs)
    print("deterministic action :", a_det.shape, a_det.tolist())
    assert a_det.shape == (2,)
    assert torch.all(a_det >= 0) and torch.all(a_det <= 1)

    # 2) Stochastic sample with log-prob — what you'll call in training.
    a_stoch, logp = actor.sample(per_agent_obs)
    print("stochastic action    :", a_stoch.shape, a_stoch.tolist())
    print("log_prob             :", logp.shape, logp.item())
    assert a_stoch.shape == (2,)
    assert logp.shape == (1,)

    # 3) Batched forward — what training will use.
    B = 8
    batched_obs = TensorDict(
        {
            "map_patch":                   torch.rand(B, M, M) * 2.0,
            "individual_map_uncertainty":  torch.rand(B, 1),
            "position":                    torch.rand(B, 2),
            "estimated_positions":         torch.rand(B, N, 2),
            "encounter_flag":              torch.zeros(B, 1, dtype=torch.bool),
        },
        batch_size=[B],
    )
    a_batch = actor(batched_obs, deterministic=False)
    a_batch_s, logp_batch = actor.sample(batched_obs)
    print("batched action       :", a_batch.shape)
    print("batched sample logp  :", logp_batch.shape)
    assert a_batch.shape == (B, 2)
    assert logp_batch.shape == (B, 1)

    print("\nAll smoke tests passed.")