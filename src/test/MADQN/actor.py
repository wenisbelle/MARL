"""
actor.py — Per-agent Actor network for the cooperative drone-mapping MARL setup.



 INTEGRATION 

The orchestrator / worker calls the policy as:

        policy(per_agent_obs_td) -> action_tensor of shape (action_dim,)

See `simulation_orchestrator.AsyncMARLOrchestrator._handle_flags` and the
`policy_callable` defined inside `worker._worker_loop`. So our `Actor.forward`
must accept a per-agent TensorDict and return just an action tensor.


 OBSERVATION FIELDS CONSUMED (taken directly from MappingEnvironment specs)

For a single agent (the orchestrator slices these out per-agent):

  - map_patch                   : float32, shape (M, M)            in [0, 2.0]
  - individual_map_uncertainty  : float32, shape (1,)               in [0, 1]
  - position                    : float32, shape (2,)               in [0, 1]
  - estimated_positions         : float32, shape (max_num_agents, 2) in [0, 1]

For BATCHED inputs (during training), each tensor has an extra leading dim.
We detect this automatically below.

 ACTION SPACE

Continuous, shape (action_dim,) — default categorical M^2 (target cell)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


# Sub-modules: one encoder per modality, then a fusion

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
    `individual_map_uncertainty`, `estimated_positions`).

    We concatenate everything into a single vector
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



class Actor(nn.Module):
    """
    Maps a per-agent observation TensorDict to an action in action_dim.

    Two ways to use it
    ------------------
    1. Decentralized rollout (what `worker.py` does):

           actor = Actor(max_num_agents=3)
           action = actor(per_agent_obs_td)
           # action has shape (action_dim,)

       By default `forward(...)` returns the DETERMINISTIC mean action
       (squashed). This is exactly what `policy_callable` in the worker wants:
       a single tensor, no log-prob. If you want exploration during rollouts
       (e.g. SAC/PPO), call `forward(obs, deterministic=False)` or just
       `actor.sample(obs)` and discard the log-prob.

    2. Training (SAC / PPO / MAPPO):

           action, log_prob = actor.sample(batched_obs_td)
           # both have a leading batch dim matching the batch of obs_td


    Batched vs unbatched
    --------------------
    During rollouts the orchestrator slices a single agent's obs (no batch
    dim). During training you'll batch many transitions together (batch dim
    present). `_features` detects which is which by inspecting `position`,
    so callers don't have to worry about it.
    """

    def __init__(
        self,
        *,
        max_num_agents: int = 3,        # affects estimated_positions size
        action_dim: int = 400,          # size of the action space (M*M for categorical)
        map_channels: int = 1,          # map_patch is single-channel uncertainty
        map_feature_dim: int = 128,
        vector_feature_dim: int = 64,
        hidden_dim: int = 256,
        # Keys inside per_agent_obs_td — change here if you rename them in the env.
        map_key: str = "map_patch",
        position_key: str = "position",
        uncertainty_key: str = "individual_map_uncertainty",
        estimated_positions_key: str = "estimated_positions",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_num_agents = max_num_agents

        self.map_key = map_key
        self.position_key = position_key
        self.uncertainty_key = uncertainty_key
        self.estimated_positions_key = estimated_positions_key

        # Vector input dim = position(2) + uncertainty(1) + estimated(N*2).
        # If you later add or remove obs fields, update this number and the
        # corresponding concatenation inside `_features`.
        self.vector_in_dim = 2 + 1 + (max_num_agents * 2)

        ##### Inputs
        self.map_encoder = MapCNN(
            in_channels=map_channels, feature_dim=map_feature_dim
        )
        self.vec_encoder = VectorEncoder(
            in_dim=self.vector_in_dim, feature_dim=vector_feature_dim
        )

        # Fusion merges the CNN and the vector features. Simple MLP
        fused_dim = map_feature_dim + vector_feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.q_head = nn.Linear(hidden_dim, action_dim)

    # Internal: extract tensors from the TensorDict
    # run encoders, and return the fused features.

    def _features(self, obs_td: TensorDict) -> Tuple[torch.Tensor, bool]:
        """
        Returns:
            features:    (B, hidden_dim) fused embedding
            is_unbatched: True if caller passed a single sample, so we know
                          to squeeze the leading dim out of the output later.
        """
        pos = obs_td[self.position_key]            # unbatched (2,)   | batched (B, 2)
        unc = obs_td[self.uncertainty_key]         # unbatched (1,)   | batched (B, 1)
        ep = obs_td[self.estimated_positions_key]  # unbatched (N, 2) | batched (B, N, 2)
        mp = obs_td[self.map_key]                  # unbatched (H, W) | batched (B, H, W)

        # Detect batching from `position` (1-D = single sample, 2-D = batched).
        is_unbatched = pos.dim() == 1
        if is_unbatched:
            pos = pos.unsqueeze(0)
            unc = unc.unsqueeze(0)
            ep = ep.unsqueeze(0)
            mp = mp.unsqueeze(0)

        # The env emits map_patch as (B, H, W) — no channel dim. The CNN
        # expects (B, C, H, W), so insert a singleton channel.
        if mp.dim() == 3:
            mp = mp.unsqueeze(1)                   # (B, 1, H, W)

        # Cast everything else to float too, just in case the env returns
        # something different (e.g. on GPU with mixed dtypes).
        pos = pos.float()
        unc = unc.float()
        ep = ep.float()
        mp = mp.float()

        # Flatten `estimated_positions` from (B, N, 2) to (B, N*2) so it can
        # be concatenated with the other 1-D vectors.
        ep_flat = ep.flatten(start_dim=1)          # (B, N*2)

        # Encode each modality and concatenate 
        map_feat = self.map_encoder(mp)                  # (B, map_feature_dim)
        vec_in = torch.cat([pos, unc, ep_flat], dim=-1)  # (B, vector_in_dim)
        vec_feat = self.vec_encoder(vec_in)              # (B, vector_feature_dim)

        fused = torch.cat([map_feat, vec_feat], dim=-1)
        return self.trunk(fused), is_unbatched     # (B, hidden_dim), bool

    def forward(
        self,
        obs_td: TensorDict
        ) -> torch.Tensor:
        """
        Run the actor forward and return ONLY the action tensor.
        Returns:
            action tensor in Shape (action_dim,) when the
            input was unbatched, or (B, action_dim) when batched.
        """
        h, is_unbatched = self._features(obs_td)
        q_values = self.q_head(h)                  # (B, action_dim) — pre-squash mean

        return q_values.squeeze(0) if is_unbatched else q_values

"""
if __name__ == "__main__":
    torch.manual_seed(0)

    # Build a dummy per-agent observation that matches the env's spec.
    M = 20
    N = 3
    per_agent_obs = TensorDict(
        {
            "map_patch":                   torch.rand(M, M) * 0.8,
            "individual_map_uncertainty":  torch.rand(1),
            "position":                    torch.rand(2),
            "estimated_positions":         torch.rand(N, 2),
        },
        batch_size=[],
    )

    actor = Actor(max_num_agents=N, action_dim=M*M)

    q_values = actor.forward(per_agent_obs)
    print(q_values)
    action = q_values.argmax(dim=-1, keepdim=True)
    print(f"Q max: {action.item()}")


    # 3) Batched forward — what training will use.
    B = 8
    batched_obs = TensorDict(
        {
            "map_patch":                   torch.rand(B, M, M) * 2.0,
            "individual_map_uncertainty":  torch.rand(B, 1),
            "position":                    torch.rand(B, 2),
            "estimated_positions":         torch.rand(B, N, 2),
        },
        batch_size=[B],
    )
    q_batch = actor.forward(batched_obs)
    action = q_batch.argmax(dim=-1, keepdim=True)
    print(f"Batched Q max indices: {action.squeeze(-1).tolist()}")
"""
