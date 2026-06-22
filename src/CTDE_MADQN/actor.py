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

class DualMapEncoder(nn.Module):
    """
    Two-resolution map encoder.
        large : (B, in_channels, N, N)  — wide FOV
        small : (B, in_channels, M, M)  — narrow FOV
        The small should have the same size as the action 
    The large branch is convolved and downsampled to M×M, concatenated on the
    channel axis with the small branch, jointly convolved, flattened, projected.
    """
    def __init__(self, in_channels: int = 2, large_size: int = 50,
                 small_size: int = 10, feature_dim: int = 128):
        super().__init__()
        self.large_size = large_size
        self.small_size = small_size

        # large branch: extract features, then lock to M×M
        self.large_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),         
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((small_size, small_size)),     # -> (B, 64, M, M) exactly
        )
        # small branch: same resolution, just extract features
        self.small_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )                                                       # -> (B, 64, M, M)
        # joint head: concat on channels, convolve at M×M
        self.joint_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(self._infer_flat_dim(in_channels), feature_dim)

    @torch.no_grad()
    def _infer_flat_dim(self, in_channels: int) -> int:
        large = torch.zeros(1, in_channels, self.large_size, self.large_size)
        small = torch.zeros(1, in_channels, self.small_size, self.small_size)
        return self._fuse(large, small).flatten(1).shape[1]

    def _fuse(self, large, small):
        l = self.large_stem(large)        # (B, 64, M, M)
        s = self.small_stem(small)        # (B, 64, M, M)
        x = torch.cat([l, s], dim=1)      # (B, 128, M, M)
        return self.joint_conv(x)         # (B, 128, M, M)

    def forward(self, large, small):
        x = self._fuse(large, small).flatten(1)   # (B, 128*M*M)
        return F.relu(self.proj(x))               # (B, feature_dim)


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
        action_dim: int = 100,          # size of the action space 
        map_channels: int = 2,          # map_patch is channel uncertainty and channel distance
        large_map_size: int = 50,
        small_map_size: int = 10,
        map_feature_dim: int = 128,
        vector_feature_dim: int = 64,
        hidden_dim: int = 256,
        # Keys inside per_agent_obs_td — change here if you rename them in the env.
        large_map_key: str = "large_map_patch",
        small_map_key: str = "small_map_patch",
        position_key: str = "position",
        uncertainty_key: str = "individual_map_uncertainty",
        estimated_positions_key: str = "estimated_positions_and_time",
        valid_actions_key: str = "valid_actions", 
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_num_agents = max_num_agents

        self.large_map_key = large_map_key
        self.small_map_key = small_map_key
        self.position_key = position_key
        self.uncertainty_key = uncertainty_key
        self.estimated_positions_key = estimated_positions_key
        self.valid_actions_key = valid_actions_key

        # Vector input dim = position(2) + uncertainty(1) + estimated((N-1)*3). -> For each of the other agents (x, y, time_of_last_observation)
        # If you later add or remove obs fields, update this number and the
        # corresponding concatenation inside `_features`.
        self.vector_in_dim = 2 + 1 + ((max_num_agents - 1) * 3)

        ##### Inputs
        self.map_encoder = DualMapEncoder(
            in_channels=map_channels,
            large_size=large_map_size,
            small_size=small_map_size,
            feature_dim=map_feature_dim,
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
        pos = obs_td[self.position_key]            # unbatched (2,)     | batched (B, 2)
        unc = obs_td[self.uncertainty_key]         # unbatched (1,)     | batched (B, 1)
        ep = obs_td[self.estimated_positions_key]  # unbatched (N-1, 3) | batched (B, N-1, 3)
        large_map = obs_td[self.large_map_key].float()
        small_map = obs_td[self.small_map_key].float()                  

        # Detect batching from `position` (1-D = single sample, 2-D = batched).
        is_unbatched = pos.dim() == 1
        if is_unbatched:
            pos = pos.unsqueeze(0)
            unc = unc.unsqueeze(0)
            ep = ep.unsqueeze(0)
            large_map = large_map.unsqueeze(0)
            small_map = small_map.unsqueeze(0)

        # The env may emit map_patch as (B, H, W) — no channel dim. The CNN
        # expects (B, C, H, W), so insert a singleton channel.
        if large_map.dim() == 3:
            large_map = large_map.unsqueeze(1)                   # (B, 1, H, W)
        
        if small_map.dim() == 3:
            small_map = small_map.unsqueeze(1)                   # (B, 1, H, W)

        # Cast everything else to float too, just in case the env returns
        # something different (e.g. on GPU with mixed dtypes).
        pos = pos.float()
        unc = unc.float()
        ep = ep.float()
        large_map = large_map.float()
        small_map = small_map.float()

        # Flatten `estimated_positions` from (B, N, 2) to (B, N*2) so it can
        # be concatenated with the other 1-D vectors.
        ep_flat = ep.flatten(start_dim=1)          # (B, N*2)

        # Encode each modality and concatenate 
        map_feat = self.map_encoder(large_map, small_map)                  # (B, map_feature_dim)
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

        try:
            valid = obs_td[self.valid_actions_key]
            if is_unbatched and valid.dim() == 1:
                valid = valid.unsqueeze(0)
            q_values = q_values.masked_fill(~valid.bool(), float("-inf"))
        except KeyError:
            pass
        
        return q_values.squeeze(0) if is_unbatched else q_values

"""
if __name__ == "__main__":
    N, M, C = 50, 20, 2
    a = Actor(max_num_agents=3, action_dim=100, large_map_size=N, small_map_size=M)
    ub = TensorDict({"map_patch_large": torch.rand(C, N, N),
                     "map_patch_small": torch.rand(C, M, M),
                     "individual_map_uncertainty": torch.rand(1),
                     "position": torch.rand(2),
                     "estimated_positions_and_time": torch.rand(2, 3),
                     "valid_actions": torch.ones(100)}, batch_size=[])
    print(a(ub).shape)                                  # (100,)
    b = ub.expand(8).clone(); b.batch_size = [8]
    print(a(b).shape)                                   # (8, 100)
"""