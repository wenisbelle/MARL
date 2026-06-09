from dataclasses import dataclass, field
from typing import Callable
import torch
from tensordict import TensorDict

from env.mapping_environment import FlagMessage


@dataclass
class PendingTransition:
    ##### Agent
    agent_state: TensorDict     # the per-agent slice of the observation at decision time
    agent_action: torch.Tensor  # shape (action_dim,)
    ##### Global
    global_state: TensorDict   # the "global_state" subtree at decision time
    joint_action: torch.Tensor # shape (N, action_dim) — last-committed action per agent
    ##### Agent
    agent_reward_sum: float = 0.0
    agent_n_sim_steps: int = 0  # how many sim steps this macro-action has lasted



class AsyncMARLOrchestrator:
    """
    Wraps a MappingEnvironment and produces SMDP-style transitions.

    `policy_fn(per_agent_obs_td) -> action_tensor` is the decision rule
    random, scripted, or a neural network. It only ever sees observations
    for agents that are actually making a decision this step.
    """

    def __init__(self, env, policy_fn: Callable[[TensorDict], torch.Tensor], scale: int, reward_decay: float = 0.99):
        self.env = env
        self.policy_fn = policy_fn
        self.N = env.max_num_agents
        self.action_dim = env.action_spec["agents", "action"].shape[-1]
        self.REWARD_SCALE = scale
        self.REWARD_DECAY = reward_decay

        self.pending_transition: list[PendingTransition | None] = [None] * self.N 
        
        # The most recent action each agent committed to. Updated only on flag events.
        self.last_committed_action = torch.zeros(self.N, self.action_dim)

        # Output buffers
        self.transitions: list[dict] = []



    @staticmethod
    def _read_flags(obs_td: TensorDict) -> torch.Tensor:
        """Return a (N,) bool tensor: True if the agent has a pending decision."""
        return obs_td["agents", "observation", "encounter_flag"].squeeze(-1).bool()

    def _slice_agent_obs(self, obs_td: TensorDict, i: int) -> TensorDict:
        """Take just agent i's observation, cloned so the env can't mutate it."""
        obs = obs_td["agents", "observation"]
        return TensorDict(
            {key: value[i].clone() for key, value in obs.items()},
            batch_size=[],
            device=obs.device,
        )

    def _snapshot_global(self, obs_td: TensorDict) -> TensorDict:
        return obs_td["global_state"].clone()

    def reset(self):
        td = self.env.reset()
        self.pending_transition = [None] * self.N
        self.last_committed_action.zero_()

        # Output buffers
        self.transitions: list[dict] = []

        # At reset all real agents are flagged INTERNAL — open transitions for them.
        self._handle_flags(td, opening_episode=True)
        return td

    def step(self, td: TensorDict) -> TensorDict:
        # Random actions for unflagged agents (they'll be discarded by the env's gate anyway).
        td = self.env.rand_action(td)

        # Overwrite the rows where we have a committed action from the policy.
        action = td["agents", "action"]
        for i in range(self.N):
            if self.pending_transition[i] is not None and self.pending_transition[i].agent_n_sim_steps == 0:
                action[i] = self.pending_transition[i].agent_action

        # Step.
        td = self.env.step(td)

        # Accumulate rewards into all pending transitions.
        global_reward = td["next", "global_reward"].item()
        mask = td["next", "agents", "mask"]
        step_reward = global_reward / self.REWARD_SCALE

        for i in range(self.N):
            p = self.pending_transition[i]
            if p is not None and mask[i]:
                # Avoid reward to explode
                temporal_reward = (self.REWARD_DECAY**p.agent_n_sim_steps) * step_reward
                p.agent_reward_sum += max(-2.0, min(2.0, temporal_reward))         
                p.agent_n_sim_steps += 1


        # Episode end.
        if td["next", "done"].item():
            self._close_all_pending(td["next"], terminal=True)
            return td

        # Per-agent deaths.
        per_agent_done = td["next", "agents", "done"].squeeze(-1)
        for i in range(self.N):
            if per_agent_done[i] and self.pending_transition[i] is not None:
                self._close_agent(i, td["next"], terminal=True)

        # New flags at t+1.
        self._handle_flags(td["next"])

        return td

    ##### flag handling 
    def _handle_flags(self, obs_td: TensorDict, opening_episode: bool = False):
        flags = self._read_flags(obs_td)
        mask = obs_td["agents", "mask"]

        for i in range(self.N):
            if not mask[i] or (not flags[i].item() and not opening_episode):
                continue
            
            if self.pending_transition[i] is not None:
                self._close_agent(i, obs_td, terminal=False)

            per_agent_obs = self._slice_agent_obs(obs_td, i)
            new_action = self.policy_fn(per_agent_obs)
            self.last_committed_action[i] = new_action

            self.pending_transition[i] = PendingTransition(
                agent_state=per_agent_obs,
                agent_action=new_action.clone(),
                global_state=self._snapshot_global(obs_td),
                joint_action=self.last_committed_action.clone(),
            )
            

    def _close_agent(self, i: int, next_obs_td: TensorDict, terminal: bool):
        p = self.pending_transition[i]
        transition = TensorDict({
            # agent side
            "obs":         p.agent_state,
            "next_obs":    self._slice_agent_obs(next_obs_td, i),
            "action":      p.agent_action,
            "reward":      torch.tensor([p.agent_reward_sum], dtype=torch.float32),
            "done":        torch.tensor([terminal], dtype=torch.bool),
            "n_sim_steps": torch.tensor([p.agent_n_sim_steps], dtype=torch.long),
            "agent_idx":   torch.tensor([i], dtype=torch.long),
            # global side
            "global_obs":         p.global_state,
            "global_next_obs":    self._snapshot_global(next_obs_td),
            "joint_action":       p.joint_action,
        }, batch_size=[])
        self.transitions.append(transition)
        self.pending_transition[i] = None
        

    def _close_all_pending(self, next_obs_td: TensorDict, terminal: bool):
        for i in range(self.N):
            if self.pending_transition[i] is not None:
                self._close_agent(i, next_obs_td, terminal=terminal)

    def random_policy(per_agent_obs):
        return torch.rand(2)

