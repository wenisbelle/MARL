from dataclasses import dataclass, field
from typing import Callable
import torch
from tensordict import TensorDict

from env.mapping_environment import FlagMessage


@dataclass
class PendingAgentTransition:
    state: TensorDict     # the per-agent slice of the observation at decision time
    action: torch.Tensor  # shape (action_dim,)
    reward_sum: float = 0.0
    n_sim_steps: int = 0  # how many sim steps this macro-action has lasted


@dataclass
class PendingGlobalTransition:
    global_state: TensorDict  # snapshot of "global_state" subtree
    joint_action: torch.Tensor  # shape (N, action_dim) — last-committed action per agent
    reward_sum: float = 0.0
    n_sim_steps: int = 0


class AsyncMARLOrchestrator:
    """
    Wraps a MappingEnvironment and produces SMDP-style transitions.

    `policy_fn(per_agent_obs_td) -> action_tensor` is the decision rule
    random, scripted, or a neural network. It only ever sees observations
    for agents that are actually making a decision this step.
    """

    def __init__(self, env, policy_fn: Callable[[TensorDict], torch.Tensor]):
        self.env = env
        self.policy_fn = policy_fn
        self.N = env.max_num_agents
        self.action_dim = env.action_spec["agents", "action"].shape[-1]

        self.pending_agents: list[PendingAgentTransition | None] = [None] * self.N
        self.pending_global: PendingGlobalTransition | None = None
        
        # The most recent action each agent committed to. Updated only on flag events.
        self.last_committed_action = torch.zeros(self.N, self.action_dim)

        # Output buffers
        self.agent_transitions: list[dict] = []
        self.global_transitions: list[dict] = []


    @staticmethod
    def _read_flags(obs_td: TensorDict) -> torch.Tensor:
        """Return a (N,) tensor of integer flag indices."""
        return obs_td["agents", "observation", "encounter_flag"].argmax(dim=-1)

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
        self.pending_agents = [None] * self.N
        self.pending_global = None
        self.last_committed_action.zero_()

        # Output buffers
        self.agent_transitions: list[dict] = []
        self.global_transitions: list[dict] = []

        # At reset all real agents are flagged INTERNAL — open transitions for them.
        self._handle_flags(td, opening_episode=True)
        return td

    def step(self, td: TensorDict) -> TensorDict:
        # Random actions for unflagged agents (they'll be discarded by the env's gate anyway).
        td = self.env.rand_action(td)

        # Overwrite the rows where we have a committed action from the policy.
        action = td["agents", "action"]
        for i in range(self.N):
            if self.pending_agents[i] is not None and self.pending_agents[i].n_sim_steps == 0:
                action[i] = self.pending_agents[i].action

        # Step.
        td = self.env.step(td)

        # Accumulate rewards into all pending transitions.
        rewards = td["next", "agents", "reward"].squeeze(-1)
        global_reward = td["next", "global_reward"].item()
        mask = td["next", "agents", "mask"]

        for i in range(self.N):
            p = self.pending_agents[i]
            if p is not None and mask[i]:
                p.reward_sum += rewards[i].item()
                p.n_sim_steps += 1

        if self.pending_global is not None:
            self.pending_global.reward_sum += global_reward
            self.pending_global.n_sim_steps += 1

        # Episode end.
        if td["next", "done"].item():
            self._close_all_pending(td["next"], terminal=True)
            return td

        # Per-agent deaths.
        per_agent_done = td["next", "agents", "done"].squeeze(-1)
        for i in range(self.N):
            if per_agent_done[i] and self.pending_agents[i] is not None:
                self._close_agent(i, td["next"], terminal=True)

        # New flags at t+1.
        self._handle_flags(td["next"])

        return td

    ##### flag handling 

    def _handle_flags(self, obs_td: TensorDict, opening_episode: bool = False):
        flags = self._read_flags(obs_td)
        mask = obs_td["agents", "mask"]
        any_flag_fired = False

        for i in range(self.N):
            if not mask[i]:
                continue
            f = flags[i].item()
            if f == FlagMessage.NONE.value and not opening_episode:
                continue
            any_flag_fired = True

            # Close the existing pending (if any), using the current obs as s'.
            if self.pending_agents[i] is not None:
                self._close_agent(i, obs_td, terminal=False)

            # Ask the policy for a new action.
            per_agent_obs = self._slice_agent_obs(obs_td, i)
            new_action = self.policy_fn(per_agent_obs)

            # Just store it. The next call to step() will copy it into the td.
            self.last_committed_action[i] = new_action

            self.pending_agents[i] = PendingAgentTransition(
                state=per_agent_obs,
                action=new_action.clone(),
            )

        if any_flag_fired:
            if self.pending_global is not None:
                self._close_global(obs_td, terminal=False)
            self.pending_global = PendingGlobalTransition(
                global_state=self._snapshot_global(obs_td),
                joint_action=self.last_committed_action.clone(),
            )

    def _close_agent(self, i: int, next_obs_td: TensorDict, terminal: bool):
        p = self.pending_agents[i]
        self.agent_transitions.append({
            "state": p.state,
            "action": p.action,
            "reward": p.reward_sum,
            "next_state": self._slice_agent_obs(next_obs_td, i),
            "done": terminal,
            "n_sim_steps": p.n_sim_steps,
            "agent_idx": i,
        })
        self.pending_agents[i] = None

    def _close_global(self, next_obs_td: TensorDict, terminal: bool):
        g = self.pending_global
        self.global_transitions.append({
            "state": g.global_state,
            "joint_action": g.joint_action,
            "reward": g.reward_sum,
            "next_state": self._snapshot_global(next_obs_td),
            "done": terminal,
            "n_sim_steps": g.n_sim_steps,
        })
        self.pending_global = None

    def _close_all_pending(self, next_obs_td: TensorDict, terminal: bool):
        for i in range(self.N):
            if self.pending_agents[i] is not None:
                self._close_agent(i, next_obs_td, terminal=terminal)
        if self.pending_global is not None:
            self._close_global(next_obs_td, terminal=terminal)

    def random_policy(per_agent_obs):
        return torch.rand(2)

