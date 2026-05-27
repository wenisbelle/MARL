"""
main_test_workers.py — smoke test for the patched worker / WorkersOrchestrator.

Sync lifecycle (per iteration):
    set_weights -> broadcast -> resume -> collect (returns with workers PAUSED)
    -> train -> next iteration

Workers are bootstrap-paused on startup, so iteration 0 follows the same
pattern as every other iteration.
"""

import multiprocessing as mp
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
import time

from env.mapping_environment import MappingEnvironment, MappingEnvironmentConfig
from workers_orchestrator import WorkersOrchestrator


def make_env():
    config = MappingEnvironmentConfig(
        render_mode="visual",
        algorithm_iteration_interval=1.0,
        min_num_agents=3,
        max_num_agents=3,
        map_width=50,
        map_height=50,
        observation_map_size=20,
        max_episode_length=1000,
        agent_death_probability=0.0,
    )
    return MappingEnvironment(config)


class TinyPolicy(nn.Module):
    def __init__(self, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, per_agent_obs_td):
        pos = per_agent_obs_td["position"].to(torch.float32)
        return self.net(pos)

class RandomPolicy(nn.Module):
    """
    Stateless random policy. Ignores the observation, returns a uniformly
    random action in [0, 1]^action_dim — matches the env's Bounded action spec.

    No parameters: state_dict() is empty, load_state_dict({}) is a no-op,
    so the broadcast plumbing still runs cleanly.
    """
    def __init__(self, action_dim: int = 2):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, per_agent_obs_td):
        return torch.rand(self.action_dim)


def make_policy():
    return RandomPolicy(action_dim=2)



def main():
    NUM_WORKERS                 = 1
    STEPS_PER_BATCH             = 50
    NUM_ITERATIONS              = 2
    MIN_TRANSITIONS_PER_COLLECT = 10
    COLLECT_TIMEOUT_S           = 500.0

    agent_buffer  = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=20_000))
    global_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=20_000))

    trainer_policy = make_policy()

    with WorkersOrchestrator(
        num_workers=NUM_WORKERS,
        env_fn=make_env,
        policy_fn=make_policy,
        agent_buffer=agent_buffer,
        global_buffer=global_buffer,
        steps_per_batch=STEPS_PER_BATCH,
        base_seed=42,
        sync=True,
        new_batch_new_simulation=False,
    ) as orch:

        for it in range(NUM_ITERATIONS):
            # Same four-step rhythm on every iteration, including the first.
            orch.set_weights(trainer_policy.state_dict())
            orch.broadcast()
            orch.resume()
            new = orch.collect(
                min_new_transitions=MIN_TRANSITIONS_PER_COLLECT,
                timeout=COLLECT_TIMEOUT_S,
            )

            print(
                f"[iter {it}] new agent transitions={new}  "
                f"agent_buffer={len(agent_buffer)}  "
                f"global_buffer={len(global_buffer)}"
            )
            # stop the execution for 100 ms
            time.sleep(0.1)

            with torch.no_grad():
                for p in trainer_policy.parameters():
                    p.add_(0.01 * torch.randn_like(p))

        if len(agent_buffer) > 0:
            sample = agent_buffer.sample(1)
            print("\nAgent sample keys:", list(sample.keys()))
            print("  reward      :", sample["reward"].squeeze().tolist())
            print("  n_sim_steps :", sample["n_sim_steps"].squeeze().tolist())
            print("  agent_idx   :", sample["agent_idx"].squeeze().tolist())
            print("  done        :", sample["done"].squeeze().tolist())

    print("\nShutdown complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()