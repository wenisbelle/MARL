from typing import cast
import torch

from data_harvesting.environment.data_collection import make_data_collection_env


def _masking_config(*, sequential_obs: bool = True) -> dict:
    return {
        "environment": {
            "sequential_obs": sequential_obs,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 4,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 10.0,
            "max_episode_length": 20,
            "max_seconds_stalled": 10,
            "communication_range": 0.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": True,
        }
    }


def _episode_agent_slots(env) -> tuple[list[int], list[int]]:
    active_slots = [
        agent.slot_index
        for agent in env.episode_agents
        if agent.exists and agent.active
    ]
    inactive_slots = [
        agent.slot_index
        for agent in env.episode_agents
        if not agent.exists
    ]
    return active_slots, inactive_slots


def _reset_until(env, predicate, max_seed: int = 200):
    for seed in range(max_seed):
        td = env.reset(seed=seed)
        if predicate(env):
            return td
    raise AssertionError("Could not find reset matching mask test condition")


def test_mask_marks_only_active_agents_on_reset() -> None:
    env = make_data_collection_env(_masking_config())
    try:
        td = _reset_until(env, lambda env: any(not agent.exists for agent in env.episode_agents))

        active, inactive = _episode_agent_slots(env)
        max_drones = cast(int, env.max_num_agents)
        assert len(env.episode_agents) == max_drones
        mask = td.get(("agents", "mask"))
        assert tuple(mask.shape) == (max_drones,)
        assert mask.dtype == torch.bool
        assert mask[active].tolist() == [True] * len(active)
        assert mask[inactive].tolist() == [False] * len(inactive)
        assert [agent.exists for agent in env.episode_agents] == [True] * len(active) + [False] * len(inactive)
        assert [agent.active for agent in env.episode_agents] == [True] * len(active) + [False] * len(inactive)

        inactive_done = td.get(("agents", "done"))[inactive, 0]
        inactive_truncated = td.get(("agents", "truncated"))[inactive, 0]
        assert inactive_done.tolist() == [True] * len(inactive)
        assert inactive_truncated.tolist() == [True] * len(inactive)
    finally:
        env.close()


def test_mask_stays_consistent_after_step() -> None:
    env = make_data_collection_env(_masking_config())
    try:
        td = _reset_until(env, lambda env: any(not agent.exists for agent in env.episode_agents))
        active, inactive = _episode_agent_slots(env)
        max_drones = cast(int, env.max_num_agents)

        action = torch.zeros((max_drones, 2), dtype=torch.float32, device=env.device)
        td.set(("agents", "action"), action)

        td = env.step(td)
        next_td = td.get("next")
        mask = next_td.get(("agents", "mask"))

        assert mask[active].tolist() == [True] * len(active)
        assert mask[inactive].tolist() == [False] * len(inactive)
        assert next_td.get(("agents", "truncated"))[inactive, 0].tolist() == [True] * len(inactive)
    finally:
        env.close()


def test_mask_all_true_when_active_equals_max() -> None:
    env = make_data_collection_env(_masking_config())
    try:
        td = _reset_until(env, lambda env: all(agent.exists for agent in env.episode_agents))

        mask = td.get(("agents", "mask"))
        assert mask.tolist() == [True] * cast(int, env.max_num_agents)
    finally:
        env.close()


def test_mask_spec_shape_matches_max_drones() -> None:
    env = make_data_collection_env(_masking_config(sequential_obs=False))
    try:
        assert tuple(env.observation_spec["agents", "mask"].shape) == (4,)
    finally:
        env.close()
