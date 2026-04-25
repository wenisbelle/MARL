import pytest
import torch

from data_harvesting.environment import EndCause
from data_harvesting.environment.data_collection import make_data_collection_env


def _death_config(*, min_num_agents: int = 2, max_num_agents: int = 2) -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": min_num_agents,
            "max_num_agents": max_num_agents,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 10.0,
            "max_episode_length": 20,
            "max_seconds_stalled": 20,
            "communication_range": 0.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
            "agent_death_probability": 1.0,
        }
    }


def _set_drone_position(env, slot_index: int, x: float, y: float) -> None:
    agent = env.episode_agents[slot_index]
    node = env.simulator.get_node(agent.node_id)
    node.position = (x, y, 0.0)
    protocol = node.protocol_encapsulator.protocol
    protocol.current_position = (x, y, 0.0)
    protocol.ready = True


def _set_single_sensor(env, x: float, y: float, collected: bool = False) -> None:
    node = env.simulator.get_node(env.sensor_node_ids[0])
    node.position = (x, y, 0.0)
    node.protocol_encapsulator.protocol.has_collected = collected


def _zero_action(env) -> torch.Tensor:
    return torch.zeros((env.max_num_agents, 2), dtype=torch.float32, device=env.device)


def test_death_step_terminates_only_the_dying_agent_and_keeps_episode_running() -> None:
    env = make_data_collection_env(_death_config())
    try:
        td = env.reset(seed=11)
        _set_drone_position(env, 0, -1.0, 0.0)
        _set_drone_position(env, 1, 1.0, 0.0)
        _set_single_sensor(env, 9.0, 0.0, collected=False)

        env._sample_dying_agents = lambda stepped_agents: [stepped_agents[0]]

        td.set(("agents", "action"), _zero_action(env))
        next_td = env.step(td).get("next")

        assert env.episode_agents[0].exists is True
        assert env.episode_agents[0].active is False
        assert env.episode_agents[1].active is True

        assert bool(next_td.get("done").item()) is False
        assert bool(next_td.get("terminated").item()) is False
        assert bool(next_td.get("truncated").item()) is False

        agent_done = next_td.get(("agents", "done"))[:, 0].tolist()
        agent_terminated = next_td.get(("agents", "terminated"))[:, 0].tolist()
        agent_truncated = next_td.get(("agents", "truncated"))[:, 0].tolist()
        mask = next_td.get(("agents", "mask")).tolist()
        rewards = next_td.get(("agents", "reward"))[:, 0].tolist()

        assert agent_done == [True, False]
        assert agent_terminated == [True, False]
        assert agent_truncated == [False, False]
        assert mask == [False, True]
        assert rewards[0] == pytest.approx(-1.0)
        assert rewards[1] == pytest.approx(-1.0)

        survivor_drones = next_td.get(("agents", "observation", "drones"))[1]
        assert survivor_drones[0].tolist() == pytest.approx([-1.0, -1.0])
    finally:
        env.close()


def test_dead_agent_becomes_truncated_bookkeeping_on_later_steps() -> None:
    env = make_data_collection_env(_death_config())
    try:
        td = env.reset(seed=12)
        _set_drone_position(env, 0, -1.0, 0.0)
        _set_drone_position(env, 1, 1.0, 0.0)
        _set_single_sensor(env, 9.0, 0.0, collected=False)

        call_count = {"count": 0}

        def _sample(stepped_agents):
            call_count["count"] += 1
            return [stepped_agents[0]] if call_count["count"] == 1 else []

        env._sample_dying_agents = _sample

        td.set(("agents", "action"), _zero_action(env))
        td = env.step(td).get("next")

        td.set(("agents", "action"), _zero_action(env))
        next_td = env.step(td).get("next")

        assert next_td.get(("agents", "mask")).tolist() == [False, True]
        assert next_td.get(("agents", "done"))[:, 0].tolist() == [True, False]
        assert next_td.get(("agents", "terminated"))[:, 0].tolist() == [False, False]
        assert next_td.get(("agents", "truncated"))[:, 0].tolist() == [True, False]
        assert next_td.get(("agents", "reward"))[:, 0].tolist()[0] == pytest.approx(0.0)
        assert bool(next_td.get("done").item()) is False
    finally:
        env.close()


def test_last_agent_death_ends_episode_with_all_agents_dead_cause() -> None:
    env = make_data_collection_env(_death_config(min_num_agents=1, max_num_agents=1))
    try:
        td = env.reset(seed=13)
        _set_drone_position(env, 0, 0.0, 0.0)
        _set_single_sensor(env, 9.0, 0.0, collected=False)

        env._sample_dying_agents = lambda stepped_agents: list(stepped_agents)

        td.set(("agents", "action"), _zero_action(env))
        next_td = env.step(td).get("next")

        assert bool(next_td.get("done").item()) is True
        assert bool(next_td.get("terminated").item()) is True
        assert bool(next_td.get("truncated").item()) is False
        assert next_td.get(("agents", "done"))[:, 0].tolist() == [True]
        assert next_td.get(("agents", "terminated"))[:, 0].tolist() == [True]
        assert next_td.get(("agents", "truncated"))[:, 0].tolist() == [False]
        assert float(next_td.get(("agents", "info", "cause"))[0].item()) == pytest.approx(
            float(EndCause.ALL_AGENTS_INACTIVE.value)
        )
    finally:
        env.close()


def test_death_calls_protocol_die_and_drone_stays_stationary_afterwards() -> None:
    env = make_data_collection_env(_death_config())
    try:
        td = env.reset(seed=14)
        _set_drone_position(env, 0, -1.0, 0.0)
        _set_drone_position(env, 1, 1.0, 0.0)
        _set_single_sensor(env, 9.0, 0.0, collected=False)

        def _sample(stepped_agents):
            return [stepped_agents[0]] if len(stepped_agents) == 2 else []

        env._sample_dying_agents = _sample

        td.set(("agents", "action"), _zero_action(env))
        td = env.step(td).get("next")

        dead_agent = env.episode_agents[0]
        dead_node = env.simulator.get_node(dead_agent.node_id)
        dead_protocol = dead_node.protocol_encapsulator.protocol

        assert dead_protocol.dead is True
        assert dead_node.position[2] == pytest.approx(0.0)

        dead_position_before = dead_node.position
        td.set(("agents", "action"), _zero_action(env))
        env.step(td)
        dead_position_after = dead_node.position

        assert dead_position_after == pytest.approx(dead_position_before)
    finally:
        env.close()
