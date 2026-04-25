from typing import cast
import numpy as np
import pytest

from data_harvesting.environment.data_collection import make_data_collection_env


def _two_drone_config() -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 2,
            "max_num_agents": 2,
            "min_num_sensors": 2,
            "max_num_sensors": 2,
            "scenario_size": 10.0,
            "max_episode_length": 50,
            "max_seconds_stalled": 10,
            "communication_range": 0.0,
            "state_num_closest_sensors": 3,
            "state_num_closest_drones": 3,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": True,
        }
    }


def _many_entities_config() -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 5,
            "max_num_agents": 5,
            "min_num_sensors": 5,
            "max_num_sensors": 5,
            "scenario_size": 20.0,
            "max_episode_length": 50,
            "max_seconds_stalled": 10,
            "communication_range": 0.0,
            "state_num_closest_sensors": 2,
            "state_num_closest_drones": 2,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": True,
        }
    }


def _config_with_overrides(base_config: dict, **env_overrides) -> dict:
    config = {
        "environment": base_config["environment"].copy(),
    }
    config["environment"].update(env_overrides)
    return config


def _set_drone_positions(env, positions: list[tuple[float, float]]) -> None:
    active_agents = [agent for agent in env.episode_agents if agent.exists and agent.active]
    for agent, (x, y) in zip(active_agents, positions, strict=True):
        node = env.simulator.get_node(agent.node_id)
        node.position = (x, y, 0.0)
        protocol = node.protocol_encapsulator.protocol
        protocol.current_position = (x, y, 0.0)
        protocol.ready = True


def _set_sensor_state(env, positions: list[tuple[float, float]], visited: list[bool]) -> None:
    for index, ((x, y), is_visited) in enumerate(zip(positions, visited, strict=True)):
        node = env.simulator.get_node(env.sensor_node_ids[index])
        node.position = (x, y, 0.0)
        node.protocol_encapsulator.protocol.has_collected = is_visited


def _observe(env) -> dict:
    return env._observe_simulation()


def test_drones_on_opposite_ends_have_opposite_relative_positions() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=1)
        size = cast(int, env.scenario_size)
        _set_drone_positions(env, [(-size, 0.0), (size, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (0.0, 5.0)], [True, True])

        obs = _observe(env)

        drone0_dx = obs["drone0"]["drones"][0, 0]
        drone1_dx = obs["drone1"]["drones"][0, 0]
        drone0_dy = obs["drone0"]["drones"][0, 1]
        drone1_dy = obs["drone1"]["drones"][0, 1]

        assert drone0_dx == pytest.approx(1.0)
        assert drone1_dx == pytest.approx(0.0)
        assert drone0_dy == pytest.approx(0.5)
        assert drone1_dy == pytest.approx(0.5)
    finally:
        env.close()


def test_two_drones_at_origin_have_zero_relative_offset() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=2)
        _set_drone_positions(env, [(0.0, 0.0), (0.0, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (5.0, 0.0)], [True, True])

        obs = _observe(env)
        drone0_dx = obs["drone0"]["drones"][0, 0]
        drone0_dy = obs["drone0"]["drones"][0, 1]

        assert drone0_dx == pytest.approx(0.5)
        assert drone0_dy == pytest.approx(0.5)
    finally:
        env.close()


def test_empty_drone_slots_are_filled_with_minus_one() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=3)
        _set_drone_positions(env, [(-1.0, 0.0), (1.0, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (0.0, 1.0)], [True, True])

        obs = _observe(env)
        drone_slots = obs["drone0"]["drones"]

        assert np.all(drone_slots[1:] == -1)
    finally:
        env.close()


def test_sensor_on_opposite_end_has_expected_relative_position() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=4)
        size = cast(int, env.scenario_size)
        _set_drone_positions(env, [(-size, 0.0), (0.0, 0.0)])
        _set_sensor_state(env, [(size, 0.0), (0.0, 0.0)], [False, True])

        obs = _observe(env)
        sensor_dx = obs["drone0"]["sensors"][0, 0]
        sensor_dy = obs["drone0"]["sensors"][0, 1]

        assert sensor_dx == pytest.approx(1.0)
        assert sensor_dy == pytest.approx(0.5)
    finally:
        env.close()


def test_drone_and_sensor_at_origin_have_zero_relative_sensor_offset() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=5)
        _set_drone_positions(env, [(0.0, 0.0), (1.0, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (0.0, 1.0)], [False, True])

        obs = _observe(env)
        sensor_dx = obs["drone0"]["sensors"][0, 0]
        sensor_dy = obs["drone0"]["sensors"][0, 1]

        assert sensor_dx == pytest.approx(0.5)
        assert sensor_dy == pytest.approx(0.5)
    finally:
        env.close()


def test_visited_sensor_is_excluded_from_observation() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=6)
        size = cast(int, env.scenario_size)
        _set_drone_positions(env, [(-size, 0.0), (0.0, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (size, 0.0)], [True, False])

        obs = _observe(env)
        sensor_dx = obs["drone0"]["sensors"][0, 0]

        assert sensor_dx == pytest.approx(1.0)
    finally:
        env.close()


def test_empty_sensor_slots_are_filled_with_minus_one() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=7)
        _set_drone_positions(env, [(0.0, 0.0), (0.0, 0.0)])
        _set_sensor_state(env, [(1.0, 0.0), (2.0, 0.0)], [False, True])

        obs = _observe(env)
        sensor_slots = obs["drone0"]["sensors"]

        assert np.all(sensor_slots[1:] == -1)
    finally:
        env.close()


def test_drones_on_each_side_of_sensor_have_opposite_relative_signs() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=8)
        _set_drone_positions(env, [(-2.0, 0.0), (2.0, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (5.0, 0.0)], [False, True])

        obs = _observe(env)

        left_dx = obs["drone0"]["sensors"][0, 0]
        right_dx = obs["drone1"]["sensors"][0, 0]

        assert left_dx == pytest.approx(0.55)
        assert right_dx == pytest.approx(0.45)
        assert left_dx + right_dx == pytest.approx(1.0)
    finally:
        env.close()


def test_too_many_drones_keeps_only_closest_ones() -> None:
    env = make_data_collection_env(_many_entities_config())
    try:
        env.reset(seed=9)
        _set_drone_positions(
            env,
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (3.0, 0.0),
                (7.0, 0.0),
                (12.0, 0.0),
            ],
        )
        _set_sensor_state(env, [(0.0, 0.0)] * 5, [True] * 5)

        obs = _observe(env)
        rel_drones = obs["drone0"]["drones"]
        raw_dx = sorted(float(rel_drones[i, 0]) for i in range(rel_drones.shape[0]))

        assert raw_dx == pytest.approx([0.5125, 0.5375])
    finally:
        env.close()


def test_too_many_sensors_keeps_only_closest_ones() -> None:
    env = make_data_collection_env(_many_entities_config())
    try:
        env.reset(seed=10)
        _set_drone_positions(env, [(0.0, 0.0)] * 5)
        _set_sensor_state(
            env,
            [
                (1.0, 0.0),
                (4.0, 0.0),
                (8.0, 0.0),
                (12.0, 0.0),
                (16.0, 0.0),
            ],
            [False] * 5,
        )

        obs = _observe(env)
        rel_sensors = obs["drone0"]["sensors"]
        raw_dx = sorted(float(rel_sensors[i, 0]) for i in range(rel_sensors.shape[0]))

        assert raw_dx == pytest.approx([0.5125, 0.55])
    finally:
        env.close()


def test_agent_id_two_drones_has_expected_values() -> None:
    env = make_data_collection_env(_two_drone_config())
    try:
        env.reset(seed=11)
        _set_drone_positions(env, [(0.0, 0.0), (1.0, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (0.0, 1.0)], [True, True])

        obs = _observe(env)

        assert "agent_id" in obs["drone0"]
        assert "agent_id" in obs["drone1"]
        assert float(obs["drone0"]["agent_id"][0]) == pytest.approx(0.0)
        assert float(obs["drone1"]["agent_id"][0]) == pytest.approx(1.0)
    finally:
        env.close()


def test_agent_id_many_drones_is_evenly_spaced() -> None:
    env = make_data_collection_env(_many_entities_config())
    try:
        env.reset(seed=12)
        _set_drone_positions(env, [(float(i), 0.0) for i in range(5)])
        _set_sensor_state(env, [(0.0, 0.0)] * 5, [True] * 5)

        obs = _observe(env)
        observed_ids = [float(obs[f"drone{i}"]["agent_id"][0]) for i in range(5)]

        assert observed_ids == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    finally:
        env.close()


def test_agent_id_not_present_when_disabled() -> None:
    env = make_data_collection_env(_config_with_overrides(_two_drone_config(), id_on_state=False))
    try:
        env.reset(seed=13)
        _set_drone_positions(env, [(0.0, 0.0), (1.0, 0.0)])
        _set_sensor_state(env, [(0.0, 0.0), (0.0, 1.0)], [True, True])

        obs = _observe(env)

        assert "agent_id" not in obs["drone0"]
        assert "agent_id" not in obs["drone1"]
    finally:
        env.close()
