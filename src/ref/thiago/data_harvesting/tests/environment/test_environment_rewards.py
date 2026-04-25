import pytest
import torch

from data_harvesting.environment.data_collection import make_data_collection_env


RIGHT = 0.0


def _reward_config(*, num_sensors: int, communication_range: float = 2.0) -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 1,
            "min_num_sensors": num_sensors,
            "max_num_sensors": num_sensors,
            "scenario_size": 10.0,
            "max_episode_length": 50,
            "max_seconds_stalled": 20,
            "communication_range": communication_range,
            "state_num_closest_sensors": max(1, num_sensors),
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": True,
        }
    }


def _prepare_scenario(
    env,
    *,
    drone_pos: tuple[float, float],
    sensor_positions: list[tuple[float, float]],
    collected_flags: list[bool],
) -> None:
    active_agent = next(agent for agent in env.episode_agents if agent.exists and agent.active)
    drone_node = env.simulator.get_node(active_agent.node_id)
    drone_node.position = (drone_pos[0], drone_pos[1], 0.0)
    drone_protocol = drone_node.protocol_encapsulator.protocol
    drone_protocol.current_position = (drone_pos[0], drone_pos[1], 0.0)
    drone_protocol.ready = True

    for index, (position, collected) in enumerate(zip(sensor_positions, collected_flags, strict=True)):
        sensor_node = env.simulator.get_node(env.sensor_node_ids[index])
        sensor_node.position = (position[0], position[1], 0.0)
        sensor_node.protocol_encapsulator.protocol.has_collected = collected


def _step_and_get_reward(env, td, *, direction: float = RIGHT, speed: float = 0.0) -> tuple[object, float]:
    action = torch.tensor([[direction, speed]], dtype=torch.float32, device=env.device)
    td.set(("agents", "action"), action)
    td = env.step(td)
    next_td = td.get("next")
    reward = float(next_td.get(("agents", "reward"))[0, 0].item())
    return next_td, reward


@pytest.mark.parametrize("num_sensors", [1, 2])
def test_reward_is_proportional_to_number_of_collected_sensors(num_sensors: int) -> None:
    env = make_data_collection_env(_reward_config(num_sensors=num_sensors, communication_range=3.0))
    try:
        td = env.reset(seed=31)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(0.0, 0.0)] * num_sensors,
            collected_flags=[False] * num_sensors,
        )

        _, reward = _step_and_get_reward(env, td, speed=0.0)

        assert reward == pytest.approx(10.0 * num_sensors)
    finally:
        env.close()


def test_reward_punishes_for_each_remaining_sensor() -> None:
    env = make_data_collection_env(_reward_config(num_sensors=2, communication_range=0.0))
    try:
        td = env.reset(seed=32)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(9.0, 0.0), (9.0, 1.0)],
            collected_flags=[True, False],
        )

        _, reward = _step_and_get_reward(env, td, speed=0.0)

        assert reward == pytest.approx(-0.5)
    finally:
        env.close()


def test_reward_is_full_penalty_when_no_sensor_collected() -> None:
    env = make_data_collection_env(_reward_config(num_sensors=2, communication_range=0.0))
    try:
        td = env.reset(seed=34)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(9.0, 0.0), (9.0, 1.0)],
            collected_flags=[False, False],
        )

        _, reward = _step_and_get_reward(env, td, speed=0.0)

        assert reward == pytest.approx(-1.0)
    finally:
        env.close()


def test_reward_penalty_ceases_when_all_sensors_already_collected() -> None:
    env = make_data_collection_env(_reward_config(num_sensors=2, communication_range=0.0))
    try:
        td = env.reset(seed=33)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(9.0, 0.0), (9.0, 1.0)],
            collected_flags=[True, True],
        )

        _, reward = _step_and_get_reward(env, td, speed=0.0)

        assert reward == pytest.approx(0.0)
    finally:
        env.close()
