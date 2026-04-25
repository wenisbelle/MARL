import torch

from data_harvesting.environment import EndCause
from data_harvesting.environment.data_collection import make_data_collection_env


RIGHT = 0.0
LEFT = 0.5


def _one_drone_one_sensor_config() -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 1,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 10.0,
            "max_episode_length": 50,
            "max_seconds_stalled": 2,
            "communication_range": 2.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": True,
        }
    }


def _config_with_overrides(**env_overrides) -> dict:
    config = _one_drone_one_sensor_config()
    config["environment"].update(env_overrides)
    return config


def _prepare_scenario(env, *, drone_x: float, sensor_x: float) -> tuple:
    active_agent = next(agent for agent in env.episode_agents if agent.exists and agent.active)
    drone_node = env.simulator.get_node(active_agent.node_id)
    sensor_node = env.simulator.get_node(env.sensor_node_ids[0])

    drone_node.position = (drone_x, 0.0, 0.0)
    sensor_node.position = (sensor_x, 0.0, 0.0)

    drone_protocol = drone_node.protocol_encapsulator.protocol
    drone_protocol.current_position = (drone_x, 0.0, 0.0)
    drone_protocol.ready = True

    sensor_protocol = sensor_node.protocol_encapsulator.protocol
    sensor_protocol.has_collected = False
    return drone_node, sensor_node

def _terminal_cause(next_td) -> int:
    return int(next_td.get(("agents", "info", "cause"))[0].item())


def test_sensor_at_origin_is_collected_and_episode_ends_first_step() -> None:
    env = make_data_collection_env(_one_drone_one_sensor_config())
    try:
        td = env.reset(seed=123)
        _, sensor_node = _prepare_scenario(env, drone_x=0.0, sensor_x=0.0)

        action = torch.tensor([[RIGHT, 0.0]], dtype=torch.float32, device=env.device)
        td.set(("agents", "action"), action)
        td = env.step(td)
        next_td = td.get("next")

        assert bool(sensor_node.protocol_encapsulator.protocol.has_collected) is True
        assert bool(next_td.get("done").item()) is True
        assert _terminal_cause(next_td) == EndCause.ALL_COLLECTED.value
    finally:
        env.close()


def test_sensor_at_right_edge_collected_when_moving_right() -> None:
    env = make_data_collection_env(_one_drone_one_sensor_config())
    try:
        td = env.reset(seed=123)
        _, sensor_node = _prepare_scenario(env, drone_x=0.0, sensor_x=10.0)

        action = torch.tensor([[RIGHT, 1.0]], dtype=torch.float32, device=env.device)
        td.set(("agents", "action"), action)
        td = env.step(td)
        next_td = td.get("next")

        assert bool(sensor_node.protocol_encapsulator.protocol.has_collected) is True
        assert bool(next_td.get("done").item()) is True
        assert _terminal_cause(next_td) == EndCause.ALL_COLLECTED.value
    finally:
        env.close()


def test_sensor_not_collected_when_moving_away_and_episode_ends_stalled() -> None:
    env = make_data_collection_env(_one_drone_one_sensor_config())
    try:
        td = env.reset(seed=123)
        _, sensor_node = _prepare_scenario(env, drone_x=0.0, sensor_x=10.0)

        action = torch.tensor([[LEFT, 1.0]], dtype=torch.float32, device=env.device)

        final_next = None
        for step in range(1, 12):
            td.set(("agents", "action"), action)
            td = env.step(td)
            next_td = td.get("next")
            if bool(next_td.get("done").item()):
                final_next = next_td
                assert step > 1
                break
            td = next_td

        if final_next is None:
            raise AssertionError("Episode did not terminate with stall as expected")

        assert bool(sensor_node.protocol_encapsulator.protocol.has_collected) is False
        assert _terminal_cause(final_next) == EndCause.STALLED.value
    finally:
        env.close()


def test_sensor_not_collected_when_moving_away_and_episode_ends_timeout() -> None:
    env = make_data_collection_env(
        _config_with_overrides(
            max_episode_length=3,
            max_seconds_stalled=100,
        )
    )
    try:
        td = env.reset(seed=123)
        _, sensor_node = _prepare_scenario(env, drone_x=0.0, sensor_x=10.0)

        action = torch.tensor([[LEFT, 1.0]], dtype=torch.float32, device=env.device)

        final_next = None
        for _ in range(12):
            td.set(("agents", "action"), action)
            td = env.step(td)
            next_td = td.get("next")
            if bool(next_td.get("done").item()):
                final_next = next_td
                break
            td = next_td

        if final_next is None:
            raise AssertionError("Episode did not terminate with timeout as expected")

        assert bool(sensor_node.protocol_encapsulator.protocol.has_collected) is False
        assert _terminal_cause(final_next) == EndCause.TIMEOUT.value
    finally:
        env.close()
