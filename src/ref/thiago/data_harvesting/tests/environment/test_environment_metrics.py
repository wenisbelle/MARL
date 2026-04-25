import pytest
import torch

from data_harvesting.environment.data_collection import make_data_collection_env
from data_harvesting.environment import EndCause


RIGHT = 0.0


def _metrics_config(
    *,
    num_sensors: int,
    communication_range: float,
    max_seconds_stalled: int,
    max_episode_length: int = 50,
    end_when_all_collected: bool = True,
    min_num_agents: int = 1,
    max_num_agents: int = 1,
) -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": min_num_agents,
            "max_num_agents": max_num_agents,
            "min_num_sensors": num_sensors,
            "max_num_sensors": num_sensors,
            "scenario_size": 10.0,
            "max_episode_length": max_episode_length,
            "max_seconds_stalled": max_seconds_stalled,
            "communication_range": communication_range,
            "state_num_closest_sensors": max(1, num_sensors),
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": end_when_all_collected,
        }
    }


def _reset_until(env, predicate, max_seed: int = 200):
    for seed in range(max_seed):
        td = env.reset(seed=seed)
        if predicate(env):
            return td
    raise AssertionError("Could not find reset matching requested condition")


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


def _step(env, td, *, direction: float = RIGHT, speed: float = 0.0):
    action = torch.tensor([[direction, speed]], dtype=torch.float32, device=env.device)
    td.set(("agents", "action"), action)
    td = env.step(td)
    return td.get("next")


def _metrics(next_td) -> dict[str, float]:
    info = next_td.get(("agents", "info"))
    return {
        "avg_reward": float(info.get("avg_reward")[0].item()),
        "max_reward": float(info.get("max_reward")[0].item()),
        "sum_reward": float(info.get("sum_reward")[0].item()),
        "avg_collection_time": float(info.get("avg_collection_time")[0].item()),
        "episode_duration": float(info.get("episode_duration")[0].item()),
        "completion_time": float(info.get("completion_time")[0].item()),
        "all_collected": float(info.get("all_collected")[0].item()),
        "num_collected": float(info.get("num_collected")[0].item()),
        "cause": float(info.get("cause")[0].item()),
    }


def test_all_collected_metrics_are_reported_correctly() -> None:
    env = make_data_collection_env(_metrics_config(num_sensors=2, communication_range=3.0, max_seconds_stalled=20, max_episode_length=50))
    try:
        td = env.reset(seed=41)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(0.0, 0.0), (0.0, 0.0)],
            collected_flags=[False, False],
        )

        next_td = _step(env, td, speed=0.0)
        metrics = _metrics(next_td)

        assert bool(next_td.get("done").item()) is True
        assert metrics["avg_reward"] == pytest.approx(20.0)
        assert metrics["max_reward"] == pytest.approx(20.0)
        assert metrics["sum_reward"] == pytest.approx(20.0)
        assert metrics["avg_collection_time"] == pytest.approx(1.0)
        assert metrics["episode_duration"] == pytest.approx(1.0)
        assert metrics["completion_time"] == pytest.approx(env.simulator._current_timestamp)
        assert metrics["all_collected"] == pytest.approx(1.0)
        assert metrics["num_collected"] == pytest.approx(2.0)
        assert metrics["cause"] == pytest.approx(float(EndCause.ALL_COLLECTED.value))
    finally:
        env.close()


def test_stalled_metrics_are_reported_correctly() -> None:
    env = make_data_collection_env(_metrics_config(num_sensors=1, communication_range=0.0, max_seconds_stalled=2, max_episode_length=50))
    try:
        td = env.reset(seed=42)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(9.0, 0.0)],
            collected_flags=[False],
        )

        next_td = None
        for _ in range(6):
            next_td = _step(env, td, speed=0.0)
            if bool(next_td.get("done").item()):
                break
            td = next_td

        assert next_td is not None
        assert bool(next_td.get("done").item()) is True
        metrics = _metrics(next_td)

        assert metrics["avg_reward"] == pytest.approx(-1.0)
        assert metrics["max_reward"] == pytest.approx(-1.0)
        assert metrics["sum_reward"] == pytest.approx(-3.0)
        assert metrics["avg_collection_time"] == pytest.approx(50.0)
        assert metrics["episode_duration"] == pytest.approx(3.0)
        assert metrics["completion_time"] == pytest.approx(50.0)
        assert metrics["all_collected"] == pytest.approx(0.0)
        assert metrics["num_collected"] == pytest.approx(0.0)
        assert metrics["cause"] == pytest.approx(float(EndCause.STALLED.value))
    finally:
        env.close()


def test_metrics_are_zero_when_episode_not_ended() -> None:
    env = make_data_collection_env(_metrics_config(num_sensors=1, communication_range=0.0, max_seconds_stalled=20, max_episode_length=50))
    try:
        td = env.reset(seed=43)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(9.0, 0.0)],
            collected_flags=[False],
        )

        next_td = _step(env, td, speed=0.0)
        metrics = _metrics(next_td)

        assert bool(next_td.get("done").item()) is False
        for value in metrics.values():
            assert value == pytest.approx(0.0)
    finally:
        env.close()


def test_done_flag_ignores_inactive_agent_done_states() -> None:
    env = make_data_collection_env(
        _metrics_config(
            num_sensors=1,
            communication_range=0.0,
            max_seconds_stalled=20,
            min_num_agents=1,
            max_num_agents=2,
        )
    )
    try:
        td = _reset_until(env, lambda env: any(not agent.exists for agent in env.episode_agents))
        assert bool(td.get("done").item()) is False
        assert bool(td.get("terminated").item()) is False
        assert bool(td.get("truncated").item()) is False

        action = torch.zeros((env.max_num_agents, 2), dtype=torch.float32, device=env.device)
        td.set(("agents", "action"), action)
        next_td = env.step(td).get("next")

        assert bool(next_td.get("done").item()) is False
        assert bool(next_td.get("terminated").item()) is False
        assert bool(next_td.get("truncated").item()) is False
    finally:
        env.close()


def test_end_when_all_collected_false_reports_all_collected_on_terminal_step() -> None:
    max_episode_length = 50
    env = make_data_collection_env(
        _metrics_config(
            num_sensors=1,
            communication_range=3.0,
            max_seconds_stalled=2,
            max_episode_length=max_episode_length,
            end_when_all_collected=False,
        )
    )
    try:
        td = env.reset(seed=44)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(0.0, 0.0)],
            collected_flags=[False],
        )
        env.simulator.get_node(env.sensor_node_ids[0]).protocol_encapsulator.protocol.has_collected = True

        first_next = _step(env, td, speed=0.0)
        assert bool(first_next.get("done").item()) is False
        td = first_next

        final_next = None
        for _ in range(7):
            next_td = _step(env, td, speed=0.0)
            if bool(next_td.get("done").item()):
                final_next = next_td
                break
            td = next_td

        assert final_next is not None
        metrics = _metrics(final_next)
        assert metrics["all_collected"] == pytest.approx(1.0)
        assert metrics["num_collected"] == pytest.approx(1.0)
        assert metrics["completion_time"] < max_episode_length
    finally:
        env.close()


def test_timeout_or_stall_without_full_collection_sets_max_completion_time() -> None:
    max_episode_length = 50
    env = make_data_collection_env(
        _metrics_config(
            num_sensors=2,
            communication_range=3.0,
            max_seconds_stalled=2,
            max_episode_length=max_episode_length,
            end_when_all_collected=False,
        )
    )
    try:
        td = env.reset(seed=45)
        _prepare_scenario(
            env,
            drone_pos=(0.0, 0.0),
            sensor_positions=[(0.0, 0.0), (9.0, 0.0)],
            collected_flags=[False, False],
        )

        final_next = None
        for _ in range(8):
            next_td = _step(env, td, speed=0.0)
            if bool(next_td.get("done").item()):
                final_next = next_td
                break
            td = next_td

        assert final_next is not None
        metrics = _metrics(final_next)
        assert metrics["cause"] == pytest.approx(float(EndCause.STALLED.value))
        assert metrics["all_collected"] == pytest.approx(0.0)
        assert metrics["num_collected"] == pytest.approx(1.0)
        assert metrics["completion_time"] == pytest.approx(float(max_episode_length))
    finally:
        env.close()
