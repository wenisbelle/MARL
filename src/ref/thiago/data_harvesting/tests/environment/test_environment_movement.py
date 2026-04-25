import math
from typing import cast

import pytest
import torch

from data_harvesting.environment.data_collection import make_data_collection_env
from gradysim.simulator.simulation import Simulator


RIGHT = 0.0
LEFT = 0.5


def _single_drone_config() -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 1,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 100.0,
            "max_episode_length": 100,
            "max_seconds_stalled": 1000,
            "communication_range": 0.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
        }
    }


def _get_drone_xy(env) -> tuple[float, float]:
    active_agent = next(agent for agent in env.episode_agents if agent.exists and agent.active)
    node = env.simulator.get_node(active_agent.node_id)
    return float(node.position[0]), float(node.position[1])


def _step_once(env, *, direction: float, speed: float, seed: int | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
    td = env.reset(seed=seed)
    start = _get_drone_xy(env)

    action = torch.tensor([[direction, speed]], dtype=torch.float32, device=env.device)
    td.set(("agents", "action"), action)
    env.step(td)

    end = _get_drone_xy(env)
    return start, end


def test_drone_is_ready_after_reset() -> None:
    env = make_data_collection_env(_single_drone_config())
    try:
        env.reset(seed=999)
        active_agent = next(agent for agent in env.episode_agents if agent.exists and agent.active)
        protocol = cast(Simulator, env.simulator).get_node(cast(int, active_agent.node_id)).protocol_encapsulator.protocol
        assert protocol.ready is True
        assert protocol.current_position is not None
    finally:
        env.close()


def test_one_drone_moves_right_with_right_action() -> None:
    env = make_data_collection_env(_single_drone_config())
    try:
        start, end = _step_once(env, direction=RIGHT, speed=1.0, seed=123)
    finally:
        env.close()

    assert end[0] > start[0]


def test_one_drone_moves_left_with_left_action() -> None:
    env = make_data_collection_env(_single_drone_config())
    try:
        start, end = _step_once(env, direction=LEFT, speed=1.0, seed=456)
    finally:
        env.close()

    assert end[0] < start[0]


def test_one_drone_does_not_move_with_zero_speed() -> None:
    env = make_data_collection_env(_single_drone_config())
    try:
        start, end = _step_once(env, direction=RIGHT, speed=0.0, seed=789)
    finally:
        env.close()

    displacement = math.dist(start, end)
    assert displacement == pytest.approx(0.0, abs=1e-4)


def test_one_drone_moves_half_distance_at_half_speed() -> None:
    env = make_data_collection_env(_single_drone_config())
    try:
        start_half, end_half = _step_once(env, direction=RIGHT, speed=0.5, seed=31415)
        start_full, end_full = _step_once(env, direction=RIGHT, speed=1.0, seed=31415)
    finally:
        env.close()

    half_dx = end_half[0] - start_half[0]
    full_dx = end_full[0] - start_full[0]

    assert half_dx > 0
    assert full_dx > 0
    assert full_dx == pytest.approx(2 * half_dx, rel=0.15, abs=1e-3)
