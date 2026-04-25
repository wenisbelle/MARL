import pytest

from data_harvesting.environment.data_collection import make_data_collection_env


def _shape_config(
    *,
    sequential_obs: bool,
    id_on_state: bool,
    num_drones: int,
    closest_sensors: int,
    closest_drones: int,
) -> dict:
    return {
        "environment": {
            "sequential_obs": sequential_obs,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": num_drones,
            "max_num_agents": num_drones,
            "min_num_sensors": 2,
            "max_num_sensors": 2,
            "scenario_size": 10.0,
            "max_episode_length": 50,
            "max_seconds_stalled": 10,
            "communication_range": 0.0,
            "state_num_closest_sensors": closest_sensors,
            "state_num_closest_drones": closest_drones,
            "id_on_state": id_on_state,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": True,
        }
    }


@pytest.mark.parametrize("closest_sensors,closest_drones", [(1, 2), (4, 3)])
def test_sequential_observation_and_spec_shapes_with_agent_id(closest_sensors: int, closest_drones: int) -> None:
    num_drones = 3
    env = make_data_collection_env(
        _shape_config(
            sequential_obs=True,
            id_on_state=True,
            num_drones=num_drones,
            closest_sensors=closest_sensors,
            closest_drones=closest_drones,
        )
    )
    try:
        td = env.reset(seed=21)

        assert tuple(td.get(("agents", "observation", "sensors")).shape) == (num_drones, closest_sensors, 2)
        assert tuple(td.get(("agents", "observation", "drones")).shape) == (num_drones, closest_drones, 2)
        assert tuple(td.get(("agents", "observation", "agent_id")).shape) == (num_drones, 1)

        assert tuple(env.observation_spec["agents", "observation", "sensors"].shape) == (num_drones, closest_sensors, 2)
        assert tuple(env.observation_spec["agents", "observation", "drones"].shape) == (num_drones, closest_drones, 2)
        assert tuple(env.observation_spec["agents", "observation", "agent_id"].shape) == (num_drones, 1)
    finally:
        env.close()


def test_sequential_observation_and_spec_shapes_without_agent_id() -> None:
    num_drones = 3
    closest_sensors = 2
    closest_drones = 4
    env = make_data_collection_env(
        _shape_config(
            sequential_obs=True,
            id_on_state=False,
            num_drones=num_drones,
            closest_sensors=closest_sensors,
            closest_drones=closest_drones,
        )
    )
    try:
        td = env.reset(seed=22)

        assert tuple(td.get(("agents", "observation", "sensors")).shape) == (num_drones, closest_sensors, 2)
        assert tuple(td.get(("agents", "observation", "drones")).shape) == (num_drones, closest_drones, 2)
        assert ("agents", "observation", "agent_id") not in td.keys(True, True)
        assert ("agents", "observation", "agent_id") not in env.observation_spec.keys(True, True)
    finally:
        env.close()


@pytest.mark.parametrize("closest_sensors,closest_drones", [(3, 4), (2, 1)])
def test_flat_observation_and_spec_shapes_with_agent_id(closest_sensors: int, closest_drones: int) -> None:
    num_drones = 2
    env = make_data_collection_env(
        _shape_config(
            sequential_obs=False,
            id_on_state=True,
            num_drones=num_drones,
            closest_sensors=closest_sensors,
            closest_drones=closest_drones,
        )
    )
    try:
        td = env.reset(seed=23)
        expected_dim = closest_sensors * 2 + closest_drones * 2 + 1

        assert tuple(td.get(("agents", "observation")).shape) == (num_drones, expected_dim)
        assert tuple(td.get(("agents", "observation_flat", "sensors")).shape) == (num_drones, closest_sensors * 2)
        assert tuple(td.get(("agents", "observation_flat", "drones")).shape) == (num_drones, closest_drones * 2)
        assert ("agents", "observation", "agent_id") not in td.keys(True, True)

        assert tuple(env.observation_spec["agents", "observation"].shape) == (num_drones, expected_dim)
        assert tuple(env.observation_spec["agents", "observation_flat", "sensors"].shape) == (num_drones, closest_sensors * 2)
        assert tuple(env.observation_spec["agents", "observation_flat", "drones"].shape) == (num_drones, closest_drones * 2)
    finally:
        env.close()


def test_flat_observation_and_spec_shapes_without_agent_id() -> None:
    num_drones = 2
    closest_sensors = 3
    closest_drones = 4
    env = make_data_collection_env(
        _shape_config(
            sequential_obs=False,
            id_on_state=False,
            num_drones=num_drones,
            closest_sensors=closest_sensors,
            closest_drones=closest_drones,
        )
    )
    try:
        td = env.reset(seed=24)
        expected_dim = closest_sensors * 2 + closest_drones * 2

        assert tuple(td.get(("agents", "observation")).shape) == (num_drones, expected_dim)
        assert tuple(td.get(("agents", "observation_flat", "sensors")).shape) == (num_drones, closest_sensors * 2)
        assert tuple(td.get(("agents", "observation_flat", "drones")).shape) == (num_drones, closest_drones * 2)
        assert ("agents", "observation", "agent_id") not in td.keys(True, True)

        assert tuple(env.observation_spec["agents", "observation"].shape) == (num_drones, expected_dim)
        assert tuple(env.observation_spec["agents", "observation_flat", "sensors"].shape) == (num_drones, closest_sensors * 2)
        assert tuple(env.observation_spec["agents", "observation_flat", "drones"].shape) == (num_drones, closest_drones * 2)
    finally:
        env.close()
