import pytest
import torch
from torch import nn
from tensordict.nn import TensorDictModule

from data_harvesting.collector import create_collector
from data_harvesting.environment.data_collection import make_data_collection_env


class ConstantDirectionPolicy(nn.Module):
    def __init__(self, direction: float = 0.5, speed: float = 1.0):
        super().__init__()
        self.direction = nn.Parameter(torch.tensor(float(direction), dtype=torch.float32))
        self.speed = nn.Parameter(torch.tensor(float(speed), dtype=torch.float32))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        action_shape = mask.shape + (2,)
        action = torch.zeros(action_shape, dtype=torch.float32, device=mask.device)
        action[..., 0] = torch.clamp(self.direction, 0.0, 1.0)
        action[..., 1] = torch.clamp(self.speed, 0.0, 1.0)
        return action


def _make_policy(direction: float = 0.5, speed: float = 1.0) -> TensorDictModule:
    return TensorDictModule(
        module=ConstantDirectionPolicy(direction=direction, speed=speed),
        in_keys=[("agents", "mask")],
        out_keys=[("agents", "action")],
    )


def _collector_config(*, async_collector: bool, num_collectors: int, frames_per_batch: int = 6, total_timesteps: int = 48) -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 1,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 100.0,
            "max_episode_length": 200,
            "max_seconds_stalled": 100,
            "communication_range": 0.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
        },
        "collector": {
            "num_collectors": num_collectors,
            "frames_per_batch": frames_per_batch,
            "async_collector": async_collector,
            "device": "cpu",
        },
        "training": {
            "total_timesteps": total_timesteps,
        },
    }


@pytest.mark.parametrize(
    "async_collector,num_collectors,frames_per_batch",
    [
        (False, 1, 6),
        (False, 2, 6),
        (True, 1, 6),
        (True, 2, 6),
    ],
)
def test_collector_modes_return_expected_shapes_and_device(
    async_collector: bool,
    num_collectors: int,
    frames_per_batch: int,
) -> None:
    config = _collector_config(
        async_collector=async_collector,
        num_collectors=num_collectors,
        frames_per_batch=frames_per_batch,
        total_timesteps=256,
    )
    policy = _make_policy(direction=0.5, speed=1.0)

    with create_collector(policy, torch.device("cpu"), lambda: make_data_collection_env(config), config) as collector:
        iterator = iter(collector)
        batch = next(iterator)

        assert batch.numel() == frames_per_batch
        assert batch.device is not None
        assert batch.device.type == "cpu"
        assert tuple(batch.get(("agents", "action")).shape[: len(batch.shape)]) == tuple(batch.shape)
        assert batch.get(("agents", "action")).shape[-2:] == (1, 2)
        assert batch.get(("agents", "observation", "sensors")).shape[-3:] == (1, 1, 2)


@pytest.mark.parametrize(
    "async_collector,num_collectors",
    [
        (False, 1),
        (False, 2),
        (True, 1),
        (True, 2),
    ],
)
def test_collector_policy_update_changes_actions_and_observation_trend(
    async_collector: bool,
    num_collectors: int,
) -> None:
    config = _collector_config(
        async_collector=async_collector,
        num_collectors=num_collectors,
        frames_per_batch=8,
        total_timesteps=512,
    )
    policy = _make_policy(direction=0.5, speed=1.0)

    with create_collector(policy, torch.device("cpu"), lambda: make_data_collection_env(config), config) as collector:
        iterator = iter(collector)
        batch_left = next(iterator)

        left_direction_mean = float(batch_left.get(("agents", "action"))[..., 0].mean().item())
        left_sensor = batch_left.get(("agents", "observation", "sensors"))[..., 0, 0]
        left_next_sensor = batch_left.get(("next", "agents", "observation", "sensors"))[..., 0, 0]
        left_delta = float((left_next_sensor - left_sensor).mean().item())

        with torch.no_grad():
            policy.module.direction.fill_(0.0)
        collector.update_policy_weights_()

        if async_collector:
            _ = next(iterator)

        batch_right = next(iterator)
        right_direction_mean = float(batch_right.get(("agents", "action"))[..., 0].mean().item())
        right_sensor = batch_right.get(("agents", "observation", "sensors"))[..., 0, 0]
        right_next_sensor = batch_right.get(("next", "agents", "observation", "sensors"))[..., 0, 0]
        right_delta = float((right_next_sensor - right_sensor).mean().item())

        assert left_direction_mean > 0.4
        assert right_direction_mean < 0.1
        assert left_delta > 0.0
        assert right_delta < left_delta
