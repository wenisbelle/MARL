import pytest
import torch
from tensordict.nn import TensorDictModule
from torch import nn

from data_harvesting.eval import eval as run_eval


class ConstantDirectionPolicy(nn.Module):
    def __init__(self, direction: float = 0.0, speed: float = 0.0):
        super().__init__()
        self.direction = float(direction)
        self.speed = float(speed)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        action = torch.zeros(mask.shape + (2,), dtype=torch.float32, device=mask.device)
        action[..., 0] = self.direction
        action[..., 1] = self.speed
        return action


def _make_policy() -> TensorDictModule:
    return TensorDictModule(
        module=ConstantDirectionPolicy(direction=0.0, speed=0.0),
        in_keys=[("agents", "mask")],
        out_keys=[("agents", "action")],
    )


def _eval_config() -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 1,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 10.0,
            "max_episode_length": 3,
            "max_seconds_stalled": 1,
            "communication_range": 0.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
        }
    }


def test_eval_summarizes_dynamic_scalar_and_categorical_metrics() -> None:
    results = run_eval(_make_policy(), _eval_config(), num_runs=3, seed=100)

    assert results["num_runs"] == 3
    assert "avg_reward" in results["metrics"]
    assert "completion_time" in results["metrics"]
    assert results["metrics"]["avg_reward"]["mean"] == pytest.approx(-1.0)
    assert results["metrics"]["avg_reward"]["std"] == pytest.approx(0.0)
    assert results["metrics"]["episode_duration"]["mean"] == pytest.approx(2.0)
    assert results["metrics"]["completion_time"]["mean"] == pytest.approx(3.0)
    assert results["end_cause_counts"]["STALLED"] == 3
    assert results["end_cause_rate"]["STALLED"] == pytest.approx(1.0)
    assert results["end_cause_counts"]["ALL_COLLECTED"] == 0
    assert "scenario_metrics" in results

    scenario_results = results["scenario_metrics"]["agents_1__sensors_1"]
    assert scenario_results["scenario"] == {"agents": 1, "sensors": 1}
    assert scenario_results["num_runs"] == 3
    assert scenario_results["metrics"]["avg_reward"]["mean"] == pytest.approx(-1.0)
    assert scenario_results["metrics"]["completion_time"]["mean"] == pytest.approx(3.0)
    assert scenario_results["end_cause_counts"]["STALLED"] == 3
    assert scenario_results["end_cause_rate"]["STALLED"] == pytest.approx(1.0)
