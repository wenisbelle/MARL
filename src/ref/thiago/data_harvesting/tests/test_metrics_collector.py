import pytest
import torch
from tensordict import TensorDict

from data_harvesting.environment.data_collection.metrics import make_data_collection_metrics_spec
from data_harvesting.metrics import EnvironmentMetricsCollector


METRIC_KEYS = (
    "avg_reward",
    "max_reward",
    "sum_reward",
    "avg_collection_time",
    "episode_duration",
    "completion_time",
    "all_collected",
    "num_collected",
    "cause",
)


def _make_batch(done: list[bool], truncated: list[bool], info_rows: list[dict[str, float]]) -> TensorDict:
    assert len(done) == len(truncated) == len(info_rows)
    length = len(done)

    info_tensors = {
        key: torch.tensor([[row[key]] for row in info_rows], dtype=torch.float32)
        for key in METRIC_KEYS
    }

    return TensorDict(
        {
            "next": TensorDict(
                {
                    "done": torch.tensor(done, dtype=torch.bool).view(length, 1),
                    "truncated": torch.tensor(truncated, dtype=torch.bool).view(length, 1),
                    "agents": TensorDict(
                        {
                            "info": TensorDict(info_tensors, batch_size=[length, 1]),
                        },
                        batch_size=[length],
                    ),
                },
                batch_size=[length],
            )
        },
        batch_size=[length],
    )


def test_environment_metrics_collector_accumulates_multiple_terminal_steps() -> None:
    metrics_spec = make_data_collection_metrics_spec()
    collector = EnvironmentMetricsCollector(torch.device("cpu"), metrics_spec)
    batch = _make_batch(
        done=[False, True, False, True],
        truncated=[False, False, False, False],
        info_rows=[
            {
                "avg_reward": 0.0,
                "max_reward": 0.0,
                "sum_reward": 0.0,
                "avg_collection_time": 0.0,
                "episode_duration": 0.0,
                "completion_time": 0.0,
                "all_collected": 0.0,
                "num_collected": 0.0,
                "cause": 0.0,
            },
            {
                "avg_reward": 2.0,
                "max_reward": 3.0,
                "sum_reward": 4.0,
                "avg_collection_time": 5.0,
                "episode_duration": 6.0,
                "completion_time": 7.0,
                "all_collected": 1.0,
                "num_collected": 8.0,
                "cause": 2.0,
            },
            {
                "avg_reward": 0.0,
                "max_reward": 0.0,
                "sum_reward": 0.0,
                "avg_collection_time": 0.0,
                "episode_duration": 0.0,
                "completion_time": 0.0,
                "all_collected": 0.0,
                "num_collected": 0.0,
                "cause": 0.0,
            },
            {
                "avg_reward": 10.0,
                "max_reward": 11.0,
                "sum_reward": 12.0,
                "avg_collection_time": 13.0,
                "episode_duration": 14.0,
                "completion_time": 15.0,
                "all_collected": 0.0,
                "num_collected": 16.0,
                "cause": 1.0,
            },
        ],
    )

    collector.report_metrics(batch)

    assert collector.trajectories.item() == pytest.approx(2.0)
    assert collector.scalar_totals["avg_reward"].item() == pytest.approx(12.0)
    assert collector.scalar_totals["completion_time"].item() == pytest.approx(22.0)
    assert collector.scalar_totals["all_collected"].item() == pytest.approx(1.0)
    assert collector.scalar_totals["num_collected"].item() == pytest.approx(24.0)
    assert collector.categorical_counts["cause"]["ALL_COLLECTED"].item() == pytest.approx(1.0)
    assert collector.categorical_counts["cause"]["TIMEOUT"].item() == pytest.approx(1.0)


def test_environment_metrics_collector_ignores_non_terminal_batches() -> None:
    collector = EnvironmentMetricsCollector(torch.device("cpu"), make_data_collection_metrics_spec())
    batch = _make_batch(
        done=[False, False, False],
        truncated=[False, False, False],
        info_rows=[
            {
                "avg_reward": 1.0,
                "max_reward": 1.0,
                "sum_reward": 1.0,
                "avg_collection_time": 1.0,
                "episode_duration": 1.0,
                "completion_time": 1.0,
                "all_collected": 1.0,
                "num_collected": 1.0,
                "cause": 2.0,
            }
            for _ in range(3)
        ],
    )

    collector.report_metrics(batch)

    assert collector.trajectories.item() == pytest.approx(0.0)
    assert collector.scalar_totals["sum_reward"].item() == pytest.approx(0.0)
    assert sum(count.item() for count in collector.categorical_counts["cause"].values()) == pytest.approx(0.0)


def test_environment_metrics_collector_logs_average_metrics(monkeypatch) -> None:
    collector = EnvironmentMetricsCollector(torch.device("cpu"), make_data_collection_metrics_spec())
    batch = _make_batch(
        done=[True, True],
        truncated=[False, False],
        info_rows=[
            {
                "avg_reward": 2.0,
                "max_reward": 4.0,
                "sum_reward": 6.0,
                "avg_collection_time": 8.0,
                "episode_duration": 10.0,
                "completion_time": 12.0,
                "all_collected": 1.0,
                "num_collected": 2.0,
                "cause": 2.0,
            },
            {
                "avg_reward": 6.0,
                "max_reward": 8.0,
                "sum_reward": 10.0,
                "avg_collection_time": 12.0,
                "episode_duration": 14.0,
                "completion_time": 16.0,
                "all_collected": 0.0,
                "num_collected": 1.0,
                "cause": 3.0,
            },
        ],
    )
    collector.report_metrics(batch)

    logged: list[tuple[dict[str, float], int]] = []

    def _capture(metrics: dict[str, float], step: int):
        logged.append((metrics, step))

    monkeypatch.setattr("data_harvesting.metrics.mlflow.log_metrics", _capture)
    collector.log_metrics(step=123)

    assert len(logged) == 1
    metrics, step = logged[0]
    assert step == 123
    assert metrics["avg_reward"] == pytest.approx(4.0)
    assert metrics["completion_time"] == pytest.approx(14.0)
    assert metrics["all_collected"] == pytest.approx(0.5)
    assert metrics["num_collected"] == pytest.approx(1.5)
    assert metrics["end_cause_ALL_COLLECTED"] == pytest.approx(1.0)
    assert metrics["end_cause_STALLED"] == pytest.approx(1.0)
