from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .environment import EndCause


class MetricKind(str, Enum):
    SCALAR = "scalar"
    CATEGORICAL = "categorical"


class MetricReducer(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"


@dataclass(frozen=True, slots=True)
class EnvironmentMetricSpec:
    key: str
    kind: MetricKind
    reducer: MetricReducer
    expanded_key_prefix: str | None = None
    """
    If set, this prefix will be used when logging the metric to MLflow. For example, if key="cause" and 
    expanded_key_prefix="end_cause", then the metric will be logged as "end_cause_cause". This is useful 
    for categorical metrics, where we want to log each category as a separate metric."""
    value_labels: dict[int, str] | None = None
    """
    If set, this dictionary will be used to map metric values to human-readable labels when logging to MLflow. 
    For example, if key="cause" and value_labels={0: "success", 1: "failure"}, then a metric value of 0 will be 
    logged as "success" and a metric value of 1 will be logged as "failure".
    """

    @property
    def logging_prefix(self) -> str:
        return self.expanded_key_prefix or self.key


@dataclass(frozen=True, slots=True)
class EnvironmentMetricsSpec:
    metrics: tuple[EnvironmentMetricSpec, ...]

    @property
    def info_keys(self) -> tuple[str, ...]:
        return tuple(metric.key for metric in self.metrics)

    @property
    def scalar_metrics(self) -> tuple[EnvironmentMetricSpec, ...]:
        return tuple(metric for metric in self.metrics if metric.kind == MetricKind.SCALAR)

    @property
    def categorical_metrics(self) -> tuple[EnvironmentMetricSpec, ...]:
        return tuple(metric for metric in self.metrics if metric.kind == MetricKind.CATEGORICAL)

    def by_key(self, key: str) -> EnvironmentMetricSpec:
        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(f"Unknown metric key: {key}")



def make_data_collection_metrics_spec() -> EnvironmentMetricsSpec:
    return EnvironmentMetricsSpec(
        metrics=(
            EnvironmentMetricSpec("avg_reward", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec("max_reward", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec("sum_reward", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec("avg_collection_time", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec("episode_duration", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec("completion_time", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec("all_collected", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec("num_collected", MetricKind.SCALAR, MetricReducer.MEAN),
            EnvironmentMetricSpec(
                "cause",
                MetricKind.CATEGORICAL,
                MetricReducer.COUNT,
                expanded_key_prefix="end_cause",
                value_labels={cause.value: cause.name for cause in EndCause},
            ),
        )
    )
