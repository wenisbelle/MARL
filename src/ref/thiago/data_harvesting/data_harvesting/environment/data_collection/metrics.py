from __future__ import annotations

from data_harvesting.environment.environment import EndCause
from data_harvesting.environment.metrics import (
    EnvironmentMetricSpec,
    EnvironmentMetricsSpec,
    MetricKind,
    MetricReducer,
)


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
