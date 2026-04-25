from data_harvesting.environment import MetricKind, MetricReducer, make_metrics_spec


def test_data_collection_metric_spec_matches_current_environment_contract() -> None:
    metrics_spec = make_metrics_spec()

    assert metrics_spec.info_keys == (
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

    cause_metric = metrics_spec.by_key("cause")
    assert cause_metric.kind == MetricKind.CATEGORICAL
    assert cause_metric.reducer == MetricReducer.COUNT
    assert cause_metric.logging_prefix == "end_cause"
    assert cause_metric.value_labels == {
        0: "NONE",
        1: "TIMEOUT",
        2: "ALL_COLLECTED",
        3: "STALLED",
        4: "ALL_AGENTS_INACTIVE",
    }
