from __future__ import annotations

from copy import deepcopy
from typing import Any

import mlflow
import torch
from mlflow import pytorch as mlflow_pytorch
from mlflow import MlflowClient
from torchrl.envs.utils import ExplorationType, set_exploration_type

from data_harvesting.environment import MetricKind, make_env, make_metrics_spec


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    data = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
    }


def _scenario_key(num_agents: int, num_sensors: int) -> str:
    return f"agents_{num_agents}__sensors_{num_sensors}"


def _empty_categorical_counts(metrics_spec) -> dict[str, dict[str, int]]:
    return {
        metric.logging_prefix: {
            label: 0 for label in (metric.value_labels or {}).values()
        }
        for metric in metrics_spec.categorical_metrics
    }


def _empty_scenario_bucket(metrics_spec, *, num_agents: int, num_sensors: int) -> dict[str, Any]:
    return {
        "scenario": {"agents": num_agents, "sensors": num_sensors},
        "num_runs": 0,
        "scalar_samples": {
            metric.key: []
            for metric in metrics_spec.scalar_metrics
        },
        "categorical_counts": _empty_categorical_counts(metrics_spec),
    }


def _get_episode_scenario(episode_info) -> tuple[int, int]:
    try:
        num_agents = int(float(episode_info["num_agents"]))
        num_sensors = int(float(episode_info["num_sensors"]))
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(
            f"Evaluation requires '{missing_key}' in terminal agents.info to group scenario metrics."
        ) from exc
    return num_agents, num_sensors


def _finalize_scenario_metrics(scenarios: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key, bucket in scenarios.items():
        num_runs = bucket["num_runs"]
        scenario_result = {
            "scenario": bucket["scenario"],
            "num_runs": num_runs,
            "metrics": {
                metric_name: _metric_stats(values)
                for metric_name, values in bucket["scalar_samples"].items()
            },
        }
        for prefix, counts in bucket["categorical_counts"].items():
            scenario_result[f"{prefix}_counts"] = counts
            scenario_result[f"{prefix}_rate"] = {
                label: (count / num_runs if num_runs else 0.0)
                for label, count in counts.items()
            }
        finalized[key] = scenario_result
    return finalized


def _resolve_model_id_from_run(
    run_id: str,
    *,
    model_name: str = "policy_model",
) -> str:
    client = MlflowClient()
    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id

    models = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )

    if not models:
        raise ValueError(f"No logged model was found for run '{run_id}'.")

    preferred = [model for model in models if model.name == model_name]
    candidates = preferred if preferred else models
    candidates.sort(key=lambda item: item.creation_timestamp or 0, reverse=True)
    return candidates[0].model_id


def load_policy_from_mlflow_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
    model_name: str = "policy_model",
):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_id = _resolve_model_id_from_run(run_id, model_name=model_name)
    model_uri = f"models:/{model_id}"
    policy = mlflow_pytorch.load_model(model_uri)
    return policy, model_id


def eval(
    policy,
    config: dict[str, Any],
    num_runs: int,
    *,
    visual: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    if num_runs <= 0:
        raise ValueError("num_runs must be greater than 0")

    eval_config = deepcopy(config)
    env_config = eval_config.setdefault("environment", {})
    env_config["render_mode"] = "visual" if visual else None

    env = make_env(eval_config)
    metrics_spec = make_metrics_spec()

    if hasattr(policy, "eval"):
        policy.eval()

    scalar_samples: dict[str, list[float]] = {
        metric.key: []
        for metric in metrics_spec.scalar_metrics
    }
    categorical_counts = _empty_categorical_counts(metrics_spec)
    scenario_buckets: dict[str, dict[str, Any]] = {}

    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for run_index in range(num_runs):
            if seed is not None:
                env.set_seed(seed + run_index)

            rollout = env.rollout(
                max_steps=eval_config["environment"]["max_episode_length"],
                policy=policy
            )
            episode_info = rollout.get(("next", "agents", "info"))[-1, 0]
            num_agents, num_sensors = _get_episode_scenario(episode_info)
            scenario_key = _scenario_key(num_agents, num_sensors)
            scenario_bucket = scenario_buckets.setdefault(
                scenario_key,
                _empty_scenario_bucket(
                    metrics_spec,
                    num_agents=num_agents,
                    num_sensors=num_sensors,
                ),
            )
            scenario_bucket["num_runs"] += 1

            for metric in metrics_spec.metrics:
                if metric.kind == MetricKind.SCALAR:
                    value = float(episode_info[metric.key])
                    scalar_samples[metric.key].append(value)
                    scenario_bucket["scalar_samples"][metric.key].append(value)
                    continue

                value = int(float(episode_info[metric.key]))
                label = (metric.value_labels or {}).get(value)
                if label is not None:
                    categorical_counts[metric.logging_prefix][label] += 1
                    scenario_bucket["categorical_counts"][metric.logging_prefix][label] += 1

    if hasattr(env, "close"):
        env.close()

    results: dict[str, Any] = {
        "num_runs": num_runs,
        "metrics": {key: _metric_stats(values) for key, values in scalar_samples.items()},
        "scenario_metrics": _finalize_scenario_metrics(scenario_buckets),
    }
    for prefix, counts in categorical_counts.items():
        results[f"{prefix}_counts"] = counts
        results[f"{prefix}_rate"] = {
            label: count / num_runs for label, count in counts.items()
        }

    return results
