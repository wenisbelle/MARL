import time

import mlflow
import torch
from tensordict import TensorDictBase

from data_harvesting.environment import EnvironmentMetricsSpec, MetricKind, MetricReducer


class EnvironmentMetricsCollector:
    def __init__(self, device: torch.device, metrics_spec: EnvironmentMetricsSpec):
        self._device = device
        self._metrics_spec = metrics_spec
        self.trajectories: torch.Tensor = torch.zeros((), device=device)
        self.scalar_totals: dict[str, torch.Tensor] = {
            metric.key: torch.zeros((), device=device)
            for metric in metrics_spec.scalar_metrics
        }
        self.categorical_counts: dict[str, dict[str, torch.Tensor]] = {
            metric.key: {
                label: torch.zeros((), device=device)
                for label in (metric.value_labels or {}).values()
            }
            for metric in metrics_spec.categorical_metrics
        }

    def report_metrics(self, batch: TensorDictBase) -> None:
        """
        Report metrics from a batch of transitions. This should be called for every batch of transitions collected, and
        it will internally identify terminal transitions to update the metrics accordingly.
        :param batch: A batch of transitions, expected to contain the keys ("next", "done") for identifying terminal
        transitions, and ("next", "agents", "info") for extracting metric values from the environment info dictionary.
        """

        # Use environment-level termination to identify terminal transitions.
        # Agent-level done includes truncated inactive slots, which can be true
        # even when an episode has not ended.
        done = batch.get(("next", "done")).reshape(-1).to(torch.bool)
        if not bool(done.any()):
            return

        info = batch.get(("next", "agents", "info"))[done, 0]
        det_info = info.detach()
        self.trajectories += done.sum()

        for metric in self._metrics_spec.scalar_metrics:
            self.scalar_totals[metric.key] += det_info[metric.key].sum()

        for metric in self._metrics_spec.categorical_metrics:
            values = det_info[metric.key]
            for raw_value, label in (metric.value_labels or {}).items():
                self.categorical_counts[metric.key][label] += (values == float(raw_value)).sum()

    def _build_log_metrics(self) -> dict[str, float]:
        trajectories = self.trajectories.item()
        if trajectories == 0:
            return {}

        metrics: dict[str, float] = {}
        for metric in self._metrics_spec.scalar_metrics:
            total = self.scalar_totals[metric.key]
            if metric.reducer == MetricReducer.MEAN:
                metrics[metric.key] = float((total / self.trajectories).item())
            elif metric.reducer == MetricReducer.SUM:
                metrics[metric.key] = float(total.item())

        for metric in self._metrics_spec.categorical_metrics:
            if metric.reducer != MetricReducer.COUNT:
                raise NotImplemented(f"Unsupported reducer '{metric.reducer}' for categorical metric '{metric.key}'")
            prefix = metric.logging_prefix
            for label, count in self.categorical_counts[metric.key].items():
                metrics[f"{prefix}_{label}"] = float(count.item())
        return metrics

    def log_metrics(self, step: int) -> None:
        """
        Log accumulated metrics to MLflow and reset the internal counters. This should be called periodically
         (e.g. every N steps) to log the metrics.
        :param step: The current training step to log the metrics under.
        """
        metrics = self._build_log_metrics()
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def metric_value(self, key: str) -> float:
        """
        Retrieve the current value of a scalar metric by key.
        :param key: The key of the metric to retrieve. Must be a scalar metric defined in the metrics spec.
        :return:
        """
        total = self.scalar_totals[key]
        metric = self._metrics_spec.by_key(key)
        if self.trajectories.item() == 0:
            return 0.0
        if metric.reducer == MetricReducer.MEAN:
            return float((total / self.trajectories).item())
        if metric.reducer == MetricReducer.SUM:
            return float(total.item())
        raise ValueError(f"Metric '{key}' is not a scalar aggregate")


class LearningMetricsCollector:
    def __init__(self, device: torch.device):
        self._device = device
        self.losses: dict[str, torch.Tensor] = {}
        self.iterations: torch.Tensor = torch.zeros((), device=device)
        self.start_time: float | None = None

    def report_loss(self, loss_name: str, loss_value: torch.Tensor):
        if loss_name not in self.losses:
            self.losses[loss_name] = torch.zeros((), device=self._device)
        self.losses[loss_name] += loss_value.detach()

        self.start_time = time.time()

        self.iterations += 1

    def log_metrics(self, step: int):
        # Batch all learning metrics in a single call
        metrics: dict[str, float] = {}
        iterations = self.iterations.item()
        if iterations == 0:
            return

        for loss_name, loss_value in self.losses.items():
            avg_loss = (loss_value / self.iterations).item()
            metrics[f"loss_{loss_name}"] = avg_loss

        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            sps = 1 / elapsed_time if elapsed_time > 0 else 0
            metrics["sps"] = sps

        if metrics:
            mlflow.log_metrics(metrics, step=step)
        self.losses.clear()
        self.iterations.zero_()
