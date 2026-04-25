import types

from torch import nn

from data_harvesting.eval import NestedEvaluationRunLogger
from data_harvesting.train import _maybe_run_periodic_evaluation, _run_nested_evaluation


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.policy = _FakePolicy()


class _FakeLogger:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict]] = []

    def log_evaluation(self, step: int, results: dict) -> None:
        self.logged.append((step, results))


def _evaluation_config() -> dict:
    return {
        "evaluation": {
            "enabled": True,
            "eval_every_n_steps": 100,
            "num_runs": 4,
            "seed": 7,
        }
    }


def test_run_nested_evaluation_uses_eval_mode_and_restores_training_state(monkeypatch) -> None:
    algorithm = _FakeAlgorithm()
    algorithm.policy.train(True)
    logger = _FakeLogger()

    def _fake_eval(policy, config, num_runs, seed):
        assert policy.training is False
        assert num_runs == 4
        assert seed == 7
        return {"num_runs": 4, "metrics": {}, "scenario_metrics": {}}

    monkeypatch.setattr("data_harvesting.train.run_eval", _fake_eval)

    _run_nested_evaluation(
        algorithm,
        _evaluation_config(),
        experience_steps=120,
        logger=logger,
        num_runs=4,
        seed=7,
    )

    assert algorithm.policy.training is True
    assert logger.logged == [(120, {"num_runs": 4, "metrics": {}, "scenario_metrics": {}})]


def test_maybe_run_periodic_evaluation_respects_interval(monkeypatch) -> None:
    algorithm = _FakeAlgorithm()
    logger = _FakeLogger()
    calls: list[int] = []

    def _fake_run_nested_evaluation(algorithm, config, *, experience_steps, logger, num_runs, seed):
        calls.append(experience_steps)

    monkeypatch.setattr("data_harvesting.train._run_nested_evaluation", _fake_run_nested_evaluation)

    last_eval_step = _maybe_run_periodic_evaluation(
        algorithm,
        _evaluation_config(),
        experience_steps=50,
        last_eval_step=0,
        logger=logger,
    )
    assert last_eval_step == 0
    assert calls == []

    last_eval_step = _maybe_run_periodic_evaluation(
        algorithm,
        _evaluation_config(),
        experience_steps=100,
        last_eval_step=0,
        logger=logger,
    )
    assert last_eval_step == 100
    assert calls == [100]


def test_nested_evaluation_run_logger_reuses_child_runs(monkeypatch) -> None:
    created_runs: list[tuple[str, dict[str, str]]] = []
    logged_metrics: list[tuple[str, str, float, int]] = []
    terminated: list[str] = []
    next_id = {"value": 0}

    class _FakeClient:
        def create_run(self, experiment_id, start_time=None, tags=None, run_name=None):
            run_id = f"run-{next_id['value']}"
            next_id["value"] += 1
            created_runs.append((run_name, tags or {}))
            return types.SimpleNamespace(
                info=types.SimpleNamespace(run_id=run_id),
            )

        def log_metric(self, run_id, key, value, timestamp=None, step=None, synchronous=None, dataset_name=None, dataset_digest=None, model_id=None):
            logged_metrics.append((run_id, key, value, step))
            return None

        def set_terminated(self, run_id, status=None, end_time=None):
            terminated.append(run_id)

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", lambda: _FakeClient())

    parent_run = types.SimpleNamespace(
        info=types.SimpleNamespace(experiment_id="exp-1", run_id="parent-1"),
    )
    logger = NestedEvaluationRunLogger(parent_run)
    results = {
        "num_runs": 3,
        "metrics": {
            "all_collected": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0},
        },
        "end_cause_counts": {"ALL_COLLECTED": 1, "STALLED": 2},
        "end_cause_rate": {"ALL_COLLECTED": 1 / 3, "STALLED": 2 / 3},
        "scenario_metrics": {
            "agents_3__sensors_2": {
                "scenario": {"agents": 3, "sensors": 2},
                "num_runs": 2,
                "metrics": {
                    "all_collected": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                },
                "end_cause_counts": {"ALL_COLLECTED": 2, "STALLED": 0},
                "end_cause_rate": {"ALL_COLLECTED": 1.0, "STALLED": 0.0},
            }
        },
    }

    logger.log_evaluation(100, results)
    logger.log_evaluation(200, results)
    logger.close()

    assert [name for name, _ in created_runs] == [
        "evaluation_overall",
        "evaluation_agents_3__sensors_2",
    ]
    assert any(run_id == "run-0" and key == "all_collected_mean" and step == 100 for run_id, key, _, step in logged_metrics)
    assert any(run_id == "run-0" and key == "all_collected_mean" and step == 200 for run_id, key, _, step in logged_metrics)
    assert any(run_id == "run-1" and key == "all_collected_mean" and step == 100 for run_id, key, _, step in logged_metrics)
    assert terminated == ["run-0", "run-1"]
