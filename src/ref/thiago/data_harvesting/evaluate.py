from __future__ import annotations

import argparse

import mlflow
import yaml

from data_harvesting.eval import eval as run_eval
from data_harvesting.eval import load_policy_from_mlflow_run


def _print_summary(results: dict) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"Runs: {results['num_runs']}")

    print("\nMetrics:")
    for metric_name, values in results["metrics"].items():
        print(
            f"- {metric_name}: "
            f"mean={values['mean']:.4f}, "
            f"std={values['std']:.4f}, "
            f"min={values['min']:.4f}, "
            f"max={values['max']:.4f}"
        )

    print("\nEnd causes:")
    counts = results["end_cause_counts"]
    rates = results["end_cause_rate"]
    for cause_name in counts:
        print(f"- {cause_name}: count={counts[cause_name]}, rate={rates[cause_name]:.2%}")

    scenario_metrics = results.get("scenario_metrics", {})
    if scenario_metrics:
        print("\nScenario breakdown:")
        for scenario_key, scenario_results in sorted(scenario_metrics.items()):
            scenario = scenario_results["scenario"]
            print(
                f"\n[{scenario_key}] "
                f"agents={scenario['agents']}, sensors={scenario['sensors']}, runs={scenario_results['num_runs']}"
            )
            for metric_name, values in scenario_results["metrics"].items():
                print(
                    f"- {metric_name}: "
                    f"mean={values['mean']:.4f}, "
                    f"std={values['std']:.4f}, "
                    f"min={values['min']:.4f}, "
                    f"max={values['max']:.4f}"
                )
            counts = scenario_results["end_cause_counts"]
            rates = scenario_results["end_cause_rate"]
            for cause_name in counts:
                print(
                    f"- end_cause[{cause_name}]: "
                    f"count={counts[cause_name]}, rate={rates[cause_name]:.2%}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved MLflow model run.")
    parser.add_argument("--run-id", "-R", required=True, help="MLflow run ID to evaluate")
    parser.add_argument(
        "--num-runs",
        "-N",
        required=True,
        type=int,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Run environment in visual mode during evaluation",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to YAML params file used to build the environment",
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--model-name",
        default="policy_model",
        help="Preferred logged model name for the given run",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    with open(args.params, "r") as handle:
        config: dict = yaml.safe_load(handle)

    policy, model_id = load_policy_from_mlflow_run(
        args.run_id,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
    )

    print(f"Evaluating run_id={args.run_id}")
    print(f"Loaded model_id={model_id}")
    print(f"Visual mode={'on' if args.visual else 'off'}")

    results = run_eval(policy, config, args.num_runs, visual=args.visual)
    _print_summary(results)


if __name__ == "__main__":
    main()
