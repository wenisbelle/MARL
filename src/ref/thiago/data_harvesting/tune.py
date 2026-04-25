import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from copy import deepcopy

import mlflow
from hyperopt import fmin, hp, tpe
from time import sleep


parser = argparse.ArgumentParser()
parser.add_argument(
    "-E",
    type=str,
    required=False,
    help="MLflow experiment name (defaults to 'default')",
    dest="experiment_name",
)
parser.add_argument(
    "-S",
    type=int,
    required=False,
    default=300_000,
    help="Total training timesteps for each trial (default: 300,000)",
    dest="total_timesteps",
)
parser.add_argument(
    "-N",
    type=int,
    required=False,
    default=30,
    help="Number of hyperparameter configurations to try (default: 30)",
    dest="num_trials",
)
args = parser.parse_args()

MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)

    config["training"]["total_timesteps"] = args.total_timesteps

    # We do not want to save the model for every trial during hyperparameter tuning, as it can consume a lot of disk space 
    # and is not necessary for identifying the best hyperparameters. 
    config["metrics"]["save_model"] = False

    experiment_name = args.experiment_name if args.experiment_name else "default"
    mlflow.set_experiment(experiment_name)
    
    space = {
        "training": {
            "batch_size": hp.choice("batch_size", [256, 512, 1024]),
        },
        "optimization": {
            "num_optimizer_steps": hp.choice("num_optimizer_steps", [1, 2, 5, 10, 20, 40]),
            "lr": hp.loguniform("lr", math.log(1e-5), math.log(3e-3)),
            "tau": hp.loguniform("tau", math.log(1e-3), math.log(5e-2)),
            "gamma": hp.uniform("gamma", 0.95, 0.999),
            "grad_clip": hp.choice("grad_clip", [0.0, 0.5, 1.0, 5.0, 10.0]),
        },
        "collector": {
            "frames_per_batch": hp.choice("frames_per_batch", [512, 1024, 2048, 4096]),
        },
        "flex_encoder": {
            "sequential_heads": {
                "embed_dim": hp.choice("seq_embed_dim", [64, 128, 256]),
                "num_heads": hp.choice("seq_num_heads", [2, 4, 8]),
                "ff_dim": hp.choice("seq_ff_dim", [128, 256]),
                "depth": hp.choice("seq_depth", [1, 2, 3]),
            },
            "flat_heads": {
                "embed_dim": hp.choice("flat_embed_dim", [64, 128, 256]),
                "depth": hp.choice("flat_depth", [1, 2]),
                "num_cells": hp.choice("flat_num_cells", [64, 128, 256]),
            },
            "mix_layer_num_cells": hp.choice("mix_layer_num_cells", [128, 256]),
            "mix_layer_depth": hp.choice("mix_layer_depth", [1, 2, 3]),
        },
    }

    def _deep_update(target: dict, updates: dict) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _deep_update(target[key], value)
            else:
                target[key] = value

    def _collect_leafs(prefix: str, value: object, out: list[tuple[str, object]]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                _collect_leafs(next_prefix, child, out)
        else:
            out.append((prefix, value))

    def _train_in_subprocess(run_config: dict, run_name: str | None) -> float:
        """Runs train() in a fresh Python process and returns its numeric result.

        This avoids memory accumulation in the parent process across many trials.
        """

        fd, result_path = tempfile.mkstemp(prefix="tune_result_", suffix=".json")
        os.close(fd)

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tune_worker",
                    "--tracking-uri",
                    MLFLOW_TRACKING_URI,
                    "--experiment-name",
                    experiment_name,
                    "--run-name",
                    run_name or "",
                    "--result-path",
                    result_path,
                ],
                input=json.dumps(run_config),
                text=True,
                # Inherit stdout/stderr so training progress is visible live.
                stdout=None,
                stderr=None,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            if not os.path.exists(result_path):
                raise RuntimeError(
                    "Training subprocess did not write a result file. "
                    "(Was it killed, or did it crash before finishing?)"
                )

            payload = json.loads(open(result_path, "r", encoding="utf-8").read())

            print(f"Trial result: {payload['result']}, waiting 40s for subprocess to release resources...")
            sleep(40) # give the subprocess some time to release resources before starting the next trial

            return float(payload["result"])
        finally:
            try:
                os.remove(result_path)
            except OSError:
                pass

    def tune(tune_args: dict):
        run_config = deepcopy(config)

        tuned_arg_leafs: list[tuple[str, object]] = []

        _deep_update(run_config, tune_args)
        _collect_leafs("", tune_args, tuned_arg_leafs)

        run_name = " ".join([f"{k}={v}" for k, v in tuned_arg_leafs])

        return -_train_in_subprocess(run_config, run_name=run_name)

    best = fmin(
        fn=tune,
        space=space,
        algo=tpe.suggest,
        max_evals=args.num_trials,
        show_progressbar=False
    )
    print("Best hyperparameters found:")
    print(best)