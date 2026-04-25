import argparse
import time
from typing import Dict

import torch
from torchrl.envs import check_env_specs, TransformedEnv, RewardSum
import mlflow

from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
from data_harvesting.collector import create_collector
from data_harvesting.environment import make_env
from train import train


def _build_env(config: Dict, check: bool = False) -> TransformedEnv:
    base_env = make_env(config)
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map.keys()),
        ),
    )
    if check:
        check_env_specs(env)
    return env


def _make_algorithm(config: Dict, device: torch.device, env: TransformedEnv):
    algo_name = config["training"]["algorithm"].lower()
    if algo_name == "mappo":
        return MAPPOAlgorithm(env, device, config)
    return MADDPGAlgorithm(env, device, config)


def _override_config(
    config: Dict,
    num_iters: int,
    warmup_iters: int,
    frames_per_batch: int | None,
) -> Dict:
    updated = {**config}
    updated_training = {**config["training"]}
    updated_collector = {**config["collector"]}

    if frames_per_batch is not None:
        updated_collector["frames_per_batch"] = frames_per_batch

    total_frames = updated_collector["frames_per_batch"] * (num_iters + warmup_iters)
    updated_training["total_timesteps"] = total_frames

    updated["training"] = updated_training
    updated["collector"] = updated_collector
    return updated


def profile_training(
    config: Dict,
    num_iters: int,
    frames_per_batch: int | None,
    warmup_iters: int,
    enable_flash_sdp: bool | None,
    enable_mem_efficient_sdp: bool | None,
    enable_math_sdp: bool | None,
    use_train_loop: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _override_config(config, num_iters, warmup_iters, frames_per_batch)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("profile_train")

    if torch.cuda.is_available():
        if enable_flash_sdp is not None:
            torch.backends.cuda.enable_flash_sdp(enable_flash_sdp)
        if enable_mem_efficient_sdp is not None:
            torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient_sdp)
        if enable_math_sdp is not None:
            torch.backends.cuda.enable_math_sdp(enable_math_sdp)
        print(
            "SDP kernels: "
            f"flash={torch.backends.cuda.flash_sdp_enabled()} "
            f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
            f"math={torch.backends.cuda.math_sdp_enabled()}"
        )

    sample_env = _build_env(config, check=True)
    algorithm = _make_algorithm(config, device, sample_env)

    collection_device = torch.device(config["collector"]["device"])

    schedule = None
    if use_train_loop and warmup_iters > 0:
        schedule = torch.profiler.schedule(wait=warmup_iters, warmup=0, active=num_iters, repeat=1)

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        schedule=schedule,
    )

    frames_seen = 0
    start_time = time.perf_counter()

    if use_train_loop:
        with prof:
            train(config, run_name="profile_train", profiler=prof)
        frames_seen = config["training"]["total_timesteps"]
    else:
        with create_collector(algorithm.exploratory_policy, collection_device, lambda: _build_env(config), config) as collector:
            iterator = enumerate(collector)

            # Warmup iterations to pay compile + cache cost outside the profiler window.
            for iteration, batch in iterator:
                if iteration >= warmup_iters:
                    break
                batch = batch.reshape(-1)
                algorithm.learn(batch)

            with prof:
                for iteration, batch in iterator:
                    if iteration >= warmup_iters + num_iters:
                        break
                    current_frames = batch.numel()
                    batch = batch.reshape(-1)

                    with torch.profiler.record_function("learn_step"):
                        algorithm.learn(batch)

                    frames_seen += current_frames
                    prof.step()

    elapsed = time.perf_counter() - start_time
    fps = frames_seen / max(elapsed, 1e-9)

    print(f"Profiled {frames_seen} frames in {elapsed:.2f}s => {fps:.1f} frames/s")
    prof.export_chrome_trace("profile_trace.json")


def main():
    parser = argparse.ArgumentParser(description="Profile a short RL training run.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--iters", type=int, default=3, help="Number of collection iterations to profile")
    parser.add_argument("--warmup-iters", type=int, default=2, help="Number of warmup iterations before profiling")
    parser.add_argument(
        "--flash-sdp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable flash scaled dot-product attention",
    )
    parser.add_argument(
        "--mem-efficient-sdp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable mem-efficient scaled dot-product attention",
    )
    parser.add_argument(
        "--math-sdp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable math scaled dot-product attention",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=None,
        help="Override frames_per_batch for profiling",
    )
    parser.add_argument(
        "--use-train-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Profile the real training loop in train.py",
    )
    args = parser.parse_args()

    import yaml

    with open(args.params, "r") as f:
        config: Dict = yaml.safe_load(f)

    profile_training(
        config,
        num_iters=args.iters,
        frames_per_batch=args.frames_per_batch,
        warmup_iters=args.warmup_iters,
        enable_flash_sdp=args.flash_sdp,
        enable_mem_efficient_sdp=args.mem_efficient_sdp,
        enable_math_sdp=args.math_sdp,
        use_train_loop=args.use_train_loop,
    )


if __name__ == "__main__":
    main()