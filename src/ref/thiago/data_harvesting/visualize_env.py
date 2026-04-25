from __future__ import annotations

import argparse
from copy import deepcopy

import yaml

from data_harvesting.environment import EndCause, make_env


def _load_config() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as handle:
        config: dict = yaml.safe_load(handle)
    config = deepcopy(config)
    config["environment"]["render_mode"] = "visual"
    return config


def _resolve_end_cause(final_next_td) -> EndCause:
    cause_value = int(float(final_next_td.get(("agents", "info", "cause"))[0].item()))
    try:
        return EndCause(cause_value)
    except ValueError:
        return EndCause.NONE


def run_visualization(episodes: int, seed: int | None) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be greater than 0")

    config = _load_config()
    env = make_env(config)

    try:
        for episode_index in range(episodes):
            episode_seed = None if seed is None else seed + episode_index
            td = env.reset(seed=episode_seed)

            steps = 0
            final_next_td = td
            action_spec = env.full_action_spec["agents", "action"]
            while True:
                td.set(("agents", "action"), action_spec.rand())
                final_next_td = env.step(td).get("next")
                steps += 1
                if bool(final_next_td.get("done").item()):
                    break
                td = final_next_td

            end_cause = _resolve_end_cause(final_next_td)
            info = final_next_td.get(("agents", "info"))
            total_reward = float(info.get("sum_reward")[0].item())
            num_collected = int(float(info.get("num_collected")[0].item()))

            print(
                f"Episode {episode_index + 1}/{episodes}: "
                f"steps={steps}, end_cause={end_cause.name}, "
                f"num_collected={num_collected}, sum_reward={total_reward:.3f}"
            )
    finally:
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the environment with random actions.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed; episode i uses seed + i",
    )
    args = parser.parse_args()

    run_visualization(episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
