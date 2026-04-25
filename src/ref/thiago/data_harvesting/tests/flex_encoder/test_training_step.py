import torch

from data_harvesting.algorithm import MADDPGAlgorithm
from data_harvesting.collector import create_collector
from data_harvesting.environment.data_collection import make_data_collection_env


def _flex_maddpg_config() -> dict:
    return {
        "environment": {
            "sequential_obs": True,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 3,
            "max_num_agents": 3,
            "min_num_sensors": 3,
            "max_num_sensors": 3,
            "scenario_size": 20.0,
            "max_episode_length": 50,
            "max_seconds_stalled": 20,
            "communication_range": 0.0,
            "state_num_closest_sensors": 3,
            "state_num_closest_drones": 2,
            "id_on_state": True,
            "min_sensor_priority": 1.0,
            "max_sensor_priority": 1.0,
            "full_random_drone_position": False,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
        },
        "actor": {
            "share_parameters": True,
            "network_depth": 1,
            "network_width": 32,
            "activation_function": "Tanh",
            "centralized": False,
        },
        "critic": {
            "network_depth": 1,
            "network_width": 32,
            "activation_function": "Tanh",
            "share_parameters": True,
            "centralized": True,
        },
        "training": {
            "total_timesteps": 64,
            "batch_size": 8,
            "algorithm": "maddpg",
            "exploration_sigma_init": 0.2,
            "exploration_sigma_end": 0.1,
            "exploration_annealing_steps": 100,
        },
        "collector": {
            "num_collectors": 1,
            "frames_per_batch": 16,
            "async_collector": False,
            "device": "cpu",
        },
        "replay_buffer": {
            "buffer_size": 256,
            "prefetch": 0,
            "buffer_device": "cpu",
        },
        "optimization": {
            "gamma": 0.99,
            "lr": 1e-3,
            "tau": 0.5,
            "num_optimizer_steps": 1,
            "grad_clip": 0,
            "use_amp": False,
        },
        "flex_encoder": {
            "enabled": True,
            "sequential_heads": {
                "embed_dim": 16,
                "head_dim": 8,
                "num_heads": 2,
                "ff_dim": 32,
                "depth": 1,
                "dropout": 0.0,
                "critic_agent_embedding": True,
            },
            "flat_heads": {
                "embed_dim": 8,
                "depth": 1,
                "num_cells": 16,
                "activation_function": "Tanh",
            },
            "mix_layer_depth": 1,
            "mix_layer_num_cells": 32,
            "mix_activation_function": "Tanh",
        },
    }


def _collect_training_batch(algorithm: MADDPGAlgorithm, config: dict) -> torch.Tensor:
    with create_collector(
        algorithm.exploratory_policy,
        torch.device("cpu"),
        lambda: make_data_collection_env(config),
        config,
    ) as collector:
        batch = next(iter(collector))
    return batch.reshape(-1)


def _clone_params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in module.parameters()]


def _clone_td_params(td_params) -> list[torch.Tensor]:
    return [value.detach().clone() for value in td_params.flatten_keys().values()]


def _any_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return any(not torch.allclose(prev, curr) for prev, curr in zip(before, after, strict=True))


def test_flex_maddpg_single_learn_step_updates_models_and_targets() -> None:
    config = _flex_maddpg_config()
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        batch = _collect_training_batch(algorithm, config)

        actor_before = _clone_params(algorithm.policy)
        critic_before = _clone_params(algorithm.critic)
        target_actor_before = _clone_td_params(algorithm.loss_module.target_actor_network_params)
        target_value_before = _clone_td_params(algorithm.loss_module.target_value_network_params)

        losses = algorithm.learn(batch)

        assert torch.isfinite(losses["loss_actor"])
        assert torch.isfinite(losses["loss_value"])
        assert len(algorithm.replay_buffer) > 0

        actor_after = _clone_params(algorithm.policy)
        critic_after = _clone_params(algorithm.critic)
        target_actor_after = _clone_td_params(algorithm.loss_module.target_actor_network_params)
        target_value_after = _clone_td_params(algorithm.loss_module.target_value_network_params)

        assert _any_changed(actor_before, actor_after)
        assert _any_changed(critic_before, critic_after)
        assert _any_changed(target_actor_before, target_actor_after)
        assert _any_changed(target_value_before, target_value_after)
    finally:
        env.close()
