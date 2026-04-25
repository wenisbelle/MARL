import pytest
import torch

from data_harvesting.actor import create_actor
from data_harvesting.critic import create_critic
from data_harvesting.environment.data_collection import make_data_collection_env


def _base_config() -> dict:
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


def _make_config(*, actor_mode: str = "shared", critic_mode: str = "centralized") -> dict:
    config = _base_config()

    if actor_mode == "shared":
        config["actor"]["centralized"] = False
        config["actor"]["share_parameters"] = True
    elif actor_mode == "per_agent":
        config["actor"]["centralized"] = False
        config["actor"]["share_parameters"] = False
    elif actor_mode == "centralized":
        config["actor"]["centralized"] = True
        config["actor"]["share_parameters"] = False
    else:
        raise ValueError(f"Unknown actor_mode: {actor_mode}")

    if critic_mode == "shared":
        config["critic"]["centralized"] = False
        config["critic"]["share_parameters"] = True
    elif critic_mode == "per_agent":
        config["critic"]["centralized"] = False
        config["critic"]["share_parameters"] = False
    elif critic_mode == "centralized":
        config["critic"]["centralized"] = True
        config["critic"]["share_parameters"] = False
    else:
        raise ValueError(f"Unknown critic_mode: {critic_mode}")

    return config


@pytest.mark.parametrize("actor_mode", ["shared", "per_agent", "centralized"])
def test_flex_actor_outputs_shape_bounds_and_device(actor_mode: str) -> None:
    config = _make_config(actor_mode=actor_mode, critic_mode="centralized")
    env = make_data_collection_env(config)

    try:
        actor = create_actor(env, torch.device("cpu"), config)
        td = env.reset(seed=0).unsqueeze(0)
        td = actor(td)

        action = td.get(("agents", "action"))
        assert tuple(action.shape) == (1, 3, 2)
        assert action.device.type == "cpu"
        assert torch.isfinite(action).all()
        assert torch.all(action >= 0.0)
        assert torch.all(action <= 1.0)
    finally:
        env.close()


@pytest.mark.parametrize("critic_mode", ["shared", "per_agent", "centralized"])
def test_flex_critic_outputs_shape_and_finite(critic_mode: str) -> None:
    config = _make_config(actor_mode="shared", critic_mode=critic_mode)
    env = make_data_collection_env(config)

    try:
        actor = create_actor(env, torch.device("cpu"), config)
        critic = create_critic(env, torch.device("cpu"), config)

        td = env.reset(seed=0).unsqueeze(0)
        td = actor(td)
        td = critic(td)

        values = td.get(("agents", "state_action_value"))
        assert tuple(values.shape) == (1, 3, 1)
        assert values.device.type == "cpu"
        assert torch.isfinite(values).all()
    finally:
        env.close()


def test_flex_centralized_actor_outputs_match_across_agents() -> None:
    config = _make_config(actor_mode="centralized", critic_mode="centralized")
    env = make_data_collection_env(config)

    try:
        actor = create_actor(env, torch.device("cpu"), config)
        td = env.reset(seed=0).unsqueeze(0)
        td = actor(td)

        action = td.get(("agents", "action"))
        assert torch.allclose(action[:, 0], action[:, 1], atol=1e-6, rtol=1e-6)
        assert torch.allclose(action[:, 1], action[:, 2], atol=1e-6, rtol=1e-6)
    finally:
        env.close()


def test_flex_critic_mask_changes_values_when_agent_is_masked() -> None:
    config = _make_config(actor_mode="shared", critic_mode="centralized")
    env = make_data_collection_env(config)

    try:
        actor = create_actor(env, torch.device("cpu"), config)
        critic = create_critic(env, torch.device("cpu"), config)

        base_td = env.reset(seed=0).unsqueeze(0)
        base_td = actor(base_td)

        all_true = base_td.clone()
        all_true.set(("agents", "mask"), torch.ones_like(all_true.get(("agents", "mask"))))
        value_all_true = critic(all_true).get(("agents", "state_action_value"))

        masked = base_td.clone()
        mask = masked.get(("agents", "mask")).clone()
        mask[..., 1] = False
        masked.set(("agents", "mask"), mask)
        value_masked = critic(masked).get(("agents", "state_action_value"))

        assert not torch.allclose(value_all_true, value_masked, atol=1e-6, rtol=1e-6)
    finally:
        env.close()
