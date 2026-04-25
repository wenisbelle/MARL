from __future__ import annotations

import pytest
import torch
from torch import nn

from data_harvesting.actor import create_actor, create_ppo_actor
from data_harvesting.critic import create_critic, create_ppo_value_net
from data_harvesting.environment import requires_masking as requires_environment_masking
from data_harvesting.environment.data_collection import make_data_collection_env
from data_harvesting.environment.data_collection.config import (
    requires_masking as requires_data_collection_masking,
)
from data_harvesting.environment.data_collection.data_collection import DataCollectionEnvironmentConfig
from data_harvesting.algorithm import MADDPGAlgorithm
from data_harvesting.optimization import create_ppo_loss


def _base_env_config(
    *,
    min_num_agents: int,
    max_num_agents: int,
    sequential_obs: bool = False,
    reward: str = "punish",
    agent_death_probability: float = 0.0,
) -> dict:
    return {
        "sequential_obs": sequential_obs,
        "algorithm_iteration_interval": 1.0,
        "min_num_agents": min_num_agents,
        "max_num_agents": max_num_agents,
        "min_num_sensors": 2,
        "max_num_sensors": 2,
        "scenario_size": 20.0,
        "max_episode_length": 50,
        "max_seconds_stalled": 20,
        "communication_range": 0.0,
        "state_num_closest_sensors": 2,
        "state_num_closest_drones": 1,
        "id_on_state": True,
        "min_sensor_priority": 1.0,
        "max_sensor_priority": 1.0,
        "full_random_drone_position": False,
        "reward": reward,
        "speed_action": True,
        "end_when_all_collected": False,
        "agent_death_probability": agent_death_probability,
    }


def _base_config(
    *,
    min_num_agents: int,
    max_num_agents: int,
    sequential_obs: bool = False,
    flex_enabled: bool = False,
    algorithm: str = "maddpg",
    agent_death_probability: float = 0.0,
) -> dict:
    return {
        "environment": _base_env_config(
            min_num_agents=min_num_agents,
            max_num_agents=max_num_agents,
            sequential_obs=sequential_obs,
            agent_death_probability=agent_death_probability,
        ),
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
            "algorithm": algorithm,
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
            "enabled": flex_enabled,
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
        "ppo": {
            "clip_epsilon": 0.2,
            "gae_lambda": 0.95,
            "entropy_coef": 0.0,
            "value_coef": 0.5,
            "num_epochs": 1,
            "minibatch_size": 8,
        },
    }


@pytest.mark.parametrize(
    "min_num_agents,max_num_agents,expected",
    [
        (2, 2, False),
        (1, 3, True),
    ],
)
def test_requires_masking_helpers_match_environment_config(
    min_num_agents: int,
    max_num_agents: int,
    expected: bool,
) -> None:
    data_collection_config = DataCollectionEnvironmentConfig(
        render_mode=None,
        algorithm_iteration_interval=1.0,
        min_num_agents=min_num_agents,
        max_num_agents=max_num_agents,
        min_num_sensors=2,
        max_num_sensors=2,
        scenario_size=20.0,
        max_episode_length=50,
        max_seconds_stalled=20,
        communication_range=0.0,
        state_num_closest_sensors=2,
        state_num_closest_drones=1,
        id_on_state=True,
        min_sensor_priority=1.0,
        max_sensor_priority=1.0,
        full_random_drone_position=False,
        reward="punish",
        speed_action=True,
        end_when_all_collected=False,
    )
    full_config = {
        "environment": {
            **_base_env_config(
                min_num_agents=min_num_agents,
                max_num_agents=max_num_agents,
                sequential_obs=True,
            )
        }
    }

    assert requires_data_collection_masking(data_collection_config) is expected
    assert requires_environment_masking(full_config) is expected


def test_requires_masking_when_mid_episode_death_is_enabled() -> None:
    data_collection_config = DataCollectionEnvironmentConfig(
        render_mode=None,
        algorithm_iteration_interval=1.0,
        min_num_agents=2,
        max_num_agents=2,
        min_num_sensors=2,
        max_num_sensors=2,
        scenario_size=20.0,
        max_episode_length=50,
        max_seconds_stalled=20,
        communication_range=0.0,
        state_num_closest_sensors=2,
        state_num_closest_drones=1,
        id_on_state=True,
        min_sensor_priority=1.0,
        max_sensor_priority=1.0,
        full_random_drone_position=False,
        reward="punish",
        speed_action=True,
        end_when_all_collected=False,
        agent_death_probability=0.1,
    )
    full_config = {
        "environment": {
            **_base_env_config(
                min_num_agents=2,
                max_num_agents=2,
                sequential_obs=True,
                agent_death_probability=0.1,
            )
        }
    }

    assert requires_data_collection_masking(data_collection_config) is True
    assert requires_environment_masking(full_config) is True


def test_mlp_actor_and_critic_reject_masking_required_environments() -> None:
    config = _base_config(min_num_agents=1, max_num_agents=3, sequential_obs=False, flex_enabled=False)
    env = make_data_collection_env(config)
    try:
        with pytest.raises(NotImplementedError, match="masking"):
            create_actor(env, torch.device("cpu"), config)

        with pytest.raises(NotImplementedError, match="masking"):
            create_critic(env, torch.device("cpu"), config)
    finally:
        env.close()


def test_mappo_components_reject_masking_required_environments() -> None:
    config = _base_config(
        min_num_agents=1,
        max_num_agents=3,
        sequential_obs=False,
        flex_enabled=False,
        algorithm="mappo",
    )
    env = make_data_collection_env(config)
    try:
        with pytest.raises(NotImplementedError, match="masking"):
            create_ppo_actor(env, torch.device("cpu"), config)

        with pytest.raises(NotImplementedError, match="masking"):
            create_ppo_value_net(env, torch.device("cpu"), config)

        with pytest.raises(NotImplementedError, match="masking"):
            create_ppo_loss(nn.Identity(), nn.Identity(), config)
    finally:
        env.close()


def test_maddpg_loss_binds_mask_when_environment_requires_it() -> None:
    config = _base_config(min_num_agents=1, max_num_agents=3, sequential_obs=True, flex_enabled=True)
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        assert algorithm.loss_module.tensor_keys.mask == ("agents", "mask")
    finally:
        env.close()
