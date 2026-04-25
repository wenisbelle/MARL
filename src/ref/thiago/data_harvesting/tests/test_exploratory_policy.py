from typing import cast
import torch
from tensordict.nn import TensorDictSequential
from torchrl.modules import AdditiveGaussianModule

from data_harvesting.actor import create_actor, create_exploratory_actor
from data_harvesting.environment.data_collection import make_data_collection_env


def _exploration_test_config(*, sigma_init: float = 0.2, sigma_end: float = 0.1, anneal_steps: int = 5) -> dict:
    return {
        "environment": {
            "sequential_obs": False,
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 2,
            "max_num_agents": 2,
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
            "exploration_sigma_init": sigma_init,
            "exploration_sigma_end": sigma_end,
            "exploration_annealing_steps": anneal_steps,
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
            "enabled": False,
        },
    }


def test_create_exploratory_actor_returns_expected_modules() -> None:
    config = _exploration_test_config(sigma_init=0.2, sigma_end=0.1, anneal_steps=5)
    env = make_data_collection_env(config)
    try:
        actor = create_actor(env, torch.device("cpu"), config)
        exploratory_actor, exploration_noise = create_exploratory_actor(actor, torch.device("cpu"), config)

        assert isinstance(exploratory_actor, TensorDictSequential)
        assert isinstance(exploration_noise, AdditiveGaussianModule)
        assert abs(float(cast(torch.Tensor, exploration_noise.sigma).item()) - 0.2) < 1e-6
        assert abs(float(cast(torch.Tensor, exploration_noise.sigma_end).item()) - 0.1) < 1e-6
    finally:
        env.close()


def test_exploration_noise_sigma_anneals_until_end() -> None:
    config = _exploration_test_config(sigma_init=0.2, sigma_end=0.1, anneal_steps=5)
    env = make_data_collection_env(config)
    try:
        actor = create_actor(env, torch.device("cpu"), config)
        _, exploration_noise = create_exploratory_actor(actor, torch.device("cpu"), config)

        sigmas = [float(cast(torch.Tensor, exploration_noise.sigma).item())]
        for _ in range(10):
            exploration_noise.step()
            sigmas.append(float(cast(torch.Tensor, exploration_noise.sigma).item()))

        for prev, curr in zip(sigmas, sigmas[1:], strict=False):
            assert curr <= prev
        assert abs(sigmas[-1] - 0.1) < 1e-6
    finally:
        env.close()


def test_exploratory_actor_matches_actor_when_sigma_is_zero() -> None:
    config = _exploration_test_config(sigma_init=0.0, sigma_end=0.0, anneal_steps=1)
    env = make_data_collection_env(config)
    try:
        torch.manual_seed(0)
        actor = create_actor(env, torch.device("cpu"), config)
        exploratory_actor, _ = create_exploratory_actor(actor, torch.device("cpu"), config)

        td = env.reset(seed=0).unsqueeze(0)
        base_action = actor(td.clone()).get(("agents", "action"))
        exploratory_action = exploratory_actor(td.clone()).get(("agents", "action"))

        assert torch.allclose(base_action, exploratory_action, atol=1e-7, rtol=0)
    finally:
        env.close()


def test_exploratory_actor_changes_actions_when_sigma_positive() -> None:
    config = _exploration_test_config(sigma_init=0.5, sigma_end=0.5, anneal_steps=1)
    env = make_data_collection_env(config)
    try:
        torch.manual_seed(0)
        actor = create_actor(env, torch.device("cpu"), config)
        exploratory_actor, _ = create_exploratory_actor(actor, torch.device("cpu"), config)

        td = env.reset(seed=0).unsqueeze(0)
        base_action = actor(td.clone()).get(("agents", "action"))

        torch.manual_seed(123)
        exploratory_action = exploratory_actor(td.clone()).get(("agents", "action"))

        assert not torch.allclose(base_action, exploratory_action, atol=1e-6, rtol=1e-6)
        assert torch.all(exploratory_action >= 0.0)
        assert torch.all(exploratory_action <= 1.0)
    finally:
        env.close()
