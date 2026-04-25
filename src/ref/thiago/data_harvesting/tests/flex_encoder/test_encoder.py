import torch
import pytest

from data_harvesting.actor import create_actor
from data_harvesting.critic import create_critic
from data_harvesting.encoder import (
    FlatEncoderConfig,
    FlatEncoderInput,
    MultiAgentFlexModule,
    SequentialEncoderConfig,
    SequentialEncoderInput,
)
from data_harvesting.environment.data_collection import make_data_collection_env


def _sequential_config(*, max_agents: int = 4, agentic_encoding: bool = False) -> SequentialEncoderConfig:
    return SequentialEncoderConfig(
        embed_dim=16,
        head_dim=8,
        num_heads=2,
        ff_dim=32,
        depth=1,
        dropout=0.0,
        max_num_agents=max_agents,
        agentic_encoding=agentic_encoding,
    )


def _flat_config() -> FlatEncoderConfig:
    return FlatEncoderConfig(
        embed_dim=8,
        depth=1,
        num_cells=16,
        activation_class=torch.nn.Tanh,
    )


def _make_shared_encoder(n_agents: int = 3, output_dim: int = 5) -> MultiAgentFlexModule:
    return MultiAgentFlexModule(
        sequential_config=_sequential_config(max_agents=n_agents, agentic_encoding=False),
        sequential_inputs=[SequentialEncoderInput(key="drones", input_size=2)],
        flat_inputs=[FlatEncoderInput(key="agent_id", input_size=1)],
        flat_config=_flat_config(),
        mix_layer_depth=1,
        mix_layer_num_cells=32,
        mix_activation_class=torch.nn.Tanh,
        output_dim=output_dim,
        n_agents=n_agents,
        centralized=False,
        share_params=True,
        device=torch.device("cpu"),
    )


def _make_centralized_encoder(n_agents: int = 3, output_dim: int = 4) -> MultiAgentFlexModule:
    return MultiAgentFlexModule(
        sequential_config=_sequential_config(max_agents=n_agents, agentic_encoding=False),
        sequential_inputs=[SequentialEncoderInput(key="drones", input_size=2)],
        flat_inputs=[],
        flat_config=_flat_config(),
        mix_layer_depth=1,
        mix_layer_num_cells=16,
        mix_activation_class=torch.nn.Tanh,
        output_dim=output_dim,
        n_agents=n_agents,
        centralized=True,
        share_params=False,
        device=torch.device("cpu"),
    )


def _make_per_agent_encoder(n_agents: int = 3, output_dim: int = 5) -> MultiAgentFlexModule:
    return MultiAgentFlexModule(
        sequential_config=_sequential_config(max_agents=n_agents, agentic_encoding=False),
        sequential_inputs=[SequentialEncoderInput(key="drones", input_size=2)],
        flat_inputs=[FlatEncoderInput(key="agent_id", input_size=1)],
        flat_config=_flat_config(),
        mix_layer_depth=1,
        mix_layer_num_cells=32,
        mix_activation_class=torch.nn.Tanh,
        output_dim=output_dim,
        n_agents=n_agents,
        centralized=False,
        share_params=False,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("mode", ["shared", "per_agent", "centralized"])
def test_flex_mode_output_shape_is_correct(mode: str) -> None:
    torch.manual_seed(0)

    if mode == "shared":
        module = _make_shared_encoder(n_agents=3, output_dim=5)
        observation = {
            "drones": torch.randn(2, 3, 4, 2),
            "agent_id": torch.randn(2, 3, 1),
        }
        expected_shape = (2, 3, 5)
    elif mode == "per_agent":
        module = _make_per_agent_encoder(n_agents=3, output_dim=5)
        observation = {
            "drones": torch.randn(2, 3, 4, 2),
            "agent_id": torch.randn(2, 3, 1),
        }
        expected_shape = (2, 3, 5)
    else:
        module = _make_centralized_encoder(n_agents=3, output_dim=4)
        observation = {
            "drones": torch.randn(2, 3, 4, 2),
        }
        expected_shape = (2, 3, 4)

    mask = torch.ones(2, 3, dtype=torch.bool)
    output = module(mask=mask, **observation)

    assert tuple(output.shape) == expected_shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("mode", ["shared", "per_agent", "centralized"])
def test_flex_mode_all_true_mask_matches_no_mask(mode: str) -> None:
    torch.manual_seed(0)

    if mode == "shared":
        module = _make_shared_encoder(n_agents=3, output_dim=5)
        observation = {
            "drones": torch.randn(2, 3, 4, 2),
            "agent_id": torch.randn(2, 3, 1),
        }
    elif mode == "per_agent":
        module = _make_per_agent_encoder(n_agents=3, output_dim=5)
        observation = {
            "drones": torch.randn(2, 3, 4, 2),
            "agent_id": torch.randn(2, 3, 1),
        }
    else:
        module = _make_centralized_encoder(n_agents=3, output_dim=4)
        observation = {
            "drones": torch.randn(2, 3, 4, 2),
        }

    all_true_mask = torch.ones(2, 3, dtype=torch.bool)

    output_nomask = module(mask=None, **observation)
    output_all_true_mask = module(mask=all_true_mask, **observation)

    assert torch.allclose(output_nomask, output_all_true_mask, atol=1e-6, rtol=1e-6)


def _actor_critic_flex_config() -> dict:
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


def test_flex_shared_encoder_output_shape_is_correct() -> None:
    torch.manual_seed(0)
    module = _make_shared_encoder(n_agents=3, output_dim=5)

    drones = torch.randn(2, 3, 4, 2)
    agent_id = torch.randn(2, 3, 1)
    mask = torch.ones(2, 3, dtype=torch.bool)

    output = module(mask=mask, drones=drones, agent_id=agent_id)

    assert tuple(output.shape) == (2, 3, 5)
    assert torch.isfinite(output).all()


def test_flex_centralized_encoder_repeats_output_for_all_agents() -> None:
    torch.manual_seed(0)
    module = _make_centralized_encoder(n_agents=3, output_dim=4)

    drones = torch.randn(2, 3, 4, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)

    output = module(mask=mask, drones=drones)

    assert tuple(output.shape) == (2, 3, 4)
    assert torch.allclose(output[:, 0], output[:, 1])
    assert torch.allclose(output[:, 1], output[:, 2])


def test_flex_encoder_raises_on_missing_key() -> None:
    module = _make_shared_encoder(n_agents=3, output_dim=5)

    drones = torch.randn(2, 3, 4, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)

    try:
        module(mask=mask, drones=drones)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for missing 'agent_id' key")


def test_flex_shared_encoder_mask_changes_outputs_and_gradients() -> None:
    torch.manual_seed(0)
    module = _make_shared_encoder(n_agents=3, output_dim=5)

    drones_nomask = torch.randn(2, 3, 4, 2, requires_grad=True)
    drones_masked = drones_nomask.detach().clone().requires_grad_(True)
    agent_id = torch.randn(2, 3, 1)
    mask = torch.tensor([[True, False, True], [True, False, True]], dtype=torch.bool)

    output_nomask = module(mask=None, drones=drones_nomask, agent_id=agent_id)
    output_masked = module(mask=mask, drones=drones_masked, agent_id=agent_id)

    assert not torch.allclose(output_nomask, output_masked, atol=1e-6, rtol=1e-6)

    output_nomask.sum().backward()
    output_masked.sum().backward()
    assert drones_nomask.grad is not None
    assert drones_masked.grad is not None
    assert not torch.allclose(drones_nomask.grad, drones_masked.grad, atol=1e-6, rtol=1e-6)


def test_flex_actor_and_critic_integration_smoke() -> None:
    config = _actor_critic_flex_config()
    env = make_data_collection_env(config)

    try:
        actor = create_actor(env, torch.device("cpu"), config)
        critic = create_critic(env, torch.device("cpu"), config)

        td = env.reset(seed=0).unsqueeze(0)
        td = actor(td)
        td = critic(td)

        assert td.get(("agents", "action")).shape[-3:] == (1, 3, 2)
        assert td.get(("agents", "state_action_value")).shape[-3:] == (1, 3, 1)
        assert torch.isfinite(td.get(("agents", "state_action_value"))).all()
    finally:
        env.close()