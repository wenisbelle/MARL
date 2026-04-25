import torch

from data_harvesting.algorithm import MADDPGAlgorithm
from data_harvesting.collector import create_collector
from data_harvesting.environment.data_collection import make_data_collection_env
from data_harvesting.loss import _reduce


def _loss_test_config() -> dict:
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
            "centralized": False,
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
            "enabled": False,
        },
    }

def _collect_batch(algorithm: MADDPGAlgorithm, config: dict) -> torch.Tensor:
    with create_collector(
        algorithm.exploratory_policy,
        torch.device("cpu"),
        lambda: make_data_collection_env(config),
        config,
    ) as collector:
        batch = next(iter(collector))
    return batch.reshape(-1)


def _corrupt_agent_1(td) -> None:
    td.get(("agents", "observation"))[:, 1, :] = 1000.0
    td.get(("agents", "action"))[:, 1, :] = 1.0
    td.get(("next", "agents", "observation"))[:, 1, :] = -1000.0
    td.get(("next", "agents", "reward"))[:, 1, :] = 999.0


def test_reduce_with_mask_matches_expected_mean_and_sum() -> None:
    values = torch.tensor([[1.0, 100.0], [3.0, 200.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    mean_result = _reduce(values, "mean", mask=mask)
    sum_result = _reduce(values, "sum", mask=mask)

    assert float(mean_result.item()) == 2.0
    assert float(sum_result.item()) == 4.0


def test_masked_ddpg_loss_ignores_masked_agent_corruption() -> None:
    config = _loss_test_config()
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        loss_module = algorithm.loss_module
        loss_module.set_keys(mask=("agents", "mask"))

        batch = _collect_batch(algorithm, config)

        masked_clean = batch.clone()
        masked_clean.get(("agents", "mask"))[:, 1] = False

        masked_corrupt = masked_clean.clone()
        _corrupt_agent_1(masked_corrupt)

        out_clean = loss_module(masked_clean)
        out_corrupt_masked = loss_module(masked_corrupt)

        assert torch.allclose(out_clean["loss_actor"], out_corrupt_masked["loss_actor"], atol=1e-4, rtol=1e-4)
        assert torch.allclose(out_clean["loss_value"], out_corrupt_masked["loss_value"], atol=1e-4, rtol=1e-4)

        unmasked_corrupt = batch.clone()
        _corrupt_agent_1(unmasked_corrupt)
        unmasked_corrupt.get(("agents", "mask"))[:, :] = True

        out_corrupt_unmasked = loss_module(unmasked_corrupt)

        actor_diff = torch.abs(out_clean["loss_actor"] - out_corrupt_unmasked["loss_actor"]).item()
        value_diff = torch.abs(out_clean["loss_value"] - out_corrupt_unmasked["loss_value"]).item()

        assert actor_diff > 1e-5 or value_diff > 1e-5
    finally:
        env.close()


def test_dying_agent_current_mask_controls_both_losses_even_when_next_mask_is_false() -> None:
    config = _loss_test_config()
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        loss_module = algorithm.loss_module
        loss_module.set_keys(mask=("agents", "mask"))

        batch = _collect_batch(algorithm, config)
        sample = batch[:4].clone()
        sample.get(("next", "agents", "mask"))[:, 1] = False
        sample.get(("next", "agents", "done"))[:, 1, 0] = True
        sample.get(("next", "agents", "terminated"))[:, 1, 0] = True

        current_alive = sample.clone()
        current_alive.get(("agents", "mask"))[:, 1] = True
        _corrupt_agent_1(current_alive)

        current_dead = current_alive.clone()
        current_dead.get(("agents", "mask"))[:, 1] = False

        out_alive = loss_module(current_alive)
        out_dead = loss_module(current_dead)

        assert not torch.allclose(out_alive["loss_actor"], out_dead["loss_actor"], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(out_alive["loss_value"], out_dead["loss_value"], atol=1e-6, rtol=1e-6)
    finally:
        env.close()


def test_dead_agent_with_current_and_next_mask_false_is_fully_ignored() -> None:
    config = _loss_test_config()
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        loss_module = algorithm.loss_module
        loss_module.set_keys(mask=("agents", "mask"))

        batch = _collect_batch(algorithm, config)

        clean = batch[:4].clone()
        clean.get(("agents", "mask"))[:, 1] = False
        clean.get(("next", "agents", "mask"))[:, 1] = False
        clean.get(("agents", "done"))[:, 1, 0] = True
        clean.get(("agents", "terminated"))[:, 1, 0] = False
        clean.get(("next", "agents", "done"))[:, 1, 0] = True
        clean.get(("next", "agents", "terminated"))[:, 1, 0] = False

        corrupt = clean.clone()
        _corrupt_agent_1(corrupt)

        out_clean = loss_module(clean)
        out_corrupt = loss_module(corrupt)

        assert torch.allclose(out_clean["loss_actor"], out_corrupt["loss_actor"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(out_clean["loss_value"], out_corrupt["loss_value"], atol=1e-6, rtol=1e-6)
    finally:
        env.close()

