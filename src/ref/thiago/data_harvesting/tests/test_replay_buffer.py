import torch
from tensordict import TensorDict

from data_harvesting.replay import create_replay_buffer


def _replay_config(*, buffer_size: int = 32, batch_size: int = 8, prefetch: int = 0, buffer_device: str = "cpu") -> dict:
    return {
        "training": {
            "batch_size": batch_size,
        },
        "replay_buffer": {
            "buffer_size": buffer_size,
            "prefetch": prefetch,
            "buffer_device": buffer_device,
        },
    }


def _make_batch(frames: int, *, n_agents: int = 2, action_dim: int = 2) -> TensorDict:
    return TensorDict(
        {
            "agents": {
                "action": torch.rand(frames, n_agents, action_dim),
                "reward": torch.rand(frames, n_agents, 1),
                "done": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                "terminated": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                "mask": torch.ones(frames, n_agents, dtype=torch.bool),
                "observation": {
                    "sensors": torch.rand(frames, n_agents, 1, 2),
                    "drones": torch.rand(frames, n_agents, 1, 2),
                },
            },
            "next": {
                "agents": {
                    "reward": torch.rand(frames, n_agents, 1),
                    "done": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                    "terminated": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                    "observation": {
                        "sensors": torch.rand(frames, n_agents, 1, 2),
                        "drones": torch.rand(frames, n_agents, 1, 2),
                    },
                }
            },
        },
        batch_size=[frames],
    )


def test_replay_buffer_uses_configured_batch_size() -> None:
    config = _replay_config(buffer_size=64, batch_size=7, prefetch=0)
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))

    assert replay_buffer.batch_size == 7


def test_replay_buffer_sample_has_expected_shapes_and_keys() -> None:
    config = _replay_config(buffer_size=64, batch_size=6, prefetch=0)
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))
    replay_buffer.extend(_make_batch(24))

    sample = replay_buffer.sample()

    assert sample.batch_size == torch.Size([6])
    assert tuple(sample.get(("agents", "action")).shape) == (6, 2, 2)
    assert tuple(sample.get(("agents", "reward")).shape) == (6, 2, 1)
    assert tuple(sample.get(("next", "agents", "observation", "sensors")).shape) == (6, 2, 1, 2)
    assert sample.get(("agents", "done")).dtype == torch.bool
    assert sample.get(("agents", "terminated")).dtype == torch.bool


def test_replay_buffer_respects_capacity() -> None:
    config = _replay_config(buffer_size=10, batch_size=4, prefetch=0)
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))
    replay_buffer.extend(_make_batch(25))

    assert len(replay_buffer) == 10


def test_replay_buffer_samples_on_requested_device() -> None:
    config = _replay_config(buffer_size=32, batch_size=5, prefetch=0, buffer_device="cpu")
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))
    replay_buffer.extend(_make_batch(16))

    sample = replay_buffer.sample()

    assert sample.get(("agents", "action")).device.type == "cpu"
    assert sample.get(("next", "agents", "reward")).device.type == "cpu"
