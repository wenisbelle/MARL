from torchrl.data import ReplayBuffer, RandomSampler, LazyTensorStorage
from typing import Any, Dict

def create_replay_buffer(config: Dict[str, Any], device: Any) -> ReplayBuffer:
    """
    Creates a replay buffer using parameters from config.
    Args:
        config: Configuration dictionary (expects 'replay_buffer' and 'training' sections).
        device: The device to move samples to.
    Returns:
        Configured ReplayBuffer instance.
    """
    buffer_size = config["replay_buffer"]["buffer_size"]
    batch_size = config["training"]["batch_size"]
    prefetch = config["replay_buffer"]["prefetch"]
    buffer_device = config["replay_buffer"]["buffer_device"]
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(buffer_size, device=buffer_device),
        sampler=RandomSampler(),
        batch_size=batch_size,
        prefetch=prefetch,
    )
    replay_buffer.append_transform(lambda x: x.to(device))
    return replay_buffer
