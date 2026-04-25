from contextlib import contextmanager
import torch
from typing import Callable, Any, Dict
from torchrl.collectors import DataCollectorBase, MultiaSyncDataCollector, aSyncDataCollector, SyncDataCollector, MultiSyncDataCollector
from tensordict.nn import TensorDictModule

def _shutdown_collector(collector: Any) -> None:
    """Best-effort shutdown for collectors across TorchRL versions."""
    if hasattr(collector, "shutdown"):
        try:
            collector.shutdown()
            return
        except Exception:
            pass
    if hasattr(collector, "stop"):
        try:
            collector.stop()
            return
        except Exception:
            pass
    if hasattr(collector, "close"):
        try:
            collector.close()
        except Exception:
            pass

def _create_async_collector(
    exploratory_policy: TensorDictModule,
    device: torch.device,
    config: Dict[str, Any],
    env_creator: Callable[[], Any]
) -> Any:
    """
    Creates an asynchronous data collector (single or multi) for RL training.
    Args:
        exploratory_policy: The policy module used for exploration.
        device: The device to run the collector on.
        config: Configuration dictionary (expects 'collector' and 'training' sections).
        env_creator: Function that creates a new environment instance.
    Returns:
        An instance of aSyncDataCollector or MultiaSyncDataCollector.
    """
    num_collectors = config["collector"]["num_collectors"]
    frames_per_batch = config["collector"]["frames_per_batch"]
    total_steps = config["training"]["total_timesteps"]
    if num_collectors == 1:
        return aSyncDataCollector(
            create_env_fn=env_creator,
            policy=exploratory_policy,
            device=device,
            env_device="cpu",
            policy_device=device,
            total_frames=total_steps,
            frames_per_batch=frames_per_batch,
        )
    else:
        return MultiaSyncDataCollector(
            create_env_fn=[env_creator] * num_collectors,
            policy=exploratory_policy,
            device=device,
            env_device="cpu",
            policy_device=device,
            total_frames=total_steps,
            frames_per_batch=frames_per_batch,
        )

def _create_sync_collector(
    exploratory_policy: TensorDictModule,
    device: torch.device,
    config: Dict[str, Any],
    env_creator: Callable[[], Any]
) -> Any:
    """
    Creates a synchronous data collector (single or multi) for RL training.
    Args:
        exploratory_policy: The policy module used for exploration.
        device: The device to run the collector on.
        config: Configuration dictionary (expects 'collector' and 'training' sections).
        env_creator: Function that creates a new environment instance.
    Returns:
        An instance of SyncDataCollector or MultiSyncDataCollector.
    """
    num_collectors = config["collector"]["num_collectors"]
    frames_per_batch = config["collector"]["frames_per_batch"]
    total_steps = config["training"]["total_timesteps"]
    if num_collectors == 1:
        return SyncDataCollector(
            create_env_fn=env_creator(),
            policy=exploratory_policy,
            device=device,
            env_device="cpu",
            policy_device=device,
            total_frames=total_steps,
            frames_per_batch=frames_per_batch,
        )
    else:
        return MultiSyncDataCollector(
            create_env_fn=[env_creator] * num_collectors,
            policy=exploratory_policy,
            device=device,
            env_device="cpu",
            policy_device=device,
            total_frames=total_steps,
            frames_per_batch=frames_per_batch,
        )

@contextmanager
def create_collector(
    exploratory_policy: TensorDictModule,
    device: torch.device,
    env_creator: Callable[[], Any],
    config: Dict[str, Any]
):
    """
    Creates a data collector for RL training, choosing async or sync and single or multi based on config.
    Args:
        exploratory_policy: The policy module used for exploration.
        device: The device to run the collector on.
        env_creator: Function that creates a new environment instance.
        config: Configuration dictionary (expects 'collector' and 'training' sections).
    Returns:
        An instance of aSyncDataCollector, MultiaSyncDataCollector, SyncDataCollector, or MultiSyncDataCollector.
    """
    async_collector = config["collector"]["async_collector"]
    if async_collector:
        collector = _create_async_collector(exploratory_policy, device, config, env_creator)
    else:
        collector = _create_sync_collector(exploratory_policy, device, config, env_creator)
    
    try:
        yield collector
    finally:
        _shutdown_collector(collector)
