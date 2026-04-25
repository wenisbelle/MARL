import torch
from typing import Any, Dict, TypedDict
from torchrl.objectives import ValueEstimators, SoftUpdate
from torchrl.objectives.ppo import ClipPPOLoss

from data_harvesting.environment import requires_masking
from data_harvesting.loss import MaskedDDPGLoss

def create_loss(policy: torch.nn.Module, critic: torch.nn.Module, config: Dict[str, Any], device: torch.device) -> MaskedDDPGLoss:
    """
    Creates the DDPG loss module using parameters from config.
    Args:
        policy: The actor network.
        critic: The critic network.
        config: Configuration dictionary (expects 'optimization' section).
    Returns:
        Configured DDPGLoss instance.
    """
    gamma = config["optimization"]["gamma"]
    loss_module = MaskedDDPGLoss(
        actor_network=policy,
        value_network=critic,
        delay_value=True,
        delay_actor=True,
        loss_function="l2",
    )
    loss_module.set_keys(
        state_action_value=("agents", "state_action_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    if requires_masking(config):
        loss_module.set_keys(mask=("agents", "mask"))

    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma, device=device)
    return loss_module

OptimizerDict = TypedDict("OptimizerDict", {"loss_actor": torch.optim.Optimizer, "loss_value": torch.optim.Optimizer})

def create_optimizers(loss_module: MaskedDDPGLoss, config: Dict[str, Any]) -> OptimizerDict:
    """
    Creates optimizers for the actor and critic using parameters from config.
    Args:
        loss_module: The MaskedDDPGLoss module.
        config: Configuration dictionary (expects 'optimization' section).
    Returns:
        Dictionary with 'loss_actor' and 'loss_value' optimizers.
    """
    lr = config["optimization"]["lr"]
    optimizers = {
        "loss_actor": torch.optim.Adam(
            loss_module.actor_network_params.flatten_keys().values(), lr=lr
        ),
        "loss_value": torch.optim.Adam(
            loss_module.value_network_params.flatten_keys().values(), lr=lr
        ),
    }
    return optimizers

def create_updater(loss_module: MaskedDDPGLoss, config: Dict[str, Any]) -> SoftUpdate:
    """
    Creates a SoftUpdate target network updater using parameters from config.
    Args:
        loss_module: The DDPGLoss module.
        config: Configuration dictionary (expects 'optimization' section).
    Returns:
        Configured SoftUpdate instance.
    """
    tau = config["optimization"]["tau"]
    return SoftUpdate(loss_module, tau=tau)


def create_ppo_loss(policy: torch.nn.Module, value_net: torch.nn.Module, config: Dict[str, Any]) -> ClipPPOLoss:
    """Create a PPO loss with GAE value estimator and proper key bindings."""
    if requires_masking(config):
        raise NotImplementedError("PPO Loss does not support environments that require masking.")

    ppo_cfg = config["ppo"]
    clip_epsilon = ppo_cfg["clip_epsilon"]
    entropy_coef = ppo_cfg["entropy_coef"]
    value_coef = ppo_cfg["value_coef"]

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_net,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        critic_coef=value_coef,
    )
    loss_module.set_keys(
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    # Value estimator (GAE)
    gamma = config["optimization"]["gamma"]
    gae_lambda = ppo_cfg["gae_lambda"]
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=gae_lambda)
    return loss_module


PPOOptimizerDict = TypedDict("PPOOptimizerDict", {"loss_policy": torch.optim.Optimizer, "loss_value": torch.optim.Optimizer})

def create_ppo_optimizers(loss_module: ClipPPOLoss, config: Dict[str, Any]) -> PPOOptimizerDict:
    lr = config["optimization"]["lr"]
    return {
        "loss_policy": torch.optim.Adam(loss_module.actor_network_params.flatten_keys().values(), lr=lr),
        "loss_value": torch.optim.Adam(loss_module.critic_network_params.flatten_keys().values(), lr=lr),
    }

