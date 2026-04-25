import torch
from typing import Dict, Any
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import (
    ProbabilisticActor,
    MultiAgentMLP,
    AdditiveGaussianModule,
    NormalParamExtractor,
)
from torchrl.modules.distributions import TanhNormal
from data_harvesting.encoder import (
    FlatEncoderInput,
    MultiAgentFlexModule,
    SequentialEncoderConfig,
    FlatEncoderConfig,
    SequentialEncoderInput,
)
from data_harvesting.environment import make_output_dict, requires_masking
from data_harvesting.utils import get_faster_tanh_delta, get_activation_class

def create_mlp_module(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    if requires_masking(config):
        raise NotImplementedError(
            "MLP Actor does not support environments that require masking. "
            "Enable the flex encoder backend instead."
        )

    if config["environment"]["sequential_obs"]:
        raise NotImplementedError("MLP Actor not implemented for sequential observations.")

    activation_class = get_activation_class(config["actor"]["activation_function"])
    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
        n_agent_outputs=env.full_action_spec[("agents", "action")].shape[-1],
        n_agents=config["environment"]["min_num_agents"],
        centralised=config["actor"]["centralized"],
        share_params=config["actor"]["share_parameters"],
        device=device,
        depth=config["actor"]["network_depth"],
        num_cells=config["actor"]["network_width"],
        activation_class=activation_class
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "param")],
    )
    return policy_module

def create_flex_policy_module(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    flex_cfg = config["flex_encoder"]
    seq_heads_cfg = flex_cfg["sequential_heads"]
    flat_heads_cfg = flex_cfg["flat_heads"]

    output_keys = make_output_dict(config)
    sequential_inputs = []
    flat_inputs = []
    in_keys = {}
    
    sequential_config = SequentialEncoderConfig(
        embed_dim=seq_heads_cfg["embed_dim"],
        head_dim=seq_heads_cfg["head_dim"],
        num_heads=seq_heads_cfg["num_heads"],
        ff_dim=seq_heads_cfg["ff_dim"],
        depth=seq_heads_cfg["depth"],
        dropout=seq_heads_cfg["dropout"],
        max_num_agents=config["environment"]["max_num_agents"],
        agentic_encoding=False
    )
    flat_config = FlatEncoderConfig(
        embed_dim=flat_heads_cfg["embed_dim"],
        depth=flat_heads_cfg["depth"],
        num_cells=flat_heads_cfg["num_cells"],
        activation_class=get_activation_class(flat_heads_cfg["activation_function"])
    )

    for key, path in output_keys["flat"].items():
        flat_inputs.append(
            FlatEncoderInput(
                key=key,
                input_size=env.observation_spec[path].shape[-1],
            )
        )
        in_keys[key] = path

    for key, path in output_keys["sequential"].items():
        sequential_inputs.append(
            SequentialEncoderInput(
                key=key,
                input_size=env.observation_spec[path].shape[-1],
            )
        )
        in_keys[key] = path

    encoder = MultiAgentFlexModule(
        sequential_inputs=sequential_inputs, 
        flat_inputs=flat_inputs,
        sequential_config=sequential_config,
        flat_config=flat_config,
        mix_layer_depth=flex_cfg["mix_layer_depth"],
        mix_layer_num_cells=flex_cfg["mix_layer_num_cells"],
        mix_activation_class=get_activation_class(flex_cfg["mix_activation_function"]),
        output_dim=env.full_action_spec[("agents", "action")].shape[-1],
        n_agents=config["environment"]["max_num_agents"],
        centralized=config["actor"]["centralized"],
        share_params=config["actor"]["share_parameters"],
        device=device
    )

    policy_module = TensorDictModule(
        encoder,
        in_keys=in_keys,
        out_keys=[("agents", "param")],
        out_to_in_map=True
    )
    return policy_module

def create_actor(
    env: EnvBase,
    device: torch.device,
    config: Dict[str, Any],
) -> ProbabilisticActor:
    """Create the deterministic (Tanh-squashed delta) multi-agent actor.

    Args:
        env: TorchRL environment providing observation and action specs.
        device: Target device for modules.
        config: Hierarchical configuration dictionary.

    Returns:
        ProbabilisticActor: Actor producing actions under a TanhDelta distribution.
    """
    policy_module = (
        create_flex_policy_module(env, config, device)
        if config["flex_encoder"]["enabled"]
        else create_mlp_module(env, config, device)
    )

    high = env.full_action_spec_unbatched["agents", "action"].space.high.to(device)
    low = env.full_action_spec_unbatched["agents", "action"].space.low.to(device)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec[("agents", "action")],
        in_keys=[("agents", "param")],
        out_keys=[("agents", "action")],
        distribution_class=get_faster_tanh_delta(high, low),
        distribution_kwargs={
            "low": low,
            "high": high,
        },
        return_log_prob=False,
    )
    return policy


def create_exploratory_actor(
    actor: ProbabilisticActor,
    device: torch.device,
    config: Dict[str, Any],
) -> tuple[TensorDictModule, AdditiveGaussianModule]:
    """
    Creates an exploratory actor by adding Gaussian noise to the actions of the given actor.
    :param actor: The base actor.
    :param device: The device to run the exploratory actor on.
    :param config: Configuration dictionary.
    :return: The exploratory actor and the exploration noise module.
    """
    exploration_noise = AdditiveGaussianModule(
        spec=actor.spec,
        annealing_num_steps=config["training"]["exploration_annealing_steps"],
        action_key=("agents", "action"),
        sigma_init=config["training"]["exploration_sigma_init"],
        sigma_end=config["training"]["exploration_sigma_end"],
        device=device
    )

    return TensorDictSequential(
        actor,
        exploration_noise
    ), exploration_noise


def create_ppo_actor(
    env: EnvBase,
    device: torch.device,
    config: Dict[str, Any],
) -> ProbabilisticActor:
    """Create a multi-agent PPO actor with a TanhNormal squashed Gaussian policy.

    Returns a ProbabilisticActor that emits actions within the environment bounds and stores
    the log probability under the key ("agents", "sample_log_prob") for PPO updates.
    """
    if requires_masking(config):
        raise NotImplementedError("PPO Actor does not support environments that require masking.")

    activation_class = get_activation_class(config["actor"]["activation_function"])
    action_dim = env.full_action_spec[("agents", "action")].shape[-1]

    # Policy backbone produces concatenated normal params (mean, std) per action dim
    policy_backbone = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
            n_agent_outputs=action_dim * 2,
            n_agents=config["environment"]["max_num_agents"],
            centralised=config["actor"]["centralized"],
            share_params=config["actor"]["share_parameters"],
            device=device,
            depth=config["actor"]["network_depth"],
            num_cells=config["actor"]["network_width"],
            activation_class=activation_class,
        ),
        NormalParamExtractor()
    )

    policy_module = TensorDictModule(
        module=policy_backbone,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # Squashed Gaussian policy with log-prob output
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        spec=env.full_action_spec[("agents", "action")],
        distribution_class=TanhNormal,
        # TanhNormal will be created from loc/scale; env spec handles bounds
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    return policy
