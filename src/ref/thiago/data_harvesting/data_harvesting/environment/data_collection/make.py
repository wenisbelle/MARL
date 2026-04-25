from torchrl.envs import EnvBase

from data_harvesting.encoder.output import ActorOutputKeys
from .data_collection import DataCollectionEnvironment, DataCollectionEnvironmentConfig
from .metrics import make_data_collection_metrics_spec


def make_data_collection_env(config: dict) -> EnvBase:
    """
    Create a torchrl-wrapped GrADySEnvironment.
    """
    env_config = config["environment"].copy()
    is_sequential = env_config.pop('sequential_obs')

    # Pass through directly; GrADySEnvironmentConfig handles validation and sampling
    gradys_config = DataCollectionEnvironmentConfig(**env_config)
    env = DataCollectionEnvironment(gradys_config)

    # If the environment is not sequential, we flatten and concatenate the observation components
    if not is_sequential:
        from torchrl.envs.transforms import CatTensors, FlattenObservation
        env = env.append_transform(FlattenObservation(
            first_dim=-2,
            last_dim=-1,
            in_keys=[("agents", "observation", "sensors"), ("agents", "observation", "drones")],
            out_keys=[("agents", "observation_flat", "sensors"), ("agents", "observation_flat", "drones")],
        ))
        # Conditionally include agent_id in the concatenated observation only if present
        include_id = env_config.get("id_on_state", True)
        in_keys = [("agents", "observation_flat", "sensors"), ("agents", "observation_flat", "drones")]
        if include_id:
            in_keys.append(("agents", "observation", "agent_id"))
        env = env.append_transform(CatTensors(
            in_keys=in_keys,
            out_key=("agents", "observation"),
            del_keys=False
        ))
    return env


def make_data_collection_output_dict(config: dict) -> ActorOutputKeys:
    """
    Create the output key configuration for the data collection environment based on the provided config.
    """
    env_is_sequential = config["environment"]["sequential_obs"]

    sequential_keys: dict[str, tuple[str, ...]] = {}
    flat_keys: dict[str, tuple[str, ...]] = {}

    if env_is_sequential:
        # Configuration for the drones part of the observation
        sequential_keys["drones"] = ("agents", "observation", "drones")
        # Sequential config for the sensors part of the observation
        sequential_keys["sensors"] = ("agents", "observation", "sensors")
        if config["environment"]["id_on_state"]:
            # Flat config for the agent_id part of the observation
            flat_keys["agent_id"] = ("agents", "observation", "agent_id")
    else:
        # Flat config for the entire observation when not sequential
        flat_keys["observation"] = ("agents", "observation")

    return ActorOutputKeys(
        sequential=sequential_keys,
        flat=flat_keys
    )
