import torch
from env.mapping_environment import FlagMessage
from torchrl.envs.utils import check_env_specs

# adjust the import path to wherever you put the env
from env.mapping_environment import MappingEnvironment, MappingEnvironmentConfig


def describe_td(td, prefix=""):
    """Pretty-print the structure of a tensordict for debugging."""
    for key, value in td.items():
        full = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "items"):  # nested TensorDict
            describe_td(value, full)
        else:
            print(f"  {full:50s} shape={tuple(value.shape)} dtype={value.dtype}")


def main():
    torch.manual_seed(0)

    config = MappingEnvironmentConfig(
        render_mode="visual",            #"visual",             # no browser
        algorithm_iteration_interval=1.0,
        min_num_agents=3,
        max_num_agents=3,
        map_width=50,                 # small map -> fast iteration
        map_height=50,
        observation_map_size=20,
        max_episode_length=1000,
        agent_death_probability=0.0,  # turn off random death for the first test
    )
    env = MappingEnvironment(config)

    # 1) Structural check. This is the most valuable single test.
    #    It samples random inputs, runs reset/step, and verifies every
    #    spec (obs, action, reward, done) matches the actual tensors.
    print("Running check_env_specs ...")
    check_env_specs(env)
    print("  OK\n")

    # 2) Look at a reset.
    td = env.reset()
    print("After reset, tensordict contains:")
    describe_td(td)
    print()

    for step in range(1000):
        encounter_flag = td["agents", "observation", "encounter_flag"]   # (N, 4)
        mask           = td["agents", "mask"]                             # (N,)
        flag_idx       = encounter_flag.argmax(dim=-1)                    # (N,)
        has_flag       = (flag_idx != FlagMessage.NONE.value) & mask      # (N,)
        decision_step  = has_flag.any().item()

        # Random actions for everyone first.
        td = env.rand_action(td)

        action_tensor = td["agents", "action"]
        td["agents", "action"] = torch.randint(
            low=0,
            high=11,  # exclusive upper bound, so it generates 0 through 10
            size=action_tensor.shape,
            device=action_tensor.device
        ).to(action_tensor.dtype)

        td = env.step(td)

        if decision_step:
            flagged = has_flag.nonzero(as_tuple=True)[0].tolist()
            print(
                f"  step {step:4d}  flagged={flagged}  "
                f"reward = {td['next', 'agents', 'reward'].squeeze(-1).tolist()}  "
                f"global = {td['next', 'global_reward'].item():.4f}"
            )

        if td["next", "done"].item():
            td = env.reset()
            continue
        td = td["next"]


    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()