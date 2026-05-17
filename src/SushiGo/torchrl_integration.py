"""
TorchRL integration for the Sushi Go! environment.

This module puts the PettingZoo environment "into the TorchRL framework": it wraps
the env, exposes the grouped tensordict keys, and builds a masked multi-agent actor
and a multi-agent critic. It deliberately stops short of a training loop — plug the
returned modules into your own MARL algorithm (PPO / MAPPO / IPPO / ...).

Recommended setup for Sushi Go
------------------------------
Sushi Go is a *competitive, symmetric* game (one player's gain is roughly another's
loss), so the natural setup is **shared-parameter self-play**: a single policy
network controls every seat, and each seat optimises its own return. This differs
from MAPPO's cooperative shared-reward assumption — if you use a MAPPO-style
centralised critic, keep per-agent (individual) rewards rather than summing them.

  * Actor : MultiAgentMLP, `share_params=True` (self-play), `centralised=False`.
            Outputs Discrete(12) logits sampled through a MaskedCategorical, so the
            policy can only ever draft a card that is actually in its hand.
  * Critic: MultiAgentMLP. `centralised=False` -> independent critic on each agent's
            (already opponent-aware) observation. Set `centralised=True` for a
            MAPPO-style centralised critic.

Tested with: torch 2.12, torchrl 0.12, tensordict 0.12.
"""
import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import RewardSum, TransformedEnv, check_env_specs
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.modules import MaskedCategorical, MultiAgentMLP, ProbabilisticActor, ValueOperator

from Dev.src.SushiGo.sushi_go_env import SushiGoParallelEnv, N_TYPES

# Grouped tensordict keys produced by the PettingZoo wrapper. The env's Dict
# observation nests its "observation" entry, hence the doubled key below.
GROUP = "players"
OBS_KEY = (GROUP, "observation", "observation")
MASK_KEY = (GROUP, "action_mask")
ACTION_KEY = (GROUP, "action")
LOGITS_KEY = (GROUP, "logits")
VALUE_KEY = (GROUP, "state_value")


def make_torchrl_env(n_players=3, history_len=None, include_opponent_tableaus=True,
                     reward_scale=0.1, device="cpu"):
    """Build a Sushi Go env wrapped for TorchRL (one fixed player count per env)."""
    base = SushiGoParallelEnv(
        n_players=n_players,
        history_len=history_len,
        include_opponent_tableaus=include_opponent_tableaus,
        reward_scale=reward_scale,
    )
    env = PettingZooWrapper(
        base,
        use_mask=True,             # exposes ("players", "action_mask")
        categorical_actions=True,  # integer actions, not one-hot
        group_map={GROUP: [f"player_{i}" for i in range(n_players)]},
        device=device,
    )
    # Tracks the per-seat episode return at ("next", "players", "episode_reward").
    env = TransformedEnv(
        env, RewardSum(in_keys=[(GROUP, "reward")], out_keys=[(GROUP, "episode_reward")])
    )
    return env


def build_actor(n_players, obs_dim, num_cells=128, depth=2, device="cpu"):
    """Masked multi-agent policy: obs -> logits -> MaskedCategorical -> action."""
    net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=N_TYPES,      # one logit per card type
        n_agents=n_players,
        centralised=False,            # decentralised actor (acts on its own obs)
        share_params=True,            # one shared policy across all seats (self-play)
        depth=depth,
        num_cells=num_cells,
        activation_class=torch.nn.Tanh,
        device=device,
    )
    module = TensorDictModule(net, in_keys=[OBS_KEY], out_keys=[LOGITS_KEY])
    actor = ProbabilisticActor(
        module=module,
        in_keys={"logits": LOGITS_KEY, "mask": MASK_KEY},
        out_keys=[ACTION_KEY],
        distribution_class=MaskedCategorical,   # forbids drafting absent cards
        return_log_prob=True,
        log_prob_key=(GROUP, "sample_log_prob"),
    )
    return actor


def build_critic(n_players, obs_dim, num_cells=128, depth=2,
                 centralised=False, device="cpu"):
    """Multi-agent value network. `centralised=True` -> MAPPO-style shared critic."""
    net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=n_players,
        centralised=centralised,
        share_params=True,
        depth=depth,
        num_cells=num_cells,
        activation_class=torch.nn.Tanh,
        device=device,
    )
    return ValueOperator(net, in_keys=[OBS_KEY], out_keys=[VALUE_KEY])


# --------------------------------------------------------------------------------------
# Wiring check — confirms the env + actor + critic are correctly integrated.
# This is verification only; it does NOT train anything.
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    for n in (2, 3, 4):
        env = make_torchrl_env(n_players=n)
        check_env_specs(env)
        obs_dim = env.observation_spec[OBS_KEY].shape[-1]

        actor = build_actor(n, obs_dim)
        critic = build_critic(n, obs_dim)

        td = env.reset()
        actor(td)              # populate logits + action + log-prob
        critic(td)             # populate state value
        td = env.step(td)

        # A short random-policy rollout exercises reset/step/termination.
        rollout = env.rollout(max_steps=120, policy=actor)

        print(f"n_players={n}: obs_dim={obs_dim} | "
              f"action shape={td[ACTION_KEY].shape} | "
              f"value shape={td[VALUE_KEY].shape} | "
              f"rollout length={rollout.batch_size}")
    print("\nEnvironment, actor and critic are correctly wired for TorchRL.")
    print("Attach your MARL training loop (collector + PPO/MAPPO loss) to these modules.")
