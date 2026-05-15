import torch

from simulation_orchestrator import AsyncMARLOrchestrator
from env.mapping_environment import MappingEnvironment, MappingEnvironmentConfig

def main():

    config = MappingEnvironmentConfig(
            render_mode=None,            #"visual",          
            algorithm_iteration_interval=1.0,
            min_num_agents=3,
            max_num_agents=3,
            map_width=50,                
            map_height=50,
            observation_map_size=20,
            max_episode_length=100,
            agent_death_probability=0.0,  # turn off random death for the first test
        )

    env = MappingEnvironment(config)

    collector = AsyncMARLOrchestrator(env, policy_fn=lambda obs: torch.rand(2))
    td = collector.reset()

    for _ in range(100):
        td = collector.step(td)
        if td["next", "done"].item():
            print(f"SIMULATION FINISHED")
        else:
            td = td["next"]

    print(f"Collected {len(collector.agent_transitions)} agent transitions")
    print(f"Collected {len(collector.global_transitions)} global transitions")

    if collector.agent_transitions:
        durations = [t["n_sim_steps"] for t in collector.agent_transitions]
        rewards = [t["reward"] for t in collector.agent_transitions]
        print(f"  agent macro-action durations: "
              f"min={min(durations)}, max={max(durations)}, "
              f"mean={sum(durations)/len(durations):.1f} sim steps")
        print(f"  agent rewards: "
              f"min={min(rewards):.2f}, max={max(rewards):.2f}, "
              f"mean={sum(rewards)/len(rewards):.2f}")

if __name__ == "__main__":
    main()