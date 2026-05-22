def make_env():
    from env.mapping_environment import MappingEnvironment, MappingEnvironmentConfig
    return MappingEnvironment(MappingEnvironmentConfig(
        max_num_agents=3, min_num_agents=3, max_episode_length=200,
    ))

def make_policy():
    # whatever your actor architecture is — built fresh inside each worker
    return MyActor(...)

# Main process: buffers live here
agent_buffer  = TensorDictReplayBuffer(storage=LazyTensorStorage(100_000), batch_size=256)
global_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(50_000),  batch_size=64)

# Main process: a "trainer-side" copy of the policy and critic for gradient updates
actor  = make_policy()
critic = MyCritic(...)
optim  = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

with SystemOrchestrator(
    num_workers=4,
    env_fn=make_env,
    policy_fn=make_policy,
    agent_buffer=agent_buffer,
    global_buffer=global_buffer,
    steps_per_batch=200,
) as system:

    # initial weights
    system.broadcast_weights(actor.state_dict())

    for iteration in range(1000):
        # 1. gather
        n_new = system.collect(min_new_transitions=2000, timeout=60.0)
        print(f"iter {iteration}: collected {n_new} agent transitions, "
              f"buffer size {len(agent_buffer)}")

        # 2. train
        if len(agent_buffer) >= 5000:
            for _ in range(50):
                batch = agent_buffer.sample()
                # ... SMDP loss with γ^n_sim_steps as discussed earlier ...
                loss = compute_loss(actor, critic, batch)
                optim.zero_grad(); loss.backward(); optim.step()

            # 3. ship updated weights to all workers
            system.broadcast_weights(actor.state_dict())

for it in range(num_iters):
    system.collect(...)              # collect + pause
    train(...)
    system.set_weights(actor.state_dict())
    system.broadcast()
    system.resume()  



    