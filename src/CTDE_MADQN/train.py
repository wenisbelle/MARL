"""
main_test_workers.py — smoke test for the patched worker / WorkersOrchestrator.

Sync lifecycle (per iteration):
    set_weights -> broadcast -> resume -> collect (returns with workers PAUSED)
    -> train -> next iteration

Workers are bootstrap-paused on startup, so iteration 0 follows the same
pattern as every other iteration.
"""
import os
from collections import deque
import multiprocessing as mp
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
import time
import torch.nn.functional as F

from env.mapping_environment import MappingEnvironment, MappingEnvironmentConfig
from workers_orchestrator import WorkersOrchestrator
from replay_buffer import ReplayBuffer
from actor import Actor
from policy import RandomPolicy, DQNPolicy
from train_logs import TrainingLogger

ALGORITHM_ITERATION_INTERVAL = 1.0
MAX_NUM_AGENTS = 3
MIN_NUM_AGENTS = 3   
MAP_WIDTH = 50
MAP_HEIGHT = 50
OBSERVATION_MAP_SIZE = 50
ACTION_MAP_SIZE = 10
MAX_EPISODE_LENGTH = 2000
AGENT_DEATH_PROBABILITY = 0.0
MAP_CHANNELS = 2
VECTOR_FEATURE_DIM = 64
HIDDEN_DIM = 256
MAP_KEY = "map_patch"
POSITION_KEY = "position"
UNCERTAINTY_KEY = "individual_map_uncertainty"
ESTIMATED_POSITIONS_KEY = "estimated_positions_and_time"
EPS_INIT = 1.0
EPS_DECAY = 0.9998
EPS_MIN = 0.1
N_WORKERS = 24
STEPS_PER_BATCH = 100
NUM_ITERATIONS = 20000
MIN_TRANSITIONS_PER_COLLECT = 250
COLLECT_TIMEOUT_S = 10.0
SYNC = False
NEW_BATCH_NEW_SIMULATION = False
TRAIN_FREQUENCY = 4

BATCH_SIZE = 256
BUFFERSIZE = 30000
VALUE_NETWORK_UPDATES_PER_ITERATION = 4

GAMMA        = 0.99
REWARD_DECAY = 0.999
LR = 1e-4
TARGET_SYNC = 10 
GRAD_CLIP = 1.0
TAU = 0.005 

CHECKPOINT_EVERY    = 500              # save every K iterations (set ≤ NUM_ITERATIONS)
CHECKPOINT_DIR      = "checkpoints"
LOG_EVERY           = 1                # print every iter; raise for long runs
REWARD_WINDOW       = 100               # rolling-average window for "is it improving?"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Rolling histories — windowed so noisy single-iter values don't fool you.
reward_history     = deque(maxlen=REWARD_WINDOW)
global_reward_hist = deque(maxlen=REWARD_WINDOW)
best_avg_reward    = float("-inf")

def make_env():
    config = MappingEnvironmentConfig(
        render_mode= None, #"visual",
        algorithm_iteration_interval=ALGORITHM_ITERATION_INTERVAL,
        min_num_agents=MIN_NUM_AGENTS,
        max_num_agents=MAX_NUM_AGENTS,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        observation_map_size=OBSERVATION_MAP_SIZE,
        action_map_size=ACTION_MAP_SIZE,
        max_episode_length=MAX_EPISODE_LENGTH,
        agent_death_probability=AGENT_DEATH_PROBABILITY,
    )
    return MappingEnvironment(config)

def make_policy():
    return DQNPolicy(
        max_num_agents=MAX_NUM_AGENTS,
        action_dim= ACTION_MAP_SIZE * ACTION_MAP_SIZE,  
        map_channels=MAP_CHANNELS,
        vector_feature_dim=VECTOR_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        map_key=MAP_KEY,
        position_key=POSITION_KEY,
        uncertainty_key=UNCERTAINTY_KEY,
        estimated_positions_key=ESTIMATED_POSITIONS_KEY,
        eps_init=EPS_INIT,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    replay_buffer = ReplayBuffer(buffer_size=BUFFERSIZE, batch_size=BATCH_SIZE, centralized_training=False)
    trainer_policy = make_policy().to(device)
    optimizer = torch.optim.Adam(trainer_policy.actor.parameters(), lr=LR)

    logger = TrainingLogger(
        checkpoint_dir=CHECKPOINT_DIR,
        reward_window=REWARD_WINDOW,
        checkpoint_every=CHECKPOINT_EVERY,
        live_plot=False,
        plot_every=1,
        )

    target_actor = Actor(
        max_num_agents=MAX_NUM_AGENTS,
        action_dim=ACTION_MAP_SIZE * ACTION_MAP_SIZE,
        map_channels=MAP_CHANNELS,
        vector_feature_dim=VECTOR_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        map_key=MAP_KEY,
        position_key=POSITION_KEY,
        uncertainty_key=UNCERTAINTY_KEY,
        estimated_positions_key=ESTIMATED_POSITIONS_KEY,
        ).to(device)
    target_actor.load_state_dict(trainer_policy.actor.state_dict())
    
    for p in target_actor.parameters():
        p.requires_grad_(False)
    target_actor.eval()

    with WorkersOrchestrator(
        num_workers=N_WORKERS,
        env_fn=make_env,
        policy_fn=make_policy,
        replay_buffer=replay_buffer,
        steps_per_batch=STEPS_PER_BATCH,
        base_seed=42,
        sync=SYNC,
        reward_scale = MAP_WIDTH/5,
        reward_decay = REWARD_DECAY,
        new_batch_new_simulation=NEW_BATCH_NEW_SIMULATION,
    ) as orch:

        for it in range(NUM_ITERATIONS):
            # Same four-step rhythm on every iteration, including the first.
            orch.set_weights(trainer_policy.state_dict())
            orch.broadcast()
            orch.resume()
            new_count = orch.collect(
                min_new_transitions=MIN_TRANSITIONS_PER_COLLECT,
                timeout=COLLECT_TIMEOUT_S,
            )

            # Print for debbuging
            #print(
            #    f"[iter {it}] new agent transitions={new_count}  "
            #    f"agent_buffer={len(replay_buffer)}  "
            #)
            if SYNC:
                # stop the execution for 100 ms
                # This is necessary to give time to worker to stop before starting a new bunch of step iterations
                time.sleep(0.1)
            

            batch = None
            if len(replay_buffer) >= BATCH_SIZE:
                for _ in range(VALUE_NETWORK_UPDATES_PER_ITERATION):
                    batch = replay_buffer.sample()

                    obs = batch["obs"].to(device)
                    next_obs = batch["next_obs"].to(device)
                    action = batch["action"].to(device)
                    reward = batch["reward"].float().to(device)
                    done = batch["done"].float().to(device)

                    with torch.no_grad():
                        #### Double DQN target calculation
                        next_q_value_online = trainer_policy.actor(next_obs)       # (B, action_dim)
                        a_next = next_q_value_online.argmax(dim=-1, keepdim=True)  # (B, 1)
                        next_q_target = target_actor(next_obs).gather(-1, a_next) #  B, 1)

                        y = reward + GAMMA * next_q_target * (1.0 - done) 

                    q_pred_all = trainer_policy.actor(obs)  # (B, action_dim)
                    q_pred = q_pred_all.gather(-1, action)  # (B, 1)

                    # SmoothL1 (Huber) is the DQN convention — more robust to outliers than MSE
                    loss = F.smooth_l1_loss(q_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    # For stability:
                    torch.nn.utils.clip_grad_norm_(trainer_policy.actor.parameters(), max_norm=GRAD_CLIP)
                    optimizer.step()

                    # soft update
                    for pt, po in zip(target_actor.parameters(), trainer_policy.actor.parameters()):
                        pt.data.mul_(1 - TAU).add_(TAU * po.data)

                    logger.log_update(
                        loss=loss.item(),
                        q_values_all=q_pred_all,
                        td_error=(y - q_pred).mean().item(),
                    )

                # Update EPS
                trainer_policy.update_epsilon(max(trainer_policy.eps.item() * EPS_DECAY, EPS_MIN))                #if (it + 1) % TARGET_SYNC == 0:
                #    target_actor.load_state_dict(trainer_policy.actor.state_dict())

                logger.log_iteration(
                    it=it,
                    new_count=new_count,
                    buffer_size=len(replay_buffer),
                    eps=trainer_policy.eps,
                    reward_sample=batch["reward"] if batch is not None else None
                    )

                logger.maybe_checkpoint(it, trainer_policy.actor, target_actor, optimizer,
                                         eps=trainer_policy.eps)
    logger.close()
                        
         


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
