# Independent MADQN

Independent Multi-Agent Deep Q-Network (Independent MADQN) is a **fully decentralized** reinforcement learning algorithm. Each agent maintains its own Q-network and learns independently, treating other agents as part of the environment. There is no centralized critic and no shared gradient computation — only shared network weights (parameter sharing), which lets all agents benefit from each other's experience while retaining the ability to act entirely on their own at deployment time.

## Algorithm Overview

The algorithm is **Double DQN** applied independently to each agent in the swarm. All agents share the same network architecture and the same set of weights (parameter sharing), so any transition collected by any agent contributes to a single training update that improves the policy for all agents.

**Training paradigm:** Decentralized Training, Decentralized Execution (DTDE).

**Key algorithmic choices:**

- Double DQN — decouples action selection from value estimation to reduce overestimation bias.
- Soft target network update (τ = 0.005) — stable training without hard target freezes.
- ε-greedy exploration with exponential decay from 1.0 to 0.2.
- SMDP formulation — transitions are closed at the end of each macro-action (drone reaches destination), not at every simulation step.
- Huber loss with gradient clipping (max norm 1.0) for robustness.

## Network Architecture

The Q-network (`actor.py`) maps a per-agent observation to Q-values over the discrete action space.

```
Observation
  ├── map_patch (M×M)  ─────────────────► MapCNN
  │                                          Conv2d(1→16, k=3, p=1)
  │                                          Conv2d(16→32, k=3, p=1)
  │                                          Conv2d(32→64, k=3, p=1)
  │                                          MaxPool2d → AdaptiveAvgPool2d
  │                                          Flatten → Linear → (128,)
  │
  └── vector features ──────────────────► VectorEncoder
        position (2,)                        Concatenate all vector inputs
        individual_uncertainty (1,)          Linear(in_dim → 64) → ReLU
        estimated_positions_and_time         Linear(64 → 64) → ReLU
          ((N-1)×3,)                         Output: (64,)

Both branches fused: concat(128, 64) = (192,)
  → Linear(192 → 256) → ReLU
  → Linear(256 → 256) → ReLU
  → Linear(256 → action_dim)    ← Q-values, one per cell in the map patch
```

`action_dim = A × A`, where A is the number of cells in each direction that can be selected (default 10, giving 100 actions).

An optional **valid action mask** (provided by the environment) sets the Q-values of out-of-bounds cells to −∞ before the argmax so the agent never selects an unreachable target.

## Observations

Each agent receives a local observation at each decision point:

| Field | Shape | Description |
|---|---|---|
| `map_patch` | `(M, M)` | Local uncertainty map patch centered on drone. Values ∈ [0, 2.0]. Higher = more uncertain. |
| `individual_map_uncertainty` | `(1,)` | Total uncertainty visible to this agent, normalized to [0, 1]. |
| `position` | `(2,)` | Drone's normalized grid position ∈ [0, 1]². |
| `estimated_positions_and_time` | `((N-1)×3,)` | For each other agent: estimated [x, y, time since last contact], all normalized. Obtained via ad hoc communication. |
| `encounter_flag` | `(1,)` | Boolean. True when the drone has reached its macro-action target and a new decision is required. |
| `valid_actions` | `(A×A,)` | Boolean mask. True for cells inside the map boundaries. |

**Global state** (used only for logging and potential CTDE extensions, not fed to the independent policy):

| Field | Shape | Description |
|---|---|---|
| `full_map` | `(W, H)` | Complete uncertainty map merged from all agents. |
| `global_map_uncertainty` | `(1,)` | Sum of all cell uncertainties, normalized. |
| `all_positions` | `(N, 2)` | Positions of all active drones. |
| `all_active` | `(N,)` | Boolean mask of active agents in the episode. |

## Actions

The action space is **discrete** with `A × A` options. Each action index maps to a cell coordinate `(row, col)` within the M×M observation patch. The environment converts the selected cell to a world-space coordinate and issues a `GotoCoords` mobility command to the drone. The drone then flies to that cell; the transition is only closed once the drone arrives (SMDP).

## Rewards

Rewards are accumulated over the entire duration of a macro-action (potentially many simulation steps) before being stored in the replay buffer.

**Per-agent reward:** Reduction in the agent's locally observed map uncertainty caused by visiting the selected cell, normalized by the map width (reward scale = 50). Agents are incentivized to fly to high-uncertainty areas.

**Global reward:** Reduction in global map uncertainty shared equally among all agents. Tracked for monitoring and for CTDE baselines; the independent agents optimize only the per-agent reward.

**Reward mode (`punish`):** Agents receive a negative baseline each step and earn positive credit for uncertainty reduction, creating pressure to act efficiently and avoid redundant coverage.

**Discount factor:** γ = 0.99.

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `MAX_NUM_AGENTS` | 3 | Swarm size |
| `OBSERVATION_MAP_SIZE` (M) | 20 | Gives 400 cells to observe |
| `ACTION_MAP_SIZE` (M) | 10 | Gives 100 discrete actions |
| `MAP_WIDTH / MAP_HEIGHT` | 50 | Grid dimensions |
| `MAX_EPISODE_LENGTH` | 2000 | Simulation steps per episode |
| `N_WORKERS` | 12 | Parallel collection subprocesses |
| `STEPS_PER_BATCH` | 100 | Steps each worker collects per cycle |
| `NUM_ITERATIONS` | 10 000 | Total training iterations |
| `BATCH_SIZE` | 256 | Mini-batch size for Q-update |
| `BUFFERSIZE` | 20 000 | Replay buffer capacity |
| `GAMMA` | 0.99 | Discount factor |
| `LR` | 1e-4 | Adam learning rate |
| `TAU` | 0.005 | Soft target update coefficient |
| `EPS_INIT` | 1.0 | Initial exploration rate |
| `EPS_DECAY` | 0.999 | Per-iteration multiplicative decay |
| `EPS_MIN` | 0.2 | Minimum exploration rate |
| `REWARD_SCALE` | 50 | Normalizes per-agent rewards |
| `VALUE_NETWORK_UPDATES_PER_ITERATION` | 4 | Gradient steps per collect cycle |

## Training Loop

One iteration:

1. **Distribute weights** — latest online Q-network weights are sent to all workers.
2. **Collect** — workers run the environment for `STEPS_PER_BATCH` steps, the orchestrator closes SMDP transitions at `encounter_flag` events, and batched transitions are sent to the replay buffer.
3. **Train (×4 updates):**
   - Sample a mini-batch from the replay buffer.
   - Compute Double DQN target: `y = r + γ · Q_target(s', argmax_a Q_online(s'))·(1 − done)`.
   - Compute Huber loss between `Q_online(s, a)` and `y`.
   - Backward pass, clip gradients, Adam step.
   - Soft update target network: `θ_target ← (1−τ)·θ_target + τ·θ_online`.
4. **Decay epsilon.**
5. **Log** metrics to CSV and console.
6. **Checkpoint** every 500 iterations and whenever a new best average reward is achieved.

## File Reference

| File | Purpose |
|---|---|
| `train.py` | Entry point. Defines hyperparameters and runs the full training loop. |
| `actor.py` | Q-network definition (MapCNN + VectorEncoder + fusion head). |
| `policy.py` | `DQNPolicy` — wraps the Actor with ε-greedy selection and valid-action masking. |
| `replay_buffer.py` | TensorDict circular buffer with random sampling. |
| `simulation_orchestrator.py` | `AsyncMARLOrchestrator` — SMDP transition builder on top of the environment. |
| `worker.py` | Subprocess loop: creates env + policy, collects transitions, responds to control messages. |
| `workers_orchestrator.py` | Pool manager: spawns workers, distributes weights, drains transitions into the replay buffer. |
| `train_logs.py` | Metrics tracking, CSV export, and checkpoint management. |
| `env/` | GrADyS-SIM environment wrapper, observation/action/reward definitions. |
| `checkpoints/` | Saved model snapshots (`.pt` files) and training metrics CSV. |

## Running Training

```bash
cd "src/Independent MADQN"
python3 train.py
```

Logs are printed to stdout and saved to `checkpoints/training_metrics.csv`. The best model is saved to `checkpoints/best.pt`.
