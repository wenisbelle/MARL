# Decentralized Swarm Monitoring with Ad Hoc Communication — MARL

This repository is a research platform for training and evaluating **multi-agent reinforcement learning (MARL)** policies for a cooperative drone swarm performing a **distributed area monitoring task**. All agents operate fully decentralized at inference time: they share no centralized controller and communicate only through a local ad hoc wireless network with limited range.

## Objective

A swarm of UAVs must collectively monitor a 2-D area of interest. The area is discretized into a grid of cells. Each cell accumulates *uncertainty* over time and drones reduce that uncertainty by flying over and sensing the cell. Agents receive only local observations (a small patch of the uncertainty map centered on their position, plus estimates of teammate positions obtained through ad hoc communication) and must learn cooperative coverage policies without any global coordinator.

The key research challenges addressed are:

- **Decentralized execution** — policies run onboard each drone using only locally available information.
- **Ad hoc communication** — agents share state estimates over short-range broadcasts; information is stale and incomplete.
- **Semi-Markov decision process (SMDP)** — agents issue macro-actions (fly to a cell) rather than low-level motor commands. Transitions are closed only when a drone reaches its destination, making episode timing asynchronous across agents.
- **Cooperative reward** — agents are jointly evaluated on global map coverage, creating a cooperative game with independent learners.

## Algorithms Under Study

The repository implements and compares multiple MARL architectures. Each lives in its own directory under `src/`:

| Directory | Algorithm | Training Paradigm |
|---|---|---|
| `Independent MADQN/` | Independent DQN | Fully decentralized (DTDE) |
| `CTDE MADQN/` | DQN with centralized critic | Centralized Training / Decentralized Execution (CTDE) |
| `CTDE MAPPO/` | MAPPO with centralized critic | Centralized Training / Decentralized Execution (CTDE) |

All algorithms share the same simulation environment, observation schema, and system infrastructure so results are directly comparable.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          train.py (main process)                    │
│   Manages weights · runs training loop · saves checkpoints         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
           ┌───────────────▼────────────────┐
           │       WorkersOrchestrator       │
           │  Spawns N worker processes      │
           │  Distributes policy weights     │
           │  Collects transitions           │
           │  Sends PAUSE / CONTINUE signals │
           └───────────────┬────────────────┘
                           │  (one per worker)
           ┌───────────────▼────────────────┐
           │    _worker_loop  (subprocess)   │
           │  ┌──────────────────────────┐  │
           │  │  GrADyS-SIM Environment  │  │
           │  │  3 UAV agents, 50×50 map │  │
           │  └──────────────────────────┘  │
           │  ┌──────────────────────────┐  │
           │  │  AsyncMARLOrchestrator   │  │
           │  │  SMDP transition builder │  │
           │  │  Reward accumulation     │  │
           │  └──────────────────────────┘  │
           │  ┌──────────────────────────┐  │
           │  │  DQNPolicy               │  │
           │  │  ε-greedy action select  │  │
           │  └──────────────────────────┘  │
           └───────────────┬────────────────┘
                           │
           ┌───────────────▼────────────────┐
           │        ReplayBuffer             │
           │  TensorDict-based circular buf  │
           │  Feeds sampled batches to train │
           └───────────────┬────────────────┘
                           │
           ┌───────────────▼────────────────┐
           │       Double DQN Update         │
           │  Online Q-net + Target Q-net    │
           │  Huber loss · gradient clipping │
           │  Soft target update (τ=0.005)  │
           └────────────────────────────────┘
```

### Component Roles

**`simulation_orchestrator.py` — AsyncMARLOrchestrator**
Wraps the GrADyS-SIM environment and builds SMDP-style transitions. Because drones execute macro-actions of variable duration, the orchestrator waits for each agent's `encounter_flag` (raised when the drone reaches its target cell) before closing a transition. It accumulates per-agent and global rewards across all intermediate simulation steps, then emits a complete `(obs, action, reward, next_obs, done)` tuple.

**`workers_orchestrator.py` — WorkersOrchestrator**
Manages the pool of worker subprocesses. In synchronous mode it coordinates a collect → pause → train → broadcast → resume cycle. In asynchronous mode workers run freely while the main process drains their transition queues opportunistically. Weight distribution uses per-worker queues; only the latest weights are kept (stale updates are discarded).

**`worker.py` — _worker_loop**
Runs inside each subprocess. Creates its own environment instance, policy, and orchestrator. Listens for weight updates (non-blocking drain), collects `steps_per_batch` simulation steps, and sends batched transitions back to the main process. Responds to `PAUSE`, `CONTINUE`, and `STOP` control messages.

**`replay_buffer.py` — ReplayBuffer**
TensorDict-based circular experience buffer. Stores transitions from all workers in a shared pool and provides random mini-batch sampling for training.

**`actor.py` — Actor (Q-network)**
The neural network shared by all agents (parameter sharing). See the Independent MADQN README for a detailed architecture description.

**`train_logs.py` — TrainingLogger**
Tracks per-iteration loss, Q-values, TD error, action entropy, and per-episode rewards. Exports a CSV file, maintains rolling statistics, saves the best checkpoint, and saves periodic snapshots.

## Test Folder

The `src/test/` directory contains standalone simulation scripts whose purpose is to validate that a trained policy works **exactly as it would be deployed on real hardware**, i.e., through the native GrADyS-SIM protocol layer rather than through the RL environment wrapper used during training.

This distinction is intentional and important: the RL training loop uses an abstracted `EnvBase` interface for efficiency, but real deployment runs through GrADyS protocol handlers. The test suite closes this gap by loading a trained checkpoint directly into the drone protocol and running full simulations, measuring coverage quality and energy consumption under realistic conditions. See `src/test/Independent MADQN/README.md` for usage details.

## Repository Layout

```
MARL/
├── dockerfile                  # CUDA 12.4 + Ubuntu 22.04 image definition
├── docker-compose.yml          # GPU-enabled container with src/ volume mount
├── src/
│   ├── Independent MADQN/      # DTDE DQN algorithm
│   ├── CTDE MADQN/             # CTDE DQN algorithm
│   ├── CTDE MAPPO/             # CTDE MAPPO algorithm
│   ├── test/
│   │   └── Independent MADQN/  # Protocol-level tests for Independent MADQN
│   └── ref/                    # Reference implementations and baselines
└── avoid_collision/            # Supplementary collision-avoidance extension
```

## Getting Started

### Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2
- NVIDIA GPU with CUDA 12.4-compatible driver
- NVIDIA Container Toolkit (`nvidia-docker2`)

### Build the Image

```bash
git clone <this-repo>
cd MARL
docker build -t marl:latest -f dockerfile .
```

### Run the Container

```bash
docker compose up -d
docker exec marl_container bash
```

The `src/` directory is bind-mounted into the container at `/MARL/src`, so edits on the host are immediately visible inside the container.

### Train a Model

```bash
# Inside the container
cd /MARL/src/Independent_MADQN
python3 train.py
```

Checkpoints are saved to `checkpoints/` every 500 iterations and whenever a new best reward is achieved.

### Run Tests

```bash
cd /MARL/src/test/Independent_MADQN
python3 execution.py
```

See `src/test/Independent_MADQN/README.md` for full details.
