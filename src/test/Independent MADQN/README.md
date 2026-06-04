# Test — Independent MADQN

This directory contains standalone simulation scripts that evaluate a trained Independent MADQN policy in the same execution context it would be used in a real deployment, i.e., running directly through the **GrADyS protocol layer** rather than through the RL training environment wrapper.

## Why a Separate Test Folder

During training the policy is exposed to an abstracted `EnvBase` interface (`src/Independent MADQN/env/`) that is optimized for throughput: it steps the simulator in bulk, returns TensorDict observations, and resets quickly. This interface does not reflect how the policy would run on an actual swarm.

The test scripts here bypass that abstraction entirely. The trained neural network is loaded directly into a `Drone` GrADyS protocol object. The simulation runs the full handler stack — `TimerHandler`, `MobilityHandler`, `CommunicationHandler` — exactly as a real GrADyS deployment would. This means:

- **Timing is realistic** — communication events, mobility callbacks, and timer firings happen in the same order and at the same rates as on hardware.
- **Ad hoc communication is real** — messages are broadcast over a range-limited medium and received asynchronously.
- **No RL scaffolding** — there is no `encounter_flag`, no `step()` wrapper, no centralized reward signal. The drone protocol drives its own decision loop.

If a policy performs well in training but fails in this test, the gap points to a mismatch between the training abstraction and the deployment reality. Passing both is the definition of a deployable policy.

## File Overview

| File | Purpose |
|---|---|
| `execution.py` | Main entry point. Builds and runs a complete GrADyS-SIM simulation with trained drones. |
| `protocol.py` | `Drone` GrADyS protocol. Loads the Actor checkpoint, runs ε-greedy inference, handles communication messages and mobility callbacks. |
| `actor.py` | Copy of the Q-network architecture. Must match the checkpoint being loaded. |
| `energy.py` | Energy consumption model (`EnergyConsumption`, `BatteryError`). Tracks battery usage during a test run. |
| `fitness.py` | Computes coverage quality metrics (e.g., fraction of cells visited, average uncertainty at end of episode). |
| `visualization.py` | `MapVisualizer` — optional ASCII/matplotlib map rendering for debugging. |
| `plots.py` | Generates comparison plots from logged simulation results. |
| `logs/` | Output directory for per-run simulation logs (`simulation.log`). |

## How to Use

### 1. Train a Model

First produce a checkpoint from the training loop:

```bash
cd "src/Independent MADQN"
python3 train.py
```

The best checkpoint is saved to `src/Independent MADQN/checkpoints/best.pt`.

### 2. Set the Checkpoint Path

In `protocol.py`, the `Drone` class loads the actor weights from a path defined near the top of the file. Update it to point to the checkpoint you want to evaluate:

```python
CHECKPOINT_PATH = "../Independent MADQN/checkpoints/best.pt"
```

### 3. Run the Simulation

```bash
cd "src/test/Independent MADQN"
python3 execution.py
```

By default `execution.py` runs **20 independent episodes** and writes per-episode logs to `logs/simulation.log`. Aggregate results (coverage fraction, uncertainty at termination, energy consumed) are printed to stdout at the end.

### 4. Visualize Results

```bash
python3 plots.py
```

Generates comparison plots saved as PNG files in the current directory (e.g., `swarm_comparison_performance_MARL.png`).

## Simulation Configuration

Key parameters in `execution.py` that control the test run:

| Parameter | Default | Description |
|---|---|---|
| `NUMBER_OF_DRONES` | 3 | Swarm size. Must match training config. |
| `MAP_WIDTH / MAP_HEIGHT` | 50 | Grid dimensions. Must match training config. |
| `observation_map_size` | 10 | Local patch size. Must match the Actor's `action_dim` = M². |
| `uncertainty_rate` | 0.01 | Rate at which unvisited cells grow uncertain. |
| `vanishing_update_time` | 10.0 | Time (s) before a communicated position estimate becomes stale. |
| `CommunicationMedium.transmission_range` | 200 | Ad hoc broadcast range in meters. |
| `SimulationConfiguration.duration` | 2000 | Episode length in simulation seconds. |
| `N_EPISODES` | 20 | Number of independent runs. |

## Drone Protocol Internals (`protocol.py`)

The `Drone` class implements `IProtocol` from the GrADyS SDK. Its decision loop mirrors the RL environment exactly, but runs natively:

1. **Initialization** — loads the Actor checkpoint, initializes the local uncertainty map, sets up the camera hardware.
2. **Periodic timer** — fires at each timestep, updates the local map using camera observations, increments uncertainty for unseen cells.
3. **Communication receive** — processes `HEARTBEAT`, `SHARE_STATE`, and `BROADCAST_DESTINATION` messages from neighbors; updates estimated teammate positions and map patches received via ad hoc broadcast.
4. **Decision** — when the mobility controller signals arrival at the current target (equivalent to the training `encounter_flag`), the drone:
   - Builds a TensorDict observation matching the training schema exactly.
   - Runs a forward pass through the Actor to obtain Q-values.
   - Applies the valid-action mask.
   - Selects the highest-Q valid cell (greedy; ε can be set to 0 for pure evaluation).
   - Issues a `GotoCoords` command to fly to the selected cell.
5. **State broadcast** — periodically broadcasts the drone's position and local map patch to neighbors within communication range.

## Matching Training and Test Configurations

The test is only meaningful if the observation schema, action space, and map dimensions match exactly what the policy was trained on. Any mismatch will produce garbage Q-values. Verify the following are identical between `src/Independent MADQN/train.py` and `execution.py` before running:

- `observation_map_size` (M)
- `MAP_WIDTH` / `MAP_HEIGHT`
- `MAX_NUM_AGENTS` / `NUMBER_OF_DRONES`
- `drone_altitude`
- `distance_between_cells`
