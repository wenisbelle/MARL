"""
gradysim_env.py
---------------
Gymnasium environment that wraps a full GradySim episode.

Architecture
------------
                    ┌──────────────────────────────────┐
Worker process      │  GradySimEnv (Gymnasium API)      │
(CPU)               │                                  │
                    │  reset() ──► spawns daemon thread │
                    │             running _run_sim()    │
                    │                    │              │
                    │  step(action) ──►  action_queue   │
                    │                    │  (sim thread)│
                    │             obs_queue ◄─ RLDrone  │
                    │                    │              │
                    │  ◄── (obs, rew, done, trunc, {})  │
                    └──────────────────────────────────┘

One drone in each simulation is controlled by the RL agent (RLDrone).
The remaining NUMBER_OF_DRONES-1 drones use your original fuzzy policy so
the agent learns in a realistic cooperative context from day one.

Observation space
-----------------
  Shape : (MAP_WIDTH * MAP_HEIGHT + 3,)  float32
  Meaning: flattened uncertainty map (values ∈ [0,1]) followed by the RL
           drone's normalised (x, y, z) position.

Action space
------------
  Shape : (2,)  float32  ∈ [-1, 1]
  Meaning: desired waypoint in normalised map coordinates.
           x = 1 → east edge,  x = -1 → west edge
           y = 1 → north edge, y = -1 → south edge
           Altitude is fixed at DRONE_ALTITUDE (50 m).

Reward
------
  Fractional uncertainty reduction per decision step (see RLDrone._compute_step_reward).
  Values are roughly in [-0.05, +0.10] per step.

Episode termination
-------------------
  terminated=True  when GradySim reaches its configured duration (or all RL
                   drones run out of battery).
  truncated=False  (Sample Factory handles wall-clock truncation separately
                   via rollout length; we never truncate here).
"""

import queue
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.communication import (
    CommunicationHandler,
    CommunicationMedium,
)
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder

from .rl_drone_protocol import rl_drone_protocol_factory, STOP_SENTINEL
from .fuzzy_inteligent_mobility_protocol import Drone, drone_protocol_factory
from .lookup_table_generator import FuzzyLookupTable

# ──────────────────────────────────────────────────────────────────────────────
# Default fuzzy parameters for background drones
# These are the hand-tuned starting values from your GA baseline.
# Swap in your best GA individual once you have one.
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_FUZZY_PARAMS = np.array(
    [
        # uncertainty interval (3 floats, sum ≤ 2)
        0.25, 0.25, 0.25,
        # distance interval (3 floats, sum ≤ 300)
        80.0, 80.0, 80.0,
        # one-cell priority interval (3 floats, sum ≤ 1)
        0.25, 0.25, 0.25,
        # sum-of-priorities interval (3 floats, sum ≤ 2)
        0.50, 0.50, 0.50,
        # distance-between-targets interval (3 floats, sum ≤ 300)
        40.0, 40.0, 40.0,
        # two-cell priority interval (3 floats, sum ≤ 1)
        0.25, 0.25, 0.25,
        # rules (18 ints, each ∈ {0,1,2,3,4})
        2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2,
    ],
    dtype=np.float64,
)


class GradySimEnv(gymnasium.Env):
    """
    Gymnasium-compatible environment for GradySim drone coverage.

    Parameters
    ----------
    map_width, map_height : int
        Grid dimensions.  Smaller grids train faster; your GA used 50×50
        but 10×10 is a good starting point for RL (obs vector = 103 floats).
    number_of_drones : int
        Total drones.  Exactly one is RL-controlled; the rest use fuzzy logic.
    simulation_duration : int
        Simulated seconds per episode.  1000 matches your GA setup.
    fuzzy_params : np.ndarray or None
        Parameters for the background fuzzy drones.  Pass your best GA
        individual here once available.
    communication_range : float
        Transmission range (metres) for inter-drone communication.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        map_width: int = 10,
        map_height: int = 10,
        number_of_drones: int = 3,
        simulation_duration: int = 1000,
        fuzzy_params: Optional[np.ndarray] = None,
        communication_range: float = 200.0,
    ) -> None:
        super().__init__()

        self.MAP_WIDTH = map_width
        self.MAP_HEIGHT = map_height
        self.NUMBER_OF_DRONES = number_of_drones
        self.SIMULATION_DURATION = simulation_duration
        self.COMMUNICATION_RANGE = communication_range
        self.fuzzy_params: np.ndarray = (
            fuzzy_params if fuzzy_params is not None else _DEFAULT_FUZZY_PARAMS.copy()
        )

        # ── Spaces ────────────────────────────────────────────────────────────
        obs_dim = map_width * map_height + 3
        # Uncertainty values can slightly exceed 1 after vanishing_map additions;
        # position components are in [-1, 1].  We use [-1, 2] as a safe box.
        self.observation_space = gymnasium.spaces.Box(
            low=np.full(obs_dim, -1.0, dtype=np.float32),
            high=np.full(obs_dim, 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        # Normalised waypoint: x ∈ [-1,1], y ∈ [-1,1]
        self.action_space = gymnasium.spaces.Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Threading primitives (replaced fresh every episode) ───────────────
        self._sim_thread: Optional[threading.Thread] = None
        self._action_queue: queue.Queue = queue.Queue(maxsize=1)
        self._obs_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event: threading.Event = threading.Event()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Cleanly abort any in-progress simulation from the previous episode.
        self._teardown_episode()

        # Fresh primitives — no stale messages from last episode.
        self._action_queue = queue.Queue(maxsize=1)
        self._obs_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        # Reset the shared class counter so episodes are independent.
        # (Each Sample Factory worker is a separate process, so this only
        # matters for the multiple episodes run within one worker.)
        Drone.Number_of_Encounters = 0

        # Launch simulation in background daemon thread.
        self._sim_thread = threading.Thread(
            target=self._run_simulation,
            name="gradysim-episode",
            daemon=True,
        )
        self._sim_thread.start()

        # Block until the first mobility decision produces an observation.
        obs, _reward, _done = self._obs_queue.get()
        return obs.astype(np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Deliver the agent's chosen waypoint to the simulation thread.
        self._action_queue.put(action)

        # Wait for the simulation to reach the next decision point (or end).
        obs, reward, done = self._obs_queue.get()

        terminated: bool = bool(done)
        truncated: bool = False
        info: Dict[str, Any] = {}

        if terminated:
            # Let the sim thread finish its cleanup before the next reset().
            self._sim_thread.join(timeout=5.0)

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._teardown_episode()
        super().close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _teardown_episode(self) -> None:
        """
        Gracefully stop any simulation thread that is still running.

        Strategy:
          1. Set stop_event  → RLDrone's timeout loop sees it and exits.
          2. Put STOP_SENTINEL in action_queue → unblocks a waiting get().
          3. Join the thread with a generous timeout.
        """
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self._stop_event.set()
            try:
                # Unblock the sim thread if it is waiting for an action.
                self._action_queue.put_nowait(STOP_SENTINEL)
            except queue.Full:
                pass
            self._sim_thread.join(timeout=5.0)
            if self._sim_thread.is_alive():
                # This should not happen in normal operation.
                import warnings
                warnings.warn(
                    "GradySim simulation thread did not terminate within 5 s. "
                    "The episode may not have cleaned up properly.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._sim_thread = None

    def _run_simulation(self) -> None:
        """
        Build and execute one complete GradySim episode.

        Runs inside a daemon thread.  All GradySim state is local to this
        function — no shared mutable state with the main thread.
        """
        try:
            # ── Fuzzy tables for background drones ────────────────────────────
            fuzzy_lookup = FuzzyLookupTable(fuzzy_parameters=self.fuzzy_params)
            lookup_one_cell, lookup_two_cells = fuzzy_lookup.get_interpolators()

            fuzzy_tables = [lookup_one_cell, lookup_two_cells]

            # ── Shared config dict ────────────────────────────────────────────
            base_config = {
                "uncertainty_rate": 0.01,
                "vanishing_update_time": 10.0,
                "number_of_drones": self.NUMBER_OF_DRONES,
                "map_width": self.MAP_WIDTH,
                "map_height": self.MAP_HEIGHT,
                "fuzzy_tables": fuzzy_tables,
                "results_aggregator": {},  # populated by Drone.finish()
            }

            # ── Protocol classes ──────────────────────────────────────────────
            # Node 0 → RL-controlled drone
            RLDroneClass = rl_drone_protocol_factory(
                base_config=base_config,
                action_queue=self._action_queue,
                obs_queue=self._obs_queue,
                stop_event=self._stop_event,
            )

            # Nodes 1 … N-1 → original fuzzy-logic drones
            # Each gets its own results_aggregator slot so finish() works correctly.
            FuzzyDroneClass = drone_protocol_factory(
                uncertainty_rate=0.01,
                vanishing_update_time=10.0,
                number_of_drones=self.NUMBER_OF_DRONES,
                map_width=self.MAP_WIDTH,
                map_height=self.MAP_HEIGHT,
                fuzzy_tables=fuzzy_tables,
                results_aggregator={},
            )

            # ── Build simulation ──────────────────────────────────────────────
            config = SimulationConfiguration(
                duration=self.SIMULATION_DURATION,
                real_time=False,
            )
            builder = SimulationBuilder(config)
            builder.add_handler(TimerHandler())
            builder.add_handler(MobilityHandler())
            builder.add_handler(
                CommunicationHandler(
                    CommunicationMedium(transmission_range=self.COMMUNICATION_RANGE)
                )
            )

            builder.add_node(RLDroneClass, (0, 0, 0))
            for _ in range(self.NUMBER_OF_DRONES - 1):
                builder.add_node(FuzzyDroneClass, (0, 0, 0))

            simulation = builder.build()
            simulation.start_simulation()
            # After start_simulation() returns, finish() has been called on all
            # nodes.  RLDrone.finish() already pushed the terminal transition.

        except Exception:
            # If the simulation crashes unexpectedly, push a terminal dummy obs
            # so env.step() does not hang forever.
            import traceback
            traceback.print_exc()
            dummy_obs = np.zeros(
                self.MAP_WIDTH * self.MAP_HEIGHT + 3, dtype=np.float32
            )
            try:
                self._obs_queue.put_nowait((dummy_obs, 0.0, True))
            except queue.Full:
                pass
