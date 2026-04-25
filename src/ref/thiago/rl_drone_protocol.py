"""
rl_drone_protocol.py
--------------------
Drop-in extension of your existing Drone class.

One drone per environment is controlled by the RL agent;
the remaining NUMBER_OF_DRONES-1 drones keep the original fuzzy policy.

Thread-safety model
-------------------
GradySim runs its event loop inside a daemon thread (started by GradySimEnv).
The RL env's step() lives in the Sample Factory worker main thread.
They communicate through two Queue objects and a threading.Event:

  obs_queue  : sim thread → RL thread   (observation, reward, done)
  action_queue: RL thread → sim thread  (np.ndarray waypoint or _STOP sentinel)
  stop_event : RL thread → sim thread   (signals early episode termination)

The sentinel object _STOP is used instead of None so that a legitimate
None action can never be confused with a stop signal.
"""

import queue
import threading
import numpy as np
from typing import Optional, Type

from gradysim.protocol.messages.mobility import (
    GotoCoordsMobilityCommand,
    SetSpeedMobilityCommand,
)

# Import your existing classes exactly as they are — no modifications needed.
from .fuzzy_inteligent_mobility_protocol import Drone, DroneStatus

# ──────────────────────────────────────────────────────────────────────────────
# Sentinel that tells internal_mobility_command to exit without issuing a command
# ──────────────────────────────────────────────────────────────────────────────
_STOP = object()


class RLDrone(Drone):
    """
    Extends Drone so that waypoint decisions come from an external RL agent
    rather than from the FuzzyEvaluator.

    The three class attributes below are injected by rl_drone_protocol_factory()
    before each episode.  Do not instantiate this class directly.
    """

    # Injected per-episode by the factory
    _action_queue: Optional[queue.Queue] = None
    _obs_queue: Optional[queue.Queue] = None
    _stop_event: Optional[threading.Event] = None

    # ── Gymnasium-side constants (kept in sync with GradySimEnv) ──────────────
    # Observation upper bound: uncertainty values can slightly exceed 1.0 after
    # repeated vanishing_map additions, so we use 2.0 as a safe ceiling.
    OBS_LOW: float = -1.0
    OBS_HIGH: float = 2.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        super().initialize()
        # Track previous uncertainty so we can compute a delta reward each step.
        self._prev_uncertainty: float = float(self.MAP_WIDTH * self.MAP_HEIGHT)

    def finish(self) -> None:
        """
        Called by GradySim when the simulation ends (duration elapsed or forced).
        We push one final *terminal* transition so the env's step() can return
        terminated=True and the episode ends cleanly.
        """
        super().finish()
        if self._obs_queue is not None:
            obs = self._build_observation()
            reward = self._compute_step_reward()
            # done=True signals a terminal state to the env
            self._obs_queue.put((obs, reward, True))
        if self._stop_event is not None:
            self._stop_event.set()

    # ── Core override ─────────────────────────────────────────────────────────

    def internal_mobility_command(self) -> None:
        """
        Replaces the fuzzy cell-selection logic.

        1. Package current map state + position into an observation.
        2. Compute the per-step reward (uncertainty reduction).
        3. Push (obs, reward, done=False) to obs_queue so env.step() can return.
        4. Block on action_queue until the RL agent provides the next waypoint.
        5. Denormalise and issue a GotoCoords command.

        If _stop_event is set (env.close() or env.reset() called mid-episode),
        the method exits without issuing a new command and lets the simulation
        wind down naturally.
        """
        # Guard: if episode is being torn down, do nothing.
        if self._stop_event is not None and self._stop_event.is_set():
            return

        obs = self._build_observation()
        reward = self._compute_step_reward()
        self._obs_queue.put((obs, reward, False))

        # ── Wait for action from the RL agent ──────────────────────────────────
        # Use a timeout loop so we remain responsive to stop_event.
        action = None
        while True:
            if self._stop_event is not None and self._stop_event.is_set():
                return  # Episode terminated externally; exit without new command
            try:
                action = self._action_queue.get(timeout=0.05)
                break
            except queue.Empty:
                pass

        # _STOP sentinel: env reset/close unblocked us
        if action is _STOP:
            return

        # ── Denormalise action from [-1, 1] to world coordinates ───────────────
        half_w = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2.0
        half_h = (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2.0

        x_goto = float(np.clip(action[0], -1.0, 1.0)) * half_w
        y_goto = float(np.clip(action[1], -1.0, 1.0)) * half_h
        self.goto_command = np.array([x_goto, y_goto, self.DRONE_ALTITUDE])

        speed = SetSpeedMobilityCommand(self.speed_command)
        self.provider.send_mobility_command(speed)
        command = GotoCoordsMobilityCommand(*self.goto_command)
        self.provider.send_mobility_command(command)

    # ── Observation & reward helpers ──────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """
        Returns a 1-D float32 array:
          [uncertainty_map (W*H values, clipped to [0,1])]
          ++ [norm_x, norm_y, norm_z]   (drone position in [-1,1] / [0,1])

        Total length: MAP_WIDTH * MAP_HEIGHT + 3
        """
        # Uncertainty channel; clip to avoid surprise spikes from vanishing_map
        map_flat = np.clip(self.map[:, :, 0], 0.0, 1.0).flatten().astype(np.float32)

        if self.drone_position is not None:
            half_w = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2.0
            half_h = (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2.0
            pos = np.array(
                [
                    self.drone_position[0] / half_w,   # ∈ [-1, 1]
                    self.drone_position[1] / half_h,   # ∈ [-1, 1]
                    self.drone_position[2] / self.DRONE_ALTITUDE,  # ∈ [0, 1]
                ],
                dtype=np.float32,
            )
        else:
            pos = np.zeros(3, dtype=np.float32)

        return np.concatenate([map_flat, pos])

    def _compute_step_reward(self) -> float:
        """
        Fractional uncertainty reduction since the previous decision step.

        Reward is in [−1, 1].  Positive when the drone (and the swarm it
        shares maps with) reduced uncertainty; slightly negative when the
        vanishing_map routine added more uncertainty than was removed.
        """
        current = float(self.total_uncertainty)
        total_cells = float(self.MAP_WIDTH * self.MAP_HEIGHT)
        reward = (self._prev_uncertainty - current) / total_cells
        self._prev_uncertainty = current
        return reward


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def rl_drone_protocol_factory(
    base_config: dict,
    action_queue: queue.Queue,
    obs_queue: queue.Queue,
    stop_event: threading.Event,
) -> Type[RLDrone]:
    """
    Returns a fresh RLDrone subclass with the per-episode queues injected as
    class attributes.

    A new class is created each episode (same pattern as your existing
    drone_protocol_factory) so there are no stale references across resets.

    Parameters
    ----------
    base_config : dict
        Same _config dict accepted by the parent Drone class.
    action_queue : queue.Queue
        Env puts actions here; RLDrone reads from here.
    obs_queue : queue.Queue
        RLDrone puts (obs, reward, done) tuples here; Env reads from here.
    stop_event : threading.Event
        Set by env.close() / env.reset() to abort the episode gracefully.
    """

    class EpisodeRLDrone(RLDrone):
        _config = base_config
        _action_queue = action_queue
        _obs_queue = obs_queue
        _stop_event = stop_event

    return EpisodeRLDrone


# Re-export the sentinel so GradySimEnv can use it without importing internals.
STOP_SENTINEL = _STOP
