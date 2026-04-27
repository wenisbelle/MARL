import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

from gradysim.protocol.interface import IProtocol
from gradysim.simulator.extension.extension import Extension

@dataclass
class DroneState:
    """Gathering together all the relevant data from the drone"""
    position: np.ndarray
    velocity: np.ndarray
        
    radius: float = 2.0  
    """ The inflation radius in meters """


@dataclass
class CollisionConfiguration:
    """
    Configuration class for the collision check 
    """     
    time_step: float = 0.1 
    """ From this distance the system will calculate if the drones will colide """
    
    time_horizon: float = 5
    """ Check the collision in this time horizon """


@dataclass
class CollisionResult:
    """
    Store the attributes of the resulting estimative

    Attributes:
        will_collide        : True if the two drones will collide within the horizon
        time_of_collision   : earliest time t >= 0 where spheres first touch (None if no collision)
        time_of_closest     : time of closest approach (always computed, useful for near-misses)
        min_distance        : minimum distance between sphere surfaces at closest approach
    """
    will_collide: bool
    time_of_collision: Optional[float]
    time_of_closest: float
    min_distance: float

class CollisionCheck(Extension):
    def __init__(self, protocol: IProtocol, configuration: CollisionConfiguration):
        super().__init__(protocol)
        self._time_step = configuration.time_step
        self._time_horizon = configuration.time_horizon
        self._epsilon = 1e-3

    def check_collision(self, drone_a: DroneState, drone_b: DroneState) -> CollisionResult:
        """
        Checks whether two drones modelled as spheres will collide within a time horizon,
        assuming constant velocity (straight-line trajectory).

        f(t) = |d(t)|² - R² = a·t² + b·t + c
        where:
        d(t) = (p_a - p_b) + (v_a - v_b) · t 
        R = r_a + r_b (sum of radii)
        a = |Δv|²
        b = 2 · (Δp · Δv)
        c = |Δp|² - R²
        """
        
        dp = drone_a.position - drone_b.position
        dv = drone_a.velocity - drone_b.velocity
        
        r_sum = drone_a.radius + drone_b.radius
        
        a = np.dot(dv, dv)
        b = 2.0 * np.dot(dp, dv)
        c = np.dot(dp, dp) - (r_sum ** 2)
        
        # vertex of the parabola: t = -b / 2a
        if a < self._epsilon:
            # Velocities are identical
            t_closest = 0.0
        else:
            t_closest = max(0.0, min(-b / (2.0 * a), self._time_horizon))
            
        # Minimum distance at the time of closest approach
        dp_closest = dp + (dv * t_closest)
        # Surface-to-surface distance (negative if intersecting)
        min_distance = np.linalg.norm(dp_closest) - r_sum
        
        # Collisions
        if c <= 0:
            # Drones are already overlapping at t = 0
            return CollisionResult(
                will_collide=True, 
                time_of_collision=0.0, 
                time_of_closest=0.0, 
                min_distance=min_distance
            )
            
        delta = (b ** 2) - (4 * a * c)
        
        # If delta is negative, spheres never intersect. 
        # If b > 0, they are moving away from each other.
        if delta >= 0 and b < 0 and a > self._epsilon:
            # Calculate the earliest collision time (smallest positive root)
            t_collision = (-b - math.sqrt(delta)) / (2.0 * a)
            
            if 0 <= t_collision <= self._time_horizon:
                return CollisionResult(
                    will_collide=True,
                    time_of_collision=t_collision,
                    time_of_closest=t_closest,
                    min_distance=min_distance
                )

        # No collision within the horizon
        return CollisionResult(
            will_collide=False,
            time_of_collision=None,
            time_of_closest=t_closest,
            min_distance=min_distance
        )

       