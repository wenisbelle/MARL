from dataclasses import dataclass
from typing import Optional
import numpy as np

from gradysim.protocol.interface import IProtocol
from gradysim.simulator.extension.extension import Extension
from collision_check_plugin import DroneState

### Preciso que ele esteja a dois metros de altura no momento previsto do impacto

@dataclass
class DroneTarget:
    """Gathering together all the relevant output both uavs"""
    
    uav_velocity: list[np.ndarray]
    """The list of velocities throughtout the movement of the uav"""

    uav_target_position: list[np.ndarray]
    """The target positions the uav will try to reach"""


from dataclasses import dataclass
from typing import Optional

@dataclass
class CollisionAvoidanceConfig:
    """
    Attributes:
        height_offset       : vertical distance between the two UAVs at the time of predicted collision
        reduced_speed       : the reduced absolute velocity of the UAVs when doing this maneuver,
                              for both uavs, higher and lower priority
        reduction_scale     : the reduction scale of the original velocity that will be applied,
                              for both uavs, higher and lower priority

        Exactly one of the last two must be given for each priority level.
    """
    height_offset: float
    
    reduced_speed_higher_priority: Optional[float] = None
    reduction_scale_higher_priority: Optional[float] = None

    reduced_speed_lower_priority: Optional[float] = None
    reduction_scale_lower_priority: Optional[float] = None

    def __post_init__(self):
        if self.reduced_speed_higher_priority is None and self.reduction_scale_higher_priority is None:
            raise ValueError("Must provide either reduced_speed or reduction_scale for higher priority.")
            
        if self.reduced_speed_higher_priority is not None and self.reduction_scale_higher_priority is not None:
            raise ValueError("Cannot provide BOTH reduced_speed and reduction_scale for higher priority. Choose exactly one.")
            
        if self.reduced_speed_lower_priority is None and self.reduction_scale_lower_priority is None:
            raise ValueError("Must provide either reduced_speed or reduction_scale for lower priority.")
            
        if self.reduced_speed_lower_priority is not None and self.reduction_scale_lower_priority is not None:
            raise ValueError("Cannot provide BOTH reduced_speed and reduction_scale for lower priority. Choose exactly one.")

class CollisionAvoidance(Extension):
    
    
    def __init__(self, protocol: IProtocol, configuration: CollisionAvoidanceConfig):
        super().__init__(protocol)

        self._height_offset = configuration.height_offset
        self._reduced_speed_higher_priority = configuration.reduced_speed_higher_priority
        self._reduction_scale_higher_priority = configuration.reduction_scale_higher_priority
        self._reduced_speed_lower_priority = configuration.reduced_speed_lower_priority
        self._reduction_scale_lower_priority = configuration.reduction_scale_higher_priority
        

    def change_higher_priority_velocity(self, reduced_speed_higher_priority: Optional[float] = None, reduction_scale_higher_priority: Optional[float] = None) -> bool:
        if reduced_speed_higher_priority is None and reduction_scale_higher_priority is None:
            print("Must provide at least one new value.")
            return False

        if reduced_speed_higher_priority is not None and reduction_scale_higher_priority is not None:
            print("Cannot provide both speed and scale. Must choose one.")
            return False

        # Safely update one and clear the other
        if reduced_speed_higher_priority is not None:
            self._reduced_speed_higher_priority = reduced_speed_higher_priority
            self._reduction_scale_higher_priority = None 

        elif reduction_scale_higher_priority is not None:
            self._reduction_scale_higher_priority = reduction_scale_higher_priority
            self._reduced_speed_higher_priority = None  

        return True

    def change_lower_priority_velocity(self, reduced_speed_lower_priority: Optional[float] = None, reduction_scale_lower_priority: Optional[float] = None) -> bool:
        if reduced_speed_lower_priority is None and reduction_scale_lower_priority is None:
            print("Must provide at least one new value.")
            return False

        if reduced_speed_lower_priority is not None and reduction_scale_lower_priority is not None:
            print("Cannot provide both speed and scale. Must choose one.")
            return False

        # Safely update one and clear the other
        if reduced_speed_lower_priority is not None:
            self._reduced_speed_lower_priority = reduced_speed_lower_priority
            self._reduction_scale_lower_priority = None 

        elif reduction_scale_lower_priority is not None:
            self._reduction_scale_lower_priority = reduction_scale_lower_priority
            self._reduced_speed_lower_priority = None  

        return True
    
    def maneuver(self, drone: DroneState, current_target: np.array,
                       estimated_time_of_collision: float, is_higher_priority: bool) -> DroneTarget:
        """
        Apply a very simple maneuver when two drones are going to collide. 
        The UAV with higher priority will keep its trajectory but can reduce its speed to increase the safety.
        While the UAV with lower priority will increase its altitude to avoid the collision.
        """
        list_of_positions = []
        list_of_velocities = []
        
        if is_higher_priority is True:           
            if self._reduced_speed_higher_priority is not None:
                velocity = self._reduced_speed_higher_priority*(drone.velocity/np.linalg.norm(drone.velocity)) 
            if self._reduction_scale_higher_priority is not None:
                velocity = self._reduction_scale_higher_priority*drone.velocity
            
            list_of_positions.append(current_target)
            list_of_velocities.append(velocity)
            
            return DroneTarget(
                uav_velocity=list_of_velocities,
                uav_target_position =list_of_positions
            )
        
        else:
            ### Now it needs to perform the manuever

            ##### Velocities
            velocity_norm = np.linalg.norm(drone.velocity)
            if self._reduced_speed_lower_priority is not None:
                velocity_norm = self._reduced_speed_lower_priority
            elif self._reduction_scale_lower_priority is not None:
                velocity_norm = self._reduction_scale_lower_priority * np.linalg.norm(drone.velocity)

            ### Calculate the required Z velocity to reach the target in time
            vz = self._height_offset / (estimated_time_of_collision) 

            ##### Positions
            ## It has to arrive at the desired altitude at the time of collision. More maneuvers could be applied here in the future
            delta_xy_position = (drone.velocity * estimated_time_of_collision)

            # Ascendent part of the maneuver
            ascendent_position = drone.position + [delta_xy_position[0], delta_xy_position[1], self._height_offset]
            list_of_positions.append(ascendent_position)

            # Descendent part of the maneuver
            descendent_position = drone.position + [2*delta_xy_position[0], 2*delta_xy_position[1], 0]
            list_of_positions.append(descendent_position)

            # Now the original target
            list_of_positions.append(current_target) 

            #### Velocities
            velocity_xy = delta_xy_position/estimated_time_of_collision
            ascendent_velocity = [velocity_xy[0], velocity_xy[1], vz]
            descendent_velocity = [velocity_xy[0], velocity_xy[1], -vz]

            list_of_velocities.append(ascendent_velocity)
            list_of_velocities.append(descendent_velocity)
            # go back to the original velocity
            list_of_velocities.append(drone.velocity)  

            
            return DroneTarget(
                uav_velocity=list_of_velocities,
                uav_target_position= list_of_positions
            )

        
        

       