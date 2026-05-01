import enum
import json
import logging
from typing import TypedDict
import numpy as np

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from collision_check_plugin import CollisionCheck, CollisionConfiguration, DroneState
from collision_avoidance import DroneTarget, CollisionAvoidanceConfig, CollisionAvoidance
from gradysim.simulator.extension.visualization_controller import VisualizationController

# ==========================================
# MATCH THIS WITH THE ONE IN MAIN.PY
# ==========================================
TEST_SCENARIO = 3
NUMBER_OF_UAVs = 3

class SimpleMessage(TypedDict):
    uav_id: int
    sender_velocity: list
    sender_position: list

def report_message(message: SimpleMessage) -> str:
    return (f"Received message with {message['uav_id']} packets from ")

class SimpleUAVProtocol(IProtocol):
    _log: logging.Logger
    visualization: VisualizationController
    
    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.green = [0.0, 255.0, 0.0]
        self.red = [255.0, 0.0, 0.0]
        
        self.current_speed = 10.0
        self.provider.send_mobility_command(SetSpeedMobilityCommand(self.current_speed))

        self.desired_altitude = 10
        self.velocity_vector = [0.0, 0.0, 0.0]
        self.target = [0.0, 0.0, 0.0]

        uav_id = self.provider.get_id()

        ### list of bools to store if the UAV will collide with any other UAV of the simulation
        self.will_collide = [False] * NUMBER_OF_UAVs
        self.is_in_maneuver = False
        self._maneuver_active = False
        self._maneuver_target_uav = None
        self._maneuver_eta = None
        self._eta_change_threshold = 0.0

        # --- SCENARIO TARGET ASSIGNMENTS ---
        if TEST_SCENARIO == 0:
            if uav_id == 0:
                self.target = [50, 0, self.desired_altitude]
                self.velocity_vector = [10.0, 0.0, 0.0]
            elif uav_id == 1:
                self.target = [-50, 0, self.desired_altitude]
                self.velocity_vector = [-10.0, 0.0, 0.0]

        elif TEST_SCENARIO == 1:
            if uav_id == 0:
                self.target = [50, 0, self.desired_altitude]
                self.velocity_vector = [10.0, 0.0, 0.0]
            elif uav_id == 1:
                self.target = [0, 50, self.desired_altitude]
                self.velocity_vector = [0.0, 10.0, 0.0]

        elif TEST_SCENARIO == 2:
            if uav_id == 0:
                self.target = [50, 50, self.desired_altitude]
                self.velocity_vector = [7.07, 7.07, 0.0] # 7.07 keeps the vector magnitude at ~10
            elif uav_id == 1:
                self.target = [-50, 50, self.desired_altitude]
                self.velocity_vector = [-7.07, 7.07, 0.0]

        elif TEST_SCENARIO == 3:
            if uav_id == 0: # Trailing
                self.target = [50, 0, self.desired_altitude]
                self.velocity_vector = [10.0, 0.0, 0.0]
            elif uav_id == 1: # Lead
                self.target = [50, 0, self.desired_altitude]
                self.velocity_vector = [10.0, 0.0, 0.0]
            elif uav_id == 2: # Opposite
                self.target = [-50, 0, self.desired_altitude]
                self.velocity_vector = [-10.0, 0.0, 0.0]

        elif TEST_SCENARIO == 4:
            if uav_id == 0:
                self.target = [50, 50, self.desired_altitude]
                self.velocity_vector = [7.07, 7.07, 0.0]
            elif uav_id == 1:
                self.target = [-50, 50, self.desired_altitude]
                self.velocity_vector = [-7.07, 7.07, 0.0]
            elif uav_id == 2:
                self.target = [0, -70.7, self.desired_altitude]
                self.velocity_vector = [0.0, -10.0, 0.0]

        # Send the calculated targets
        send_target = GotoCoordsMobilityCommand(*self.target)
        self.provider.send_mobility_command(send_target)

        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 0.25)
        self.provider.schedule_timer("first_paint",self.provider.current_time() + 1.0)

        collision_config = CollisionConfiguration(
            time_step = 0.1,
            time_horizon = 5.0
        )
        self.collision  = CollisionCheck(self, collision_config)

        collision_avoidance_config = CollisionAvoidanceConfig(
            height_offset = 5.0,
            reduction_scale_higher_priority= 1.0,
            reduction_scale_lower_priority = 10.0
        )
        
        self.collision_avoidance = CollisionAvoidance(self, collision_avoidance_config)

        self._velocities_buffer = []
        self._targets_buffer = []

        self.visualization = VisualizationController(self)
    
    def _send_heartbeat(self) -> None:

        message: SimpleMessage = {
            'uav_id': self.provider.get_id(),
            'sender_velocity': np.array(self.velocity_vector).tolist(),
            'sender_position': np.array(self.drone_position).tolist()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def handle_timer(self, timer: str) -> None:

        if timer == "heartbeat":
            self._send_heartbeat()
            self.provider.schedule_timer("heartbeat", self.provider.current_time() + 0.25)
        
        if timer == "first_paint":
            self.visualization.paint_node(self.provider.get_id(), self.green)

        if timer == "maneuver":
            if self._targets_buffer is not None:
                self.velocity_vector = self._velocities_buffer[0]
                self.current_speed = np.linalg.norm(self.velocity_vector)
                self.provider.send_mobility_command(SetSpeedMobilityCommand(self.current_speed))
                    
                self.target = self._targets_buffer[0]
                send_target = GotoCoordsMobilityCommand(*self.target)
                self.provider.send_mobility_command(send_target)
                
                if len(self._targets_buffer) == 1:
                    # The last value of the target is always the original target. So if the size of the list is one
                    # the maneuver is finished or the drone is actually the one with higher priority
                    self._log.info("Maneuver fully completed.") 
                    self._maneuver_active = False
                    self._maneuver_target_uav = None
                    self._maneuver_eta = None
                else:
                    if np.linalg.norm(self.drone_position - self.target) <= 0.1:
                        # Remove the reached target and its corresponding velocity
                        self._targets_buffer.pop(0)
                        self._velocities_buffer.pop(0)

                        self._log.info("Reached a maneuver waypoint.")

                    else:
                        #self._log.info(f"Doing the maneuver")
                        pass
                    
                    self.provider.schedule_timer("maneuver", self.provider.current_time() + 0.5)
            

    def handle_packet(self, message: str) -> None:
        simple_message: SimpleMessage = json.loads(message)
        other_uav_id = simple_message['uav_id']
        other_uav_velocity = simple_message['sender_velocity']
        other_uav_position = simple_message['sender_position']

        drone_a = DroneState(
            position=np.array(self.drone_position),
            velocity=np.array(self.velocity_vector)
        )
        drone_b = DroneState(
            position=np.array(other_uav_position),
            velocity=np.array(other_uav_velocity)
        )

        results = self.collision.check_collision(drone_a, drone_b)
        self.will_collide[other_uav_id] = results.will_collide

        self._log.info(f"UAV {self.provider.get_id()} will collide with uav {other_uav_id} at the time {results.time_of_collision} ")

        is_priority = self.provider.get_id() >= other_uav_id

        if any(self.will_collide):
            self.visualization.paint_node(self.provider.get_id(), self.red)
        else:
            self.visualization.paint_node(self.provider.get_id(), self.green)

        if not results.will_collide:
            return

        should_plan = False

        if not self._maneuver_active:
            # No maneuver in progress → start one
            should_plan = True
            self._log.info(f"Starting maneuver to avoid UAV {other_uav_id}")

        elif other_uav_id != self._maneuver_target_uav:
            # Different UAV is now a threat → replan only if it's more imminent
            if results.time_of_collision < self._maneuver_eta - self._eta_change_threshold:
                should_plan = True
                self._log.info(
                    f"Switching maneuver target {self._maneuver_target_uav} -> {other_uav_id} "
                    f"(more imminent: {results.time_of_collision:.2f}s vs {self._maneuver_eta:.2f}s)"
                )

        else:
            # Same UAV → replan only if ETA shifted significantly
            if abs(results.time_of_collision - self._maneuver_eta) > self._eta_change_threshold:
                should_plan = True
                self._log.info(
                    f"Refreshing maneuver for UAV {other_uav_id} (ETA changed)"
                )

        if should_plan:
            target_result = self.collision_avoidance.maneuver(
                drone_a, self.target, results.time_of_collision, is_priority
            )
            self._velocities_buffer = target_result.uav_velocity
            self._targets_buffer = target_result.uav_target_position
            self._maneuver_active = True
            self._maneuver_target_uav = other_uav_id
            self._maneuver_eta = results.time_of_collision
            self.provider.schedule_timer("maneuver", self.provider.current_time() + 0.1)


    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.drone_position = telemetry.current_position


    def finish(self) -> None:
        pass

