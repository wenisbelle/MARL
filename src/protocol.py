import enum
import json
import logging
from typing import TypedDict
import numpy as np

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from collision_check_plugin import CollisionCheck, CollisionResult, CollisionConfiguration, DroneState
from gradysim.simulator.extension.visualization_controller import VisualizationController


# np.array(self.drone_position).tolist()
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
        self.packet_count = 0
        self.green = [0.0, 255.0, 0.0]
        self.red = [255.0, 0.0, 0.0]
        
        self.current_speed = 5.0
        self.provider.send_mobility_command(SetSpeedMobilityCommand(self.current_speed))

        self.delta_altitude = 5.0
        self.desired_altitude = 10

        if self.provider.get_id() == 0:
            self.target = [50, 0, self.desired_altitude]
            self.velocity_vector = [self.current_speed, 0.0, 0.0]
        else:
            self.target = [-50, 0, self.desired_altitude]
            self.velocity_vector = [-self.current_speed, 0.0, 0.0]
        send_target = GotoCoordsMobilityCommand(*self.target)
        self.provider.send_mobility_command(send_target)

        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 0.25)
        self.provider.schedule_timer("first_paint",self.provider.current_time() + 1.0)

        collision_config = CollisionConfiguration(
            time_step = 0.1,
            time_horizon = 5.0
        )
        self.collision  = CollisionCheck(self, collision_config)

        self.visualization = VisualizationController(self)
    
    def _send_heartbeat(self) -> None:
        #self._log.info(f"UAV {self.provider.get_id()} sending heartbeat")

        message: SimpleMessage = {
            'uav_id': self.provider.get_id(),
            'sender_velocity': self.velocity_vector,
            'sender_position': self.drone_position
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
            # check if the drone is at the right altitude
            if self.drone_position[2] - self.desired_altitude <= 0.01:
                self._log.info(f"Finised the maneuver")
                
                self.current_speed = 5.0
                self.provider.send_mobility_command(SetSpeedMobilityCommand(self.current_speed))

                ## Vector of movement
                diff = np.array(self.target) - np.array(self.drone_position)
                norm = np.linalg.norm(diff)

                if norm > 0:
                    normalized_vector = diff / norm
                else:
                    normalized_vector = diff

                self.velocity_vector = (self.current_speed * normalized_vector).tolist()
                send_target = GotoCoordsMobilityCommand(*self.target)
                self.provider.send_mobility_command(send_target)
            else:
                self.provider.schedule_timer("maneuver", self.provider.current_time() + 0.5)
                self._log.info(f"Doing the maneuver")

    def handle_packet(self, message: str) -> None:
        simple_message: SimpleMessage = json.loads(message)
        #self._log.info(report_message(simple_message))

        other_uav_id = simple_message['uav_id']
        other_uav_velocity = simple_message['sender_velocity']
        other_uav_position = simple_message['sender_position']

        drone_a = DroneState(
            position=np.array(self.drone_position),
            velocity=np.array(self.velocity_vector)
        )

        drone_b = DroneState(
            position = np.array(other_uav_position),
            velocity = np.array(other_uav_velocity) 
        )

        ### Check collision
        results = self.collision.check_collision(drone_a, drone_b)
        self._log.info(f"The uav {other_uav_id} has checked the results. The uavs will collide: {results.will_collide} in the time: {results.time_of_collision}")

        
        if results.will_collide == True:
            self.visualization.paint_node(self.provider.get_id(), self.red)
            

            if self.provider.get_id() < other_uav_id:
                
                self.current_speed = 3.0
                self.provider.send_mobility_command(SetSpeedMobilityCommand(self.current_speed))
                self.desired_altitude = self.drone_position[2] + self.delta_altitude
                new_target  = GotoCoordsMobilityCommand(self.drone_position[0], self.drone_position[1], self.drone_position[2]+ self.delta_altitude)
                self.velocity_vector = [0.0, 0.0, 1.0]
                self.provider.send_mobility_command(new_target)

                self.provider.schedule_timer("maneuver", self.provider.current_time() + 1.0)

        if results.will_collide == False:
            self.visualization.paint_node(self.provider.get_id(), self.green)

        #self._log.info(f"The uav {other_uav_id} has a velocity of: {other_uav_velocity} at the position {other_uav_position}")


    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.drone_position = telemetry.current_position


    def finish(self) -> None:
        pass

