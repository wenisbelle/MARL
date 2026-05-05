import enum
import math
import logging
from typing import TypedDict, Type, Optional
import numpy as np
from dataclasses import dataclass
import json
import random
from .fitness import FitnessEvaluator
from .energy import EnergyComsuption, BatteryError

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from gradysim.simulator.extension.camera import CameraHardware, CameraConfiguration
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand


@dataclass
class Threat:
    level: int
    position_id: int
    timestamp: float 

class DroneStatus(enum.Enum):
    MAPPING = 0
    RECRUITING = 1
    ENGAGING = 2
    DEAD = 3

class MessageType(enum.Enum):
    HEARTBEAT_MESSAGE = 0
    SHARE_MAP_MESSAGE = 1
    SHARE_GOTO_POSITION_MESSAGE = 2
    RECRUITING_MESSAGE = 3

class HeartBeatMessage(TypedDict):
    message_type: int
    status: int
    sender: int

class ShareMapMessage(TypedDict):
    message_type: int 
    map: list
    sender: int
    drone_position: list

class SendGoToMessage(TypedDict):
    message_type: int 
    goto: list
    sender: int
    priority_value: float

class FlagMessage(enum.Enum):
    NONE = 0
    INTERNAL_MOBILITY_COMMAND = 1
    EXTERNAL_MOBILITY_COMMAND_HIGHER_PRIORITY = 2
    EXTERNAL_MOBILITY_COMMAND_LOWER_PRIORITY = 3

class BufferedMobilityCommand(TypedDict):
    flag: int
    other_uav_id: Optional[int] ### only when the flag is  EXTERNAL_MOBILITY_COMMAND_LOWER_PRIORITY. In the other cases, it will be None.
    partner_position: Optional[np.array] ### only when the flag is  EXTERNAL_MOBILITY_COMMAND. In the other cases, it will be None.


class MovementDirection(enum.Enum):
    X = 0
    Z = 1


class Drone(IProtocol):
    ### Starting plugins ###
    camera: CameraHardware
    _log: logging.Logger
    ### Variable to track how many interactions happened ###
    Number_of_Encounters: int = 0

    ##### Configuration for drone inheritance #####
    _config = {
        "uncertainty_rate": 0.01,
        "vanishing_update_time": 10.0,
        "number_of_drones": 3,
        "map_width": 10,
        "map_height": 10,
    }

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.drone_position = None
        self.goto_command = np.zeros(3)

        self.TIMEOUT_TO_UPDATE_DESTINATION = 10.0

        self.UNCERTAINTY_RATE = self._config["uncertainty_rate"]
        self.VANISHING_UPDATE_TIME = self._config["vanishing_update_time"]
        self.NUMBER_OF_DRONES = self._config["number_of_drones"]
        self.MAP_WIDTH = self._config["map_width"]
        self.MAP_HEIGHT = self._config["map_height"]
        self.DISTANCE_NORM = 3611.5
        self.DISTANCE_BETWEEN_DRONE_NORM = 3563.1
        self.results_aggregator = self._config.get("results_aggregator", {})
        
        self.DRONE_ALTITUDE = 50.0
        self.DISTANCE_BETWEEN_CELLS = 20
        self.CAMERA_ANGLE = np.pi/6
        self.CELLS_EVALUETED_FOR_PRIORITY = 20

        
        ##### Initialize map #####
        self.map = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT, 2))
        self.map[:,:,0] = 1
        self.total_uncertainty = self.map[:,:,0].sum()
        self.is_cell_visited = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT))
        self.accomulated_uncertainty = 0.0
             
        ##### Initial state #####
        self.status = DroneStatus.MAPPING

        ##### Camera Configuration #####
        configuration = CameraConfiguration(100, 30, 180, 0)
        self.camera = CameraHardware(self, configuration)

        ### It's considered that the at any high the camera reach will be enough #####
        ##### Cluster plugins initialization #####
        self.fitness = FitnessEvaluator(map_width=self.MAP_WIDTH,
                                        map_height=self.MAP_HEIGHT,
                                        distance_between_cells = self.DISTANCE_BETWEEN_CELLS,
                                        distance_norm=self.DISTANCE_NORM,
                                        distance_between_drone_norm=self.DISTANCE_BETWEEN_DRONE_NORM,
                                        camera_angle=self.CAMERA_ANGLE,
                                        number_of_cells_x_y = self.CELLS_EVALUETED_FOR_PRIORITY)
        
        ##### Communication tracking. Avoiding communications loops #####
        self.last_drone_interaction_time = np.zeros(self.NUMBER_OF_DRONES)  

        ##### Initial random position #####
        self.goto_command = np.array([random.uniform(-self.DISTANCE_BETWEEN_CELLS*self.MAP_WIDTH/2, self.DISTANCE_BETWEEN_CELLS*self.MAP_WIDTH/2), random.uniform(-self.DISTANCE_BETWEEN_CELLS*self.MAP_HEIGHT/2, self.DISTANCE_BETWEEN_CELLS*self.MAP_HEIGHT/2), self.DRONE_ALTITUDE])
        command = GotoCoordsMobilityCommand(*self.goto_command)
        self.provider.send_mobility_command(command)

        self.speed_command = 15.0
        speed = SetSpeedMobilityCommand(self.speed_command)
        self.provider.send_mobility_command(speed)
        
        ##### Starting the callbacks #####
        self.provider.schedule_timer("mobility",self.provider.current_time() + 1)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 1)
        self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)
        self.provider.schedule_timer("traveled_distance", self.provider.current_time() + 2)

        #### Energy Parameters #####
        self.energy = EnergyComsuption()
        self.battery_status = self.energy.get_battery_status() 
        self.provider.schedule_timer("battery", self.provider.current_time() + 1)

        #### Mobility command buffer for RL trainning
        self.mobility_command_buffer = BufferedMobilityCommand(flag=FlagMessage.NONE.value, other_uav_id=None, partner_position=None)


    def camera_routine(self):      
        ##### New Camera Routine is needed. The previous approach was too slow for running large maps.
        ### Getting the current observation radius based on the altitude and camera angle. Assuming that the drone is always looking down.
        observation_radius = self.DRONE_ALTITUDE*np.tan(self.CAMERA_ANGLE)

        ### Converting this to number of cells to check in each direction. 
        cells_to_check = int(np.ceil(observation_radius / self.DISTANCE_BETWEEN_CELLS))
        
        ### Getting the current cell of the drone
        if self.drone_position is None:
            return
        current_x = int((self.drone_position[0] + (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2) / self.DISTANCE_BETWEEN_CELLS)
        current_y = int((self.drone_position[1] + (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2) / self.DISTANCE_BETWEEN_CELLS)

        ### Calculating the range of cells to update based on the observation radius. The range in index, so it needs to be converted to the map coordinates. 
        x_min = max(0, math.floor(current_x - cells_to_check))
        x_max = min(self.MAP_WIDTH, math.floor(current_x + cells_to_check) + 1)
        y_min = max(0, math.floor(current_y - cells_to_check))
        y_max = min(self.MAP_HEIGHT, math.floor(current_y + cells_to_check) + 1)

        #self._log.info(f"Drone {self.provider.get_id()} is updating cells in range x: [{x_min}, {x_max}), y: [{y_min}, {y_max}) based on its position {self.drone_position} and observation radius {observation_radius}")

        ### Updating the cells in the observation range
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Calculate the center coordinates of the cell
                cell_center_x = x * self.DISTANCE_BETWEEN_CELLS - (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2 
                cell_center_y = y * self.DISTANCE_BETWEEN_CELLS - (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2 

                # Check if the cell is within the observation radius
                distance_to_cell = np.sqrt((cell_center_x - self.drone_position[0])**2 + (cell_center_y - self.drone_position[1])**2)
                if distance_to_cell <= observation_radius:
                    self.map[x, y, 0] = 0.0 
                    self.map[x, y, 1] = self.provider.current_time()
                    self.is_cell_visited[x, y] = 1
        
        self.total_uncertainty = self.map[:,:,0].sum()
        self.accomulated_uncertainty += self.total_uncertainty
        self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a accomulated uncertainty of {self.accomulated_uncertainty}")
        

    ##### Map updating ##### 
    def vanishing_map_routine(self):
        self.map[:, :, 0] = self.map[:, :, 0] + self.UNCERTAINTY_RATE
        
        ##### Checking if the cell was visited #####
        ##### Importante parameter. If there are unviseted cells, there will be penalizations in the algorithm #####
        self.is_cell_visited[self.map[:, :, 1] > 0.0] = 1

        self._log.info(f"At time: {self.provider.current_time()}, the node {self.provider.get_id()} has {self.MAP_WIDTH*self.MAP_HEIGHT - np.sum(self.is_cell_visited)} unvisited cells")

         
    ##### Self mobility command. When the drone reaches the destination, it calculates the next one #####
    ##### Or after two UAVs meet. The information from the destination of the first UAV will be used in the NN not here #####
    def mobility_command(self, x: float, y: float):
        ##### receives the x and y varying from [0:1] and transform it to the map coordinates #####
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2

        target_row, target_col = int(x * self.MAP_WIDTH), int(y * self.MAP_HEIGHT)
        
        self._log.info(f"Drone {self.provider.get_id()} going to cell ({target_row}, {target_col}).")
        #### Setting the speed
        speed = SetSpeedMobilityCommand(self.speed_command)
        self.provider.send_mobility_command(speed)
        
        #### Setting the position to go to in the gradys sim coordinates.
        x_goto = target_row * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        y_goto = target_col * self.DISTANCE_BETWEEN_CELLS - map_center_offset            
        self.goto_command = np.array([x_goto, y_goto, self.DRONE_ALTITUDE])  
        command = GotoCoordsMobilityCommand(*self.goto_command)      
        self.provider.send_mobility_command(command)

        ##### Now, realease the mobility command buffer, so the drone can receive new commands from the encounters until it reaches the destination.
        self.mobility_command_buffer['flag'] = FlagMessage.NONE.value
        self.mobility_command_buffer['other_uav_id'] = None
        self.mobility_command_buffer['partner_position'] = None

    

    def send_heartbeat(self):
        #self._log.info(f"Sending heartbeat ...")
        message: HeartBeatMessage = {
            'message_type': MessageType.HEARTBEAT_MESSAGE.value,
            'status': self.status.value,
            'sender': self.provider.get_id()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def send_goto_command(self, send_command: np.array, destination_id: int, cell_priority: float):
        message: SendGoToMessage = {
            'message_type': MessageType.SHARE_GOTO_POSITION_MESSAGE.value,
            'goto': send_command.tolist(),
            'sender': self.provider.get_id(),
            'priority_value': cell_priority
        }
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)


    def compare_maps(self, incoming_map: np.ndarray) -> np.ndarray:
        condition = incoming_map[:, :, 1] > self.map[:, :, 1]
        condition_3d = condition[..., np.newaxis]
        return np.where(condition_3d, incoming_map, self.map)
    

    def received_heartbeat(self, data: dict):
        heartbeat_msg: HeartBeatMessage = data
        #self._log.info(f"Received heartbeat from {heartbeat_msg['sender']}")

        if heartbeat_msg['status'] == DroneStatus.MAPPING.value and self.status == DroneStatus.MAPPING:
            message: ShareMapMessage = {
                'message_type': MessageType.SHARE_MAP_MESSAGE.value,
                'map': self.map.tolist(),
                'sender': self.provider.get_id(),
                'drone_position': np.array(self.drone_position).tolist()
                }
            destination_id = heartbeat_msg['sender']                
            command = SendMessageCommand(json.dumps(message), destination_id)
            self.provider.send_communication_command(command)


    def updated_map(self, data: dict):
        share_map_msg: ShareMapMessage = data
        updated_map = self.compare_maps(np.array(share_map_msg['map']))
        

        return updated_map
        
    def handle_timer(self, timer: str) -> None:
        
        if timer == "vanishing_map":
                self.vanishing_map_routine()
                
                # Keep updating the uncertainty if the drone ran out of battery
                if self.status == DroneStatus.DEAD:
                    self.total_uncertainty = self.map[:,:,0].sum()
                    self.accomulated_uncertainty += self.total_uncertainty
                    self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a accomulated uncertainty of {self.accomulated_uncertainty}")
                    self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has total uncertainty of {self.total_uncertainty}")
                self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)

        if self.status == DroneStatus.MAPPING:
            if timer == "camera":
                self.camera_routine()
                self.provider.schedule_timer("camera", self.provider.current_time() + 1.0)

            if timer == "mobility": 
                if self.drone_position is not None:
                    current_pos_array = np.array(self.drone_position)    

                if np.linalg.norm(current_pos_array - self.goto_command) < 1:
                    #### In this case the drone needs to update its current goal.
                    #### This will be called in the RL framework
                    self.mobility_command_buffer['flag'] = FlagMessage.INTERNAL_MOBILITY_COMMAND.value
                    self.mobility_command_buffer['other_uav_id'] = None
                    self.mobility_command_buffer['partner_position'] = None

                self.provider.schedule_timer(
                    "mobility",
                    self.provider.current_time() + 1
                )

            if timer == "heartbeat":
                self.send_heartbeat()
                self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)

            if timer == "battery":
                movement_direction = MovementDirection.X.value
                battery_timer = 5.0
                try:
                    self.battery_status = self.energy.manage_battery_during_fly(battery_timer, self.speed_command, movement_direction)
                    #self._log.info(f"At time {self.provider.current_time()} the battery status is: {self.battery_status}")
                except BatteryError:
                    self._log.error(f"Drone {self.provider.get_id()} has no battery.")
                    self.energy.battery_status = 0.0

                    ######################################
                    #Making the drone land
                    self.goto_command = np.array(self.drone_position)
                    ### Altitude to zero
                    self.goto_command[2] = 0.0 
                    command = GotoCoordsMobilityCommand(*self.goto_command)      
                    self.provider.send_mobility_command(command)

                    self.status = DroneStatus.DEAD
                    ### The drone will stop moving and will have a larger penalty

                self.provider.schedule_timer("battery", self.provider.current_time() + battery_timer)
                


    def handle_packet(self, message: str) -> None:
        if self.status == DroneStatus.MAPPING:
            data: dict = json.loads(message)

            if 'message_type' not in data:
               self._log.warning(f"Received message without a message_type: {data}")
               return

            msg_type = data['message_type']

            if msg_type == MessageType.HEARTBEAT_MESSAGE.value:

                self.received_heartbeat(data)

            elif msg_type == MessageType.SHARE_MAP_MESSAGE.value:
                self.map = self.updated_map(data)

                if self.provider.current_time() - self.last_drone_interaction_time[data['sender']]  > self.TIMEOUT_TO_UPDATE_DESTINATION: # the drone id starts at 0
                    if self.provider.get_id() < data['sender']:
                        #### In this case, the drone with ID will decide first its next destination. Then will send to the other
                        #### drone the destination he choose. So the other drone will be able to choose a destination that is different but
                        #### also with a good fitness value. This will help to avoid that both drones go to the same place after an encounter.
                        
                        self.mobility_command_buffer['flag'] = FlagMessage.EXTERNAL_MOBILITY_COMMAND_HIGHER_PRIORITY.value
                        self.mobility_command_buffer['other_uav_id'] = None
                        self.mobility_command_buffer['partner_position'] = np.array(data['drone_position'])

                    if self.provider.get_id() > data['sender']:
                        #### This will be implemented in the RL framework. The drone with lower ID will wait for the decision of the drone with higher ID
                        #### and then will choose its destination based on that.
                        self.mobility_command_buffer['flag'] = FlagMessage.EXTERNAL_MOBILITY_COMMAND_LOWER_PRIORITY.value
                        self.mobility_command_buffer['other_uav_id'] = data['sender']
                        self.mobility_command_buffer['partner_position'] = np.array(data['drone_position'])

                    self.last_drone_interaction_time[data['sender']] = self.provider.current_time() # the drone id starts at 0
            else:
                self._log.warning(f"Received message with unknown type: {msg_type}")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.drone_position = telemetry.current_position


    def finish(self) -> None:
        pass


def drone_protocol_factory(
    uncertainty_rate: float, 
    vanishing_update_time: float, 
    number_of_drones: int,
    map_width: int,
    map_height: int,
    results_aggregator: dict
) -> Type[Drone]:
    """
    Creates a new Drone protocol class with the specified configuration.
    """
    # Create a new configuration dictionary
    config = {
        "uncertainty_rate": uncertainty_rate,
        "vanishing_update_time": vanishing_update_time,
        "number_of_drones": number_of_drones,
        "map_width": map_width,
        "map_height": map_height,
        "results_aggregator": results_aggregator
    }

    # Define a new class that inherits from Drone
    class ConfiguredDrone(Drone):
        # Override the _config class attribute with our new values
        _config = config
    
    return ConfiguredDrone