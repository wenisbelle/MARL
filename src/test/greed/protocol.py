import enum
import math
import logging
from typing import TypedDict, Type, Optional
import numpy as np
from dataclasses import dataclass
import json
import random
from energy import EnergyComsuption, BatteryError
import os

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from gradysim.simulator.extension.camera import CameraHardware, CameraConfiguration
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from visualization import MapVisualizer

import torch 
from tensordict import TensorDict

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
    SHARE_STATE_MESSAGE = 1
    SHARE_GOTO_POSITION_MESSAGE = 2
    RECRUITING_MESSAGE = 3
    BROADCAST_DESTINATION_MESSAGE = 4

class HeartBeatMessage(TypedDict):
    message_type: int
    status: int
    sender: int

class BroadcastDestination(TypedDict):
    message_type: int
    sender: int
    destination: list

class ShareStateMessage(TypedDict):
    message_type: int 
    map: list
    sender: int
    drone_position: list
    list_drone_states: list

class SendGoToMessage(TypedDict):
    message_type: int 
    goto: list
    sender: int
    sender_position: list


class MovementDirection(enum.Enum):
    X = 0
    Z = 1


class Drone(IProtocol):
    ### Starting plugins ###
    camera: CameraHardware
    _log: logging.Logger
    ### Variable to track how many interactions happened ###
    Number_of_Encounters: int = 0
    visualizer: MapVisualizer = None

    ##### Configuration for drone inheritance #####
    _config = {
        "uncertainty_rate": 0.01,
        "vanishing_update_time": 10.0,
        "number_of_drones": 3,
        "map_width": 50,
        "map_height": 50,
        "observation_map_size": 50,
        "action_map_size": 10,
    }

    def initialize(self) -> None:
        #print("Initializing drone .........................................................")
        self._log = logging.getLogger()
        self.drone_position = None
        self.goto_command = np.zeros(3)
        self.ready = False

        self.TIMEOUT_TO_UPDATE_DESTINATION = 10.0

        self.UNCERTAINTY_RATE = self._config["uncertainty_rate"]
        self.VANISHING_UPDATE_TIME = self._config["vanishing_update_time"]
        self.NUMBER_OF_DRONES = self._config["number_of_drones"]
        self.MAP_WIDTH = self._config["map_width"]
        self.MAP_HEIGHT = self._config["map_height"]
        self.OBSERVATION_MAP_SIZE = self._config["observation_map_size"]
        self.ACTION_MAP_SIZE = self._config["action_map_size"]
        self.results_aggregator = self._config.get("results_aggregator", {})

        self.NORMALIZE_TIME = 60.0
        
        self.DRONE_ALTITUDE = 50.0
        self.DISTANCE_BETWEEN_CELLS = 20
        self.CAMERA_ANGLE = np.pi/6

        
        ##### Initialize map #####
        self.map = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT, 2))
        self.map[:,:,0] = 0.5
        self.total_uncertainty = self.map[:,:,0].sum()
        self.is_cell_visited = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT))
        self.accomulated_uncertainty = 0.0
             
        ##### Initial state #####
        self.status = DroneStatus.MAPPING

        ##### Camera Configuration #####
        configuration = CameraConfiguration(100, 30, 180, 0)
        self.camera = CameraHardware(self, configuration)
        
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

        # For the second drone, it has to wait till receiving the position the first one choose
        self.waiting_for_destination = False

        
        ##### Instead of using flags, each drone will have a list with all destinations from the other drones, indexed by the drone ID.
        ##### This model assumes that the drones position will be at their destination 
        ##### This list will have the x and y positions commanded and the time since last update. So when they exchange information they update
        ##### this list, the time and the position. 
        ##### The drone will use this same list to his neural network policy, so there is no need for creating multiple conditions. 
        ##### Besides, the drone no longer need to receive the data from the other drone to update its destination. It knows the current state
        ##### and has the same network policy, so it update it locally, no need to more messages passing. 
        self.drone_states = [{'position': np.zeros(2), 'time_of_last_update': self.provider.current_time()} for _ in range(self.NUMBER_OF_DRONES)]


        # For debugging process
        #if Drone.visualizer is None:
        #    # We have 3 drones in the simulation.
        #    # I have to think a better way to do this.
        #   Drone.visualizer = MapVisualizer(num_drones=self.NUMBER_OF_DRONES, map_width=self.MAP_WIDTH, map_height=self.MAP_HEIGHT, distance_between_cells=self.DISTANCE_BETWEEN_CELLS)
        
        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])


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
        #self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has total uncertainty of {self.total_uncertainty}")   

        self.accomulated_uncertainty += self.total_uncertainty
        #self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a accomulated uncertainty of {self.accomulated_uncertainty}")
        
        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])

    ###### Getting the total map uncertainty #####
    def get_current_map_uncertainty(self):
        return self.total_uncertainty
    
    def get_current_cell(self):
        current_x = int((self.drone_position[0] + (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2) / self.DISTANCE_BETWEEN_CELLS)
        current_y = int((self.drone_position[1] + (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2) / self.DISTANCE_BETWEEN_CELLS)

        return current_x, current_y
    

    
    ##### Map updating ##### 
    def vanishing_map_routine(self):
        self.map[:, :, 0] = self.map[:, :, 0] + self.UNCERTAINTY_RATE
        
        ##### Checking if the cell was visited #####
        ##### Importante parameter. If there are unviseted cells, there will be penalizations in the algorithm #####
        self.is_cell_visited[self.map[:, :, 1] > 0.0] = 1

        self._log.info(f"At time: {self.provider.current_time()}, the node {self.provider.get_id()} has {self.MAP_WIDTH*self.MAP_HEIGHT - np.sum(self.is_cell_visited)} unvisited cells")


    def check_maximum_cells(self, reduced_map: np.array):
        min_knowledge = reduced_map[:, :].max()
        rows, cols = np.where(reduced_map[:, :] == min_knowledge)
        return [list(zip(rows, cols)), min_knowledge]

    ##### Mobility command. When the drone reaches the destination, it calculates the next one #####
    def mobility_command(self):
        map_data = self.map[:,:,0].copy()
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
        
        maximum_cells_coords, max_value = self.check_maximum_cells(map_data)
        
        if maximum_cells_coords:
            target_coords = random.choice(maximum_cells_coords)
        
        target_row, target_col = target_coords
        
        #### Setting the position to go to in the gradys sim coordinates.
        x_goto = target_row * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        y_goto = target_col * self.DISTANCE_BETWEEN_CELLS - map_center_offset            
        self.goto_command = np.array([x_goto, y_goto, self.DRONE_ALTITUDE])  
        command = GotoCoordsMobilityCommand(*self.goto_command)      
        self.provider.send_mobility_command(command)

        #### Now, broadcast this information to any other drone in the communication range
        self.send_broadcast_destination()
        #print(f"Drone {self.provider.get_id()} broadcasted its destination {self.goto_command} to other drones.") 


    def send_heartbeat(self):
        #self._log.info(f"Sending heartbeat ...")
        message: HeartBeatMessage = {
            'message_type': MessageType.HEARTBEAT_MESSAGE.value,
            'status': self.status.value,
            'sender': self.provider.get_id()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)
    
    def send_broadcast_destination(self):
        message: BroadcastDestination = {
            'message_type': MessageType.BROADCAST_DESTINATION_MESSAGE.value,
            'sender': self.provider.get_id(),
            'destination': self.goto_command.tolist()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def send_goto_command(self, send_command: np.array, destination_id: int):
        message: SendGoToMessage = {
            'message_type': MessageType.SHARE_GOTO_POSITION_MESSAGE.value,
            'goto': send_command.tolist(),
            'sender': self.provider.get_id(),
            'sender_position': np.array(self.drone_position).tolist()
        }
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)

    def send_states_message(self, destination_id: int):
        # Update the drone own state before sending 
        self.drone_states[self.provider.get_id()]['position'] = np.array(self.drone_position[0:2]) # only x and y
        self.drone_states[self.provider.get_id()]['time_of_last_update'] = self.provider.current_time()
        message: ShareStateMessage = {
                'message_type': MessageType.SHARE_STATE_MESSAGE.value,
                'map': self.map.tolist(),
                'sender': self.provider.get_id(),
                'drone_position': np.array(self.drone_position).tolist(),
                'list_drone_states': [
                    {'position': s['position'].tolist(),
                     'time_of_last_update': s['time_of_last_update']}
                    for s in self.drone_states
                ],
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
            destination_id = heartbeat_msg['sender']
            self.send_states_message(destination_id)

            
    def compare_states(self, incoming_state: list) -> list:
        for i in range(self.NUMBER_OF_DRONES):
            if incoming_state[i]['time_of_last_update'] > self.drone_states[i]['time_of_last_update']:
                self.drone_states[i] = {
                    'position': np.array(incoming_state[i]['position'], dtype=np.float32),
                    'time_of_last_update': incoming_state[i]['time_of_last_update'],
            }

    def update_states(self, data: dict):
        share_state_msg: ShareStateMessage = data
        # Update the map
        self.map = self.compare_maps(np.array(share_state_msg['map']))
        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])

        # Updating the drone states
        #print(f"Drone {self.provider.get_id()} is updating the original states: {self.drone_states}")
        self.compare_states(share_state_msg['list_drone_states'])
        #print(f"Drone {self.provider.get_id()} updated the status to {self.drone_states}")
        
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
                    self.mobility_command()

                #print(f"Drone {self.provider.get_id()} has a total uncertainty of {self.total_uncertainty} and an accomulated uncertainty of {self.accomulated_uncertainty}")

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
                # send the map and list of states to the other drone
                self.received_heartbeat(data)

            elif msg_type == MessageType.SHARE_STATE_MESSAGE.value:
                # Update the map and the states of the drones, both will be used in the NN policy
                self.update_states(data)
                if self.provider.current_time() - self.last_drone_interaction_time[data['sender']]  > self.TIMEOUT_TO_UPDATE_DESTINATION: # the drone id starts at 0
                    if self.provider.get_id() > data['sender']:
                        self.mobility_command()
                        self._log.info(f"Drone {self.provider.get_id()} updated position command.")   
                    else:
                        self.waiting_for_destination = True
                    
                    self.last_drone_interaction_time[data['sender']] = self.provider.current_time() 
            
            elif msg_type == MessageType.BROADCAST_DESTINATION_MESSAGE.value:
                sender_id = data['sender']
                destination = np.array(data['destination'])
                self.drone_states[sender_id]['position'] = destination[0:2] # only x and y
                self.drone_states[sender_id]['time_of_last_update'] = self.provider.current_time()
                #print(f"Drone {self.provider.get_id()} received broadcasted destination {destination} from drone {sender_id}.")
                if self.waiting_for_destination:
                    self.mobility_command()
                    self.waiting_for_destination = False
                    self._log.info(f"Drone {self.provider.get_id()} received broadcasted destination {destination} from drone {sender_id}")
            else:
                self._log.warning(f"Received message with unknown type: {msg_type}")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.drone_position = telemetry.current_position
        #self._log.info(f"Drone {self.provider.get_id()} position updated to {self.drone_position}")
        self.ready = True 

    def die(self) -> None:
        self.status = DroneStatus.DEAD

    def finish(self) -> None:
        pass


def drone_protocol_factory(
    uncertainty_rate: float, 
    vanishing_update_time: float, 
    number_of_drones: int,
    map_width: int,
    map_height: int,
    observation_map_size: int,
    action_map_size: int,
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
        "observation_map_size": observation_map_size,
        "action_map_size": action_map_size,
        "results_aggregator": results_aggregator
    }

    # Define a new class that inherits from Drone
    class ConfiguredDrone(Drone):
        # Override the _config class attribute with our new values
        _config = config
    
    return ConfiguredDrone