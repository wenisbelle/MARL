import math
import random
from typing import List, cast

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import SetSpeedMobilityCommand, GotoCoordsMobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.simulator.extension.visualization_controller import VisualizationController


class SensorProtocol(IProtocol):
    """
    Protocol for a sensor that collects data packets broadcasted by drones.
    Its behavior is such:
    - On initialization, it randomly assigns itself a priority between min_priority and max_priority.
    - When it receives a message from a drone, it marks itself as having collected data.
    """
    has_collected: bool

    min_priority: int = 0
    max_priority: int = 1

    def initialize(self) -> None:
        self.priority = random.uniform(self.min_priority, self.max_priority)
        self.provider.tracked_variables["priority"] = self.priority
        self.has_collected = False
        self.provider.tracked_variables["collected"] = self.has_collected
        self.controller = VisualizationController(self)
        self.controller.paint_node(self.provider.get_id(), color=(255, 0, 0))

    def handle_packet(self, message: str) -> None:
        self.has_collected = True
        self.provider.tracked_variables["collected"] = self.has_collected
        self.controller.paint_node(self.provider.get_id(), color=(0, 255, 0))

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        pass

def direction_to_unit_vector(angle: float) -> tuple[float, float]:
    """
    Convert an angle in [0, 2PI] to a unit vector (x, y).
    :param angle: Angle in radians.
    :return: Unit vector (x, y) corresponding to the direction.
    """
    return math.cos(angle), math.sin(angle)

def coords_away_from_edge(current_position: tuple[float, float], coordinate_limit: float, margin: float) -> bool:
    """
    Check if the current position is within a margin away from the edge of the scenario.
    :param current_position: Current position (x, y, z).
    :param coordinate_limit: Limit of the scenario coordinates (assumed square from -limit to +limit).
    :param margin: Margin distance from the edge.
    :return: True if within margin from edge, False otherwise.
    """
    x, y = current_position
    return (abs(x) >= coordinate_limit - margin) or (abs(y) >= coordinate_limit - margin)

def extend_unit_vector_to_edge(current_position: tuple[float, float],
                               unit_vector: tuple[float, float],
                               coordinate_limit: float) -> tuple[float, float]:
    """
    Extend the unit vector from the current position to the edge of the scenario. Does so by scaling the vector
    to the nearest edge. Ex: if the current position is (2,3) in a square of limit 10, and the unit vector is (1,0),
    the resulting vector will be (8,0) to reach the edge at x=10. The resulting vector will always point in the same
    direction as the input unit vector.
    :param current_position: Current position (x, y).
    :param unit_vector: Unit vector (x, y) representing direction.
    :param coordinate_limit: Limit of the scenario coordinates (assumed square from -limit to +limit).
    :return: Scaled vector (x, y) to reach the edge of the scenario.
    """
    if unit_vector[0] == 0 and unit_vector[1] == 0:
        return 0.0, 0.0

    # Determine the maximum distance we can go in x and y directions
    max_x_size = coordinate_limit - abs(current_position[0])
    max_y_size = coordinate_limit - abs(current_position[1])

    # Determine scaling factors for x and y to reach the edge
    scale_x = max_x_size / abs(unit_vector[0]) if unit_vector[0] != 0 else float('inf')
    scale_y = max_y_size / abs(unit_vector[1]) if unit_vector[1] != 0 else float('inf')

    # Use the minimum scaling factor to ensure we don't exceed the edge in either direction
    scale = min(scale_x, scale_y)

    return unit_vector[0] * scale, unit_vector[1] * scale

_sqrt_2 = math.sqrt(2)
def project_vector_onto_edge(current_position: tuple[float, float],
                             unit_vector: tuple[float, float],
                             coordinate_limit: float) -> tuple[float, float]:
    """
    Project the unit vector onto the edge of the scenario from the current position. This is done by nullifying the
    component of the vector that points outside the scenario.
    :param current_position: Current position (x, y).
    :param unit_vector: Unit vector (x, y) representing direction.
    :param coordinate_limit: Limit of the scenario coordinates (assumed square from -limit to +limit).
    :return: Projected vector (x, y) onto the edge of the scenario.
    """
    # Start by extending the vector to a size large enough to reach the edge in any direction
    coord_limit_vector = [
        unit_vector[0] * coordinate_limit * 2 * _sqrt_2,
        unit_vector[1] * coordinate_limit * 2 * _sqrt_2
    ]

    # Now clamp the vector components to not exceed the distance to the edge from current position
    max_x_size = coordinate_limit - abs(current_position[0])
    max_y_size = coordinate_limit - abs(current_position[1])
    if abs(coord_limit_vector[0]) > max_x_size:
        coord_limit_vector[0] = math.copysign(max_x_size, coord_limit_vector[0])
    if abs(coord_limit_vector[1]) > max_y_size:
        coord_limit_vector[1] = math.copysign(max_y_size, coord_limit_vector[1])

    return cast(tuple[float, float], tuple(coord_limit_vector))

def prevent_vector_escape(current_position: tuple[float, float],
                           unit_vector: tuple[float, float],
                           coordinate_limit: float) -> tuple[float, float]:
    """
    Prevent the unit vector from pointing outside the scenario boundaries. We do this by one of two strategies:
    1. If the current position is away from the edge, we extend the vector to the edge of the scenario, not letting it
       go beyond.
    2. If the current position is at the edge, we project the vector onto the edge, nullifying any component that
       points outside. That way the vector makes the node "slide" along the edge.
    :param current_position: Current position (x, y).
    :param unit_vector: Unit vector (x, y) representing direction.
    :param coordinate_limit: Limit of the scenario coordinates (assumed square from -limit to +limit).
    :return: Adjusted vector (x, y) that stays within scenario boundaries.
    """
    if not coords_away_from_edge(current_position, coordinate_limit, margin=1e-2):
        return extend_unit_vector_to_edge(current_position, unit_vector, coordinate_limit)
    else:
        return project_vector_onto_edge(current_position, unit_vector, coordinate_limit)

class DroneProtocol(IProtocol):
    """
    Protocol for a drone that navigates within a bounded square scenario.
    - The drone's action consists of a direction (angle in [0, 1]) and optionally a speed (in [0, 1]). Actions are
      provided through the `act` method.
    - The drone moves in the specified direction, scaling its movement to stay within the scenario boundaries.
    - If the drone reaches the edge of the scenario, it adjusts its destination to remain within bounds.
    - The drone periodically broadcasts messages to collect data packets from sensors.
    """
    current_position: tuple[float, float, float] | None
    ready: bool
    dead: bool
    speed_action: bool = False
    algorithm_interval: float = 0.1

    def act(self, action: List[float], coordinate_limit: float) -> None:
        if self.dead:
            return
        self.controller.paint_node(self.provider.get_id(), color=(0, 0, 0))

        if self.current_position is None:
            raise RuntimeError("Called act before receiving initial telemetry")
        
        self.provider.tracked_variables['current_action'] = list(action)

        if self.speed_action:
            speed: float = action[1] * 15
            command = SetSpeedMobilityCommand(speed)
            self.provider.send_mobility_command(command)

        # action[0] is a number in [0, 1] representing the direction as an angle. Convert it to an angle in [0, 2PI]
        # radians
        angle = action[0] * 2 * math.pi

        unit_vector = direction_to_unit_vector(angle)

        # Maintain direction but bound destination within scenario
        x_y_coords = (self.current_position[0], self.current_position[1])

        bounded_vector = prevent_vector_escape(x_y_coords, unit_vector, coordinate_limit)
        destination = [
            self.current_position[0] + bounded_vector[0],
            self.current_position[1] + bounded_vector[1],
            0
        ]

        # Start traveling in the direction of travel
        command = GotoCoordsMobilityCommand(*destination)
        self.provider.send_mobility_command(command)

        if self.speed_action:
            self.provider.schedule_timer("", self.provider.current_time() + self.algorithm_interval * 0.99)

    def initialize(self) -> None:
        self.current_position = None
        self.ready = False
        self.dead = False
        self._collect_packets()
        if not self.speed_action:
            self.provider.schedule_timer("", self.provider.current_time() + 0.1)

        self.controller = VisualizationController(self)
        self.controller.paint_node(self.provider.get_id(), color=(0, 0, 0))

    def handle_timer(self, timer: str) -> None:
        if self.dead:
            return
        self._collect_packets()
        if not self.speed_action:
            self.provider.schedule_timer("", self.provider.current_time() + 0.1)

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = telemetry.current_position
        self.ready = True

    def _collect_packets(self) -> None:
        if self.dead:
            return
        command = BroadcastMessageCommand("")
        self.provider.send_communication_command(command)

    def die(self) -> None:
        self.dead = True
        if self.current_position is None:
            return

        grounded_position = (
            self.current_position[0],
            self.current_position[1],
            0.0,
        )
        self.current_position = grounded_position
        self.provider.send_mobility_command(GotoCoordsMobilityCommand(*grounded_position))
        self.controller.paint_node(self.provider.get_id(), color=(128, 128, 128))

    def finish(self) -> None:
        pass
