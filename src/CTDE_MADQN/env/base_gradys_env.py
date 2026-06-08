from dataclasses import dataclass
from abc import ABC, abstractmethod
from time import sleep
from gradysim.simulator.simulation import SimulationBuilder, Simulator, SimulationConfiguration, EventLoop, Node
from gradysim.simulator.handler.interface import INodeHandler

@dataclass
class SimulationStatus:
    has_ended: bool

class BaseGrADySEnvironment(ABC):
    """
    A generic TorchRL compatible environment for simulations using the GrADyS-SIM NextGen framework. 
    This class serves as a base for creating custom environments that leverage the GrADyS-SIM NextGen simulation capabilities.
    """

    simulator: Simulator
    _algorithm_iteration_interval: float
    _algorithm_iteration_finished: bool

    def __init__(self, algorithm_iteration_interval: float, visual_mode: bool = False):
        """
        Initializes the GrADySEnvironment with the specified parameters. 
        Args:
            algorithm_iteration_interval (float): Time interval between algorithm iterations in the simulation.
            device: The device on which the environment will run.
            batch_size: The batch size for the environment.
            run_type_checks (bool): Whether to run type checks.
            allow_done_after_reset (bool): Whether to allow done after reset.
            spec_locked (bool): Whether the environment specification is locked.
            auto_reset (bool): Whether to automatically reset the environment.
        """
        self._algorithm_iteration_interval = algorithm_iteration_interval
        self._algorithm_iteration_finished = False
        self.visual_mode = visual_mode

    @abstractmethod
    def _build_simulation(self, builder: SimulationBuilder):
        """
        Set up the GrADyS-SIM NextGen simulation environment with the provided configuration.

        Args:
            config (dict): Configuration parameters for the simulation setup.
        """
        pass
    
    def reset_simulation(self, simulation_configuration: SimulationConfiguration):
        """
        Resets the simulation to its initial state.
        """
        self.finalize_simulation()
        builder = SimulationBuilder(simulation_configuration)

        class GrADySHandler(INodeHandler):
            event_loop: EventLoop

            @staticmethod
            def get_label() -> str:
                return "GrADySHandler"

            def inject(self, event_loop: EventLoop) -> None:
                self.event_loop = event_loop
                self.last_iteration = 0
                self.iterate_algorithm()

            def register_node(self, node: Node) -> None:
                pass

            def iterate_algorithm(handler_self):
                self._algorithm_iteration_finished = True

                handler_self.event_loop.schedule_event(
                    handler_self.event_loop.current_time + self._algorithm_iteration_interval,
                    handler_self.iterate_algorithm
                )
        builder.add_handler(GrADySHandler())

        self._build_simulation(builder)

        self.simulator = builder.build()

        # Running a single simulation step to get the initial observations
        self.simulator._event_loop.schedule_event(0, lambda: None)  # Schedule a no-op event to kickstart the event loop
        if not self.simulator.step_simulation():
            raise RuntimeError("Simulation failed to start")

    def step_simulation(self) -> SimulationStatus:
        """
        Advances the simulation by one step.
        """
        self._algorithm_iteration_finished = False

        simulation_ongoing = True
        while not self._algorithm_iteration_finished:
            next_event = self.simulator._event_loop.peek_event()
            if self.visual_mode and next_event is not None:
                time_until_next_event = (next_event.timestamp - self.simulator._current_timestamp)
                if time_until_next_event > 0:
                    sleep(time_until_next_event)

            simulation_ongoing = self.simulator.step_simulation()
            if not simulation_ongoing:
                break
        
        return SimulationStatus(has_ended=not simulation_ongoing)


    def finalize_simulation(self):
        """
        Finalizes the simulation and releases any resources.
        """
        if hasattr(self, "simulator"):
            self.simulator._finalize_simulation()