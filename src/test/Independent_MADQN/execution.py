import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from protocol import drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from recorder import SimulationRecorder

import numpy as np


#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation():
    ##### Creating the fuzzy lookup tables
    
    # Configuring simulation
    config = SimulationConfiguration(
        duration=2000, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    #builder.add_handler(VisualizationHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=100
    )))

    MAP_WIDTH = 50
    MAP_HEIGHT = 50
    NUMBER_OF_DRONES = 5
    DISTANCE_BETWEEN_CELLS = 20

    results_aggregator = {}
    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=0.01,
        vanishing_update_time=10.0,
        number_of_drones=NUMBER_OF_DRONES,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        observation_map_size=50,
        action_map_size=11,
        results_aggregator=results_aggregator
    )

    for _ in range(NUMBER_OF_DRONES):
        builder.add_node(ConfiguredDrone, (random.uniform(-MAP_WIDTH*DISTANCE_BETWEEN_CELLS/2, MAP_WIDTH*DISTANCE_BETWEEN_CELLS/2), 
                                           random.uniform(-MAP_HEIGHT*DISTANCE_BETWEEN_CELLS/2, MAP_HEIGHT*DISTANCE_BETWEEN_CELLS/2), 
                                           50))

  
    # Building & starting
    simulation = builder.build()
    recorder = SimulationRecorder(simulation, NUMBER_OF_DRONES, MAP_WIDTH, MAP_HEIGHT, distance_between_cells=20)

    def get_global_map():
        agents_maps = []
        for i in range(NUMBER_OF_DRONES):
            agent_node = simulation.get_node(i)
            protocol = agent_node.protocol_encapsulator.protocol
            agents_maps.append(protocol.map[:,:,0])

        return np.min(agents_maps, axis=0)


    STEP_INTERVAL = 1.0         
    next_checkpoint = STEP_INTERVAL
    running = True

    while running:
        while simulation._current_timestamp < next_checkpoint:
            running = simulation.step_simulation()
            if not running:
                break
        if running:
            global_map = get_global_map()
            sum = global_map.sum()
            print(f"At time: {simulation._current_timestamp} the global map: {sum}")
            recorder.snapshot()  
        next_checkpoint += STEP_INTERVAL
    recorder.save("logs/recording.pkl", results_aggregator)


def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'logs/simulation.log', 
        filemode='w', 
        format='%(message)s'  
    )
    for _ in range(1):
        create_and_run_simulation()
    

if __name__ == "__main__":
    main()