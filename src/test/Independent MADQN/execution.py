import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from protocol import drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

import numpy as np


#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation():
    ##### Creating the fuzzy lookup tables
    
    # Configuring simulation
    config = SimulationConfiguration(
        duration=2000, 
        real_time=True,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    #builder.add_handler(VisualizationHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=200
    )))

    MAP_WIDTH = 50
    MAP_HEIGHT = 50
    NUMBER_OF_DRONES = 3

    results_aggregator = {}
    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=0.01,
        vanishing_update_time=10.0,
        number_of_drones=NUMBER_OF_DRONES,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        observation_map_size=10,
        results_aggregator=results_aggregator
    )

    for _ in range(NUMBER_OF_DRONES):
        builder.add_node(ConfiguredDrone, (0, 0, 0))

  
    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()



def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'logs/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    for _ in range(20):
        create_and_run_simulation()
    

if __name__ == "__main__":
    main()