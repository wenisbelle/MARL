import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .protocol import drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from deap import algorithms, base, creator, tools
import numpy as np


#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(individual):
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
        distance_norm=individual[0],
        distance_between_drone_norm=individual[1],
        results_aggregator=results_aggregator
    )

    for _ in range(NUMBER_OF_DRONES):
        builder.add_node(ConfiguredDrone, (0, 0, 0))

  
    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()

    medium_uncertainty = 0
    for i in range(NUMBER_OF_DRONES):
        medium_uncertainty += 0.01*results_aggregator[i]['accomulated_uncertainty']/NUMBER_OF_DRONES

    print(f"Variable to be minimized: {medium_uncertainty}")    
    return medium_uncertainty


def main():
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'definitive_system/coordination_only/3_drones/test/logs/analytical_2000/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    for _ in range(20):
        individual =  [3611.5, 3563.1]
        create_and_run_simulation(individual)
    

if __name__ == "__main__":
    main()