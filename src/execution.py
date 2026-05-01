from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

from protocol import SimpleUAVProtocol

# ==========================================
# CHANGE THIS TO TEST DIFFERENT SCENARIOS
# 0: Head-on (Original)
# 1: Lateral (Perpendicular intersection)
# 2: Diagonal (X-pattern intersection)
# 3: 2 Same Direction vs 1 Opposite
# 4: 3-way Diagonal Intersection
# ==========================================
TEST_SCENARIO = 3

def main():
    # Configuring simulation
    config = SimulationConfiguration(
        duration=30,
        real_time=True
    )
    builder = SimulationBuilder(config)

    # Automatically load the correct starting positions for the scenario
    if TEST_SCENARIO == 0:
        builder.add_node(SimpleUAVProtocol, (-50, 0, 10))
        builder.add_node(SimpleUAVProtocol, (50, 0, 10))
        
    elif TEST_SCENARIO == 1:
        builder.add_node(SimpleUAVProtocol, (-50, 0, 10))
        builder.add_node(SimpleUAVProtocol, (0, -50, 10))
        
    elif TEST_SCENARIO == 2:
        builder.add_node(SimpleUAVProtocol, (-50, -50, 10))
        builder.add_node(SimpleUAVProtocol, (50, -50, 10))
        
    elif TEST_SCENARIO == 3:
        builder.add_node(SimpleUAVProtocol, (-60, 0, 10)) # Trailing drone
        builder.add_node(SimpleUAVProtocol, (-30, 0, 10)) # Lead drone
        builder.add_node(SimpleUAVProtocol, (50, 0, 10))  # Opposite drone
        
    elif TEST_SCENARIO == 4:
        builder.add_node(SimpleUAVProtocol, (-50, -50, 10))
        builder.add_node(SimpleUAVProtocol, (50, -50, 10))
        # 70.7 ensures this drone hits the (0,0) intersection at the exact same time as the others
        builder.add_node(SimpleUAVProtocol, (0, 70.7, 10)) 


    # Adding required handlers
    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=50
    )))
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler(VisualizationConfiguration(
        x_range=(-150, 150),
        y_range=(-150, 150),
        z_range=(0, 150),
        open_browser=True
    )))

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()


if __name__ == "__main__":
    main()