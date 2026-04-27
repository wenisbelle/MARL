from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from protocol import SimpleUAVProtocol


def main():
    # Configuring simulation
    config = SimulationConfiguration(
        duration=40,
        real_time=True
    )
    builder = SimulationBuilder(config)

    builder.add_node(SimpleUAVProtocol, (-50, 0, 10))
    builder.add_node(SimpleUAVProtocol, (50, 0, 10))


    # Adding required handlers
    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
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
