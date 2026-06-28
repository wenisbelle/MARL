import dataclasses
import math
import random
from typing import Optional
from scipy.spatial.distance import pdist

import numpy as np
import torch
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from tensordict import TensorDictBase
from torchrl.data import Bounded, Binary
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from .environment import EndCause
from .metrics import make_data_collection_metrics_spec, EnvironmentMetricsSpec
from .gradysim_environment.protocol import Drone, drone_protocol_factory, FlagMessage
from .base_gradys_env import BaseGrADySEnvironment

@dataclasses.dataclass(slots=True)
class EpisodeAgentState():
    """Bookkeeping for a single stable agent slot in the current episode."""

    slot_index: int
    name: str
    exists: bool = False
    """Whether this agent slot is occupied by an actual agent in the current episode. If False, this slot is a 
    placeholder that gets truncated immediately at reset and should not be filled with meaningful data."""

    active: bool = False
    """Whether this agent slot is currently active and should be stepped. If False, this agent should not be stepped or
     filled with meaningful data, but it may still exist in the simulation if it was active in the past"""

    node_id: int | None = None
  
    #individual_map_uncertainty: float = 0.0
    #current_position: np.array = np.zeros(2)
    #estimation_drone_positions: np.array = np.zeros((self.max_num_agents, 2))
    #flag: int = FlagMessage.NONE.value 
    #individual_patch_map: np.ndarray

@dataclasses.dataclass(slots=True)
class EpisodeGlobalState:
    global_map_uncertainty: float
    active_drones_positions: np.ndarray
    """Position of all active drones in the environment"""
    global_map: np.ndarray
    """The global map is computed by taking the lowest uncertainty value for each cell across all drones"""

@dataclasses.dataclass
class MappingEnvironmentConfig:
    """Configuration for GrADyS environment (only 'relative' observation mode retained)."""

    render_mode: Optional[str] = None  # "visual" | "console"
    algorithm_iteration_interval: float = 0.5
    # Number of drone agents is sampled from [min_num_agents, max_num_agents].
    # To fix the number, set min_num_agents == max_num_agents.
    min_num_agents: int = 3
    max_num_agents: int = 3
    map_width: int = 50
    map_height: int = 50
    observation_map_size: int = 20 # Observe a square of side 20 cells centered on the drone. 
    action_map_size: int = 10
    drone_altitude: float = 50.0
    distance_between_cells: float = 20
    uncertainty_rate: float = 0.001
    vanishing_update_time: float = 1.0

    max_episode_length: int = 500
    communication_range: float = 200
    full_random_drone_position: bool = True
    reward: str = 'punish'  # Fixed reward mode: punish
    speed_action: bool = False
    agent_death_probability: float = 0.0
    normalize_time: float = 60.0


class MappingEnvironment(BaseGrADySEnvironment, EnvBase):
    """
    A specialized environment for data collection in simulations, extending the GrADySEnvironment.
    This environment simulates sensor data collection with autonomous agents.
    Per-episode agent state is centralized in `episode_agents`.
    Per-episode global state is centralized in `episode_global`.
    """

    _simulation_configuration: SimulationConfiguration

    batch_locked: bool = True

    def __init__(self, config: MappingEnvironmentConfig, *, device=None):
        BaseGrADySEnvironment.__init__(self, config.algorithm_iteration_interval, visual_mode=(config.render_mode == "visual"))
        EnvBase.__init__(self, device=device)

        self.render_mode = config.render_mode
        self.algorithm_iteration_interval = config.algorithm_iteration_interval

        self.min_num_agents = config.min_num_agents
        self.max_num_agents = config.max_num_agents

        self.map_width = config.map_width
        self.map_height = config.map_height
        self.observation_map_size = config.observation_map_size
        self.action_map_size = config.action_map_size
        self.drone_altitude = config.drone_altitude
        self.distance_between_cells = config.distance_between_cells
        self.uncertainty_rate = config.uncertainty_rate
        self.vanishing_update_time = config.vanishing_update_time

        self.max_episode_length = config.max_episode_length
        self.communication_range = config.communication_range
        self.full_random_drone_position = config.full_random_drone_position
        if config.reward != "punish":
            raise ValueError("Only reward='punish' is supported.")
        if not 0.0 <= config.agent_death_probability <= 1.0:
            raise ValueError("agent_death_probability must be in [0, 1].")
        self.speed_action = config.speed_action
        self.agent_death_probability = config.agent_death_probability

        ###########################################
        # IMPORTANT PARAMETER CHANGES THE BEHAVIOR#
        ###########################################
        self.NORMALIZE_TIME               = config.normalize_time
        # after taking an action an immediate reward is given to the agent based on the action taken
        self.immediate_reward_from_action = [0.0 for _ in range(self.max_num_agents)]
        
        self.possible_agents = [f"drone{i}" for i in range(self.max_num_agents)]
        self.group_map = {"agents": self.possible_agents}

        self.episode_agents: list[EpisodeAgentState] = []
        self.episode_global: EpisodeGlobalState
        self.episode_duration = 0
        self.global_reward = 0.0
        self.individual_reward = [0.0 for _ in range(self.max_num_agents)]
        self.max_global_reward = -math.inf
        self.max_individual_reward = -math.inf

        

        self._metrics_spec: EnvironmentMetricsSpec = make_data_collection_metrics_spec()
        self._info_keys = self._metrics_spec.info_keys
        self._all_info_keys = self._info_keys 


        self._simulation_configuration = SimulationConfiguration(
            debug=False,
            execution_logging=False,
            duration=self.max_episode_length,
        )

        self._build_specs()
        self._cached_reset_zero = self.full_observation_spec.zero()
        self._cached_reset_zero.update(self.full_done_spec.zero())

        self._cached_step_zero = self.full_observation_spec.zero()
        self._cached_step_zero.update(self.full_reward_spec.zero())
        self._cached_step_zero.update(self.full_done_spec.zero())

    def _existing_episode_agents(self) -> list[EpisodeAgentState]:
        """Return episode slots that exist in the current episode, in slot order."""
        return [agent for agent in self.episode_agents if agent.exists]

    def _active_episode_agents(self) -> list[EpisodeAgentState]:
        """Return the episode slots that are currently active and should act."""
        return [agent for agent in self.episode_agents if agent.exists and agent.active]

    def _inactive_existing_episode_agents(self) -> list[EpisodeAgentState]:
        """Return real episode slots that are currently inactive."""
        return [agent for agent in self.episode_agents if agent.exists and not agent.active]

    def _agent_slot_tensor(self, agents: list[EpisodeAgentState]) -> torch.Tensor:
        """Return the given episode agents' slot indices on the environment device."""
        return torch.tensor([agent.slot_index for agent in agents], device=self.device, dtype=torch.long)

    def _build_simulation(self, builder: SimulationBuilder):
        """
        Set up the GrADyS-SIM NextGen simulation environment with the provided configuration.

        Args:
            builder (SimulationBuilder): Builder object for setting up the simulation.
        """
        # Adding necessary handlers to the simulation builder
        builder.add_handler(CommunicationHandler(CommunicationMedium(
            transmission_range=self.communication_range
        )))
        builder.add_handler(MobilityHandler())
        builder.add_handler(TimerHandler())

        if self.render_mode == "visual":
            scenario_size = max(self.map_width/2, self.map_height/2)*self.distance_between_cells
            builder.add_handler(VisualizationHandler(VisualizationConfiguration(
                open_browser=False,
                x_range=(-scenario_size, scenario_size),
                y_range=(-scenario_size, scenario_size),
                z_range=(0, 100),
            )))

        results_aggregator = {}
        ### Initial map most be the more diverse possible
        num = random.random()

        if num < 0.25:
            ### Uniform values between 0 and 1
            initial_map = np.random.rand(self.map_width, self.map_height)
        elif num < 0.5:
            ### Uniform values between 0.5 and 1.0
            initial_map = np.random.uniform(low=0.5, high=1.0, size=(self.map_width, self.map_height))           
        elif num < 0.75:
            ### The map will start with 25% of the cells with a high uncertainty value between 1 and 2.
            mask = np.random.rand(self.map_width, self.map_height) < 0.25
            values_0_to_1 = np.random.rand(self.map_width, self.map_height)
            values_2_to_3 = np.random.rand(self.map_width, self.map_height) + 1
            initial_map = np.where(mask, values_2_to_3, values_0_to_1)
        else:
            ### The map will start with an initial high uncertainty value of 1.0 in the whole map
            initial_map = np.ones((self.map_width, self.map_height))

        
        ConfiguredDrone = drone_protocol_factory(uncertainty_rate=self.uncertainty_rate,
                                                 vanishing_update_time=self.vanishing_update_time,
                                                 number_of_drones=self.max_num_agents,
                                                 map_width=self.map_width,
                                                 map_height=self.map_height,
                                                 results_aggregator=results_aggregator,
                                                 initial_map=initial_map)
                                                 
        # The episode state keeps stable slot identity; only existing agents get simulator nodes.
        for agent in self._existing_episode_agents():
            if self.full_random_drone_position:
                agent.node_id = builder.add_node(ConfiguredDrone, (
                    random.uniform(-self.map_width*self.distance_between_cells/2, self.map_width*self.distance_between_cells/2),
                    random.uniform(-self.map_height*self.distance_between_cells/2, self.map_height*self.distance_between_cells/2),
                    self.drone_altitude
                ))
            else:
                agent.node_id = builder.add_node(ConfiguredDrone, (
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    self.drone_altitude
                ))

        self.simulator = builder.build()

    def _build_specs(self) -> None:
        device = self.device

        ##### For now just the x, y positions. If I want to use the speed controller it will be 3. 
        M = self.observation_map_size
        A = self.action_map_size
        n_actions = A * A # total of discrete actions 
        all_positions_shape = (self.max_num_agents, 2)
        estimated_positions_and_time_shape = (self.max_num_agents, self.max_num_agents-1, 3) # It will not store its own data
        # For each agent, an estimate of every other agent's position and time since last update
        # The time will vary from 0 to 1, where 0 means the information is fresh and 1 means the information is as old as the vanishing_update_time - Hoping the system can learn
        # to give more importance to fresher information

        ### The map will be divided in 2 channels: one for uncertainty and other for the normalized distances from the 
        ### drone current cell to the others in the map.
        ### The uncertainty channel is the (uncertainty of the cell - mean value)/(standard_ deviation). So the value will be cliped 
        ### from -5 to 5. 
        large_map_patch_shape = (self.max_num_agents, 2, M, M) 
        # Reduced observed map with size equal to the action map
        small_map_patch_shape = (self.max_num_agents, 2, A, A) 
        complete_map_shape = (self.map_width, self.map_height)
        
        position_shape = (self.max_num_agents, 2)

        #### Total uncertainty will be given by the mean the and the standard deviation
        uncertainty_shape = (self.max_num_agents, 2)
        mask_shape = (self.max_num_agents,)
        action_shape = (self.max_num_agents,1)
        reward_shape = (self.max_num_agents, 1)
        done_shape = (self.max_num_agents, 1)

        obs_inner = {
            "large_map_patch": Bounded(
                torch.full(large_map_patch_shape, -5.0, device=device),
                torch.full(large_map_patch_shape, 5.0, device=device),
                large_map_patch_shape,
                dtype=torch.float32,
                device=device,
            ),
            "small_map_patch": Bounded(
                torch.full(small_map_patch_shape, -5.0, device=device),
                torch.full(small_map_patch_shape, 5.0, device=device),
                small_map_patch_shape,
                dtype=torch.float32,
                device=device,
            ),
            ##### the total uncertainty is represented by the mean and the standard deviation of the uncertainty map
            "individual_map_uncertainty": Bounded(
                torch.zeros(uncertainty_shape, device=device),
                torch.full(uncertainty_shape, 5.0, device=device),  # normalized to [0,5] - Better for training stability
                uncertainty_shape,
                dtype=torch.float32,
                device=device,
            ),
            "position": Bounded(
                torch.zeros(position_shape, device=device),
                torch.ones(position_shape, device=device),  # normalized to [0,1] - Better for training stability
                position_shape,
                dtype=torch.float32,
                device=device,
            ),
            "estimated_positions_and_time": Bounded(
                torch.zeros(estimated_positions_and_time_shape, device=device),       #### In some cases this will not be used 
                torch.ones(estimated_positions_and_time_shape, device=device),        #### normalized to [0,1]
                estimated_positions_and_time_shape,
                dtype=torch.float32,
                device=device,
            ),
            "encounter_flag": Binary(
                n=1,
                shape=(self.max_num_agents, 1),
                device=device,
                dtype=torch.bool,
            ),
            "valid_actions": Bounded(
                torch.zeros((self.max_num_agents, A * A), device=device),
                torch.ones((self.max_num_agents, A * A), device=device),
                (self.max_num_agents, A * A),
                dtype=torch.bool,
                device=device,
            ),
        }

        ##### For critics, it will observe the position of all drones and the full map, with lowest uncertainty in each cell from all individual observations.
        obs_global = {
            "full_map": Bounded(
                torch.full(complete_map_shape, -5.0, device=device),
                torch.full(complete_map_shape, 5.0, device=device),  # same reasoning as map_patch
                complete_map_shape,
                dtype=torch.float32,
                device=device,
            ),
            "global_map_uncertainty": Bounded(
                torch.zeros((1,2), device=device),
                torch.full((1,2), 5.0, device=device),  # normalized to [0,5] - Better for training stability
                (1,2),
                dtype=torch.float32,
                device=device,
            ),
            "all_positions": Bounded(
                torch.full(all_positions_shape, -1.0, device=device),
                torch.ones(all_positions_shape, device=device),
                all_positions_shape,
                dtype=torch.float32,
                device=device,
            ),
            "all_active": Categorical(
                n=2,
                shape=(self.max_num_agents,),
                dtype=torch.bool,
                device=device,
            ),
        }

        observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Composite(obs_inner, device=device),
                        "mask": Categorical(
                            n=2,
                            shape=mask_shape,
                            dtype=torch.bool,
                            device=device,
                        ),
                        "info": Composite(
                            {
                                key: Unbounded(
                                    shape=(self.max_num_agents,),
                                    device=device,
                                    dtype=torch.float32,
                                )
                                for key in self._all_info_keys
                            },
                            device=device,
                        ),
                    },
                    device=device,
                ),

                "global_state": Composite(obs_global, device=device),
            },
            device=device,
        )
        
        ##### Output is also bounded [0,1]
        action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": Categorical(
                            n=n_actions,  # Generates ints from 0 to (M^2 - 1)
                            shape=action_shape,
                            dtype=torch.int64,
                            device=device,
                        )
                    },
                    device=device,
                )
            },
            device=device,
        )

        reward_spec = Composite(
            {
                "agents": Composite(
                    {
                        "reward": Unbounded(
                            shape=reward_shape,
                            device=device,
                            dtype=torch.float32
                        )
                    },
                    device=device,
                ),
                "global_reward": Unbounded(
                    shape=(1,),
                    device=device,
                    dtype=torch.float32
                )
            },
            device=device,
        )

        done_spec = Composite(
            {
                "done": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "terminated": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "truncated": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "agents": Composite(
                    {
                        "done": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                        "terminated": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                        "truncated": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                    },
                    device=device,
                ),
            },
            device=device,
        )

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        ##### Get actions from the dictionary
        actions = tensordict.get(("agents", "action"))

        ##### Get the individual uncertainty before the step, to compute the reward later
        pre_step_agents = self._active_episode_agents()
        collected_individual_uncertainty_before_mean = self.get_individual_uncertainty_from_simulation(pre_step_agents)
        collected_global_uncertainty_before_mean = self.get_global_map_from_simulation(pre_step_agents).mean()

        self._apply_actions(actions)
        status = self.step_simulation()
        end_cause = EndCause.NONE

        ##### Get the individual uncertainty after the step, to compute the reward later
        ##### For now, in this calculation we are considering that the reward is given to the specific agent even if he dies in this step
        collected_individual_uncertainty_after_mean = self.get_individual_uncertainty_from_simulation(pre_step_agents)
        collected_global_uncertainty_after_mean = self.get_global_map_from_simulation(pre_step_agents).mean()

        #print(f"Pre steps uncertainty: {collected_individual_uncertainty_before_mean}")
        #print(f"Post steps uncertainty: {collected_individual_uncertainty_after_mean}")
        #print(f"Pre steps global uncertainty: {collected_global_uncertainty_before_mean:.4f}")
        #print(f"Post steps global uncertainty: {collected_global_uncertainty_after_mean:.4f}")

        ##### Reward
        individual_rewards = self._compute_individual_rewards(collected_individual_uncertainty_before_mean,
                                                              collected_individual_uncertainty_after_mean,
                                                              pre_step_agents)
        
        global_reward = self._compute_global_rewards(collected_global_uncertainty_before_mean, collected_global_uncertainty_after_mean, pre_step_agents)
        
        #"Step reward calculation: individual rewards = {individual_rewards}, global reward = {global_reward:.4f}")
        
        ##### Check for dying agents
        post_stepped_agents = self._active_episode_agents()
        ##### Random probability of a given agent dying after the step
        dying_agents = self._sample_dying_agents(post_stepped_agents)
        self._deactivate_agents(dying_agents)
        self._reward_sum_update(individual_rewards, global_reward, pre_step_agents)

        if not post_stepped_agents:
            end_cause = EndCause.ALL_AGENTS_INACTIVE
        elif status.has_ended:
            end_cause = EndCause.TIMEOUT

        # Filling the output tensordict for the step
        tensordict_out = self._cached_step_zero.clone()
        self._fill_observation(tensordict_out, self._observe_simulation())
        self._fill_rewards(tensordict_out, individual_rewards, global_reward, pre_step_agents)
        self._fill_done(tensordict_out, end_cause, dying_agents)
        self._fill_info(tensordict_out, end_cause, end_cause != EndCause.NONE)
        return tensordict_out

    def _reset_statistics(self):
        """
        Resets the statistics for a new episode.
        """
        self.episode_duration = 0
        self.individual_reward = [0.0 for _ in range(self.max_num_agents)]
        self.global_reward = 0
        self.max_global_reward = -math.inf
        self.max_individual_reward = -math.inf

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        ##### Reset the environment 
        num_active_agents = random.randint(self.min_num_agents, self.max_num_agents)
        self.episode_agents = [
            EpisodeAgentState(
                slot_index=i,
                name=f"drone{i}",
                exists=i < num_active_agents,
                active=i < num_active_agents,
            )
            for i in range(self.max_num_agents)
        ]

        self._reset_statistics()

        self.reset_simulation(self._simulation_configuration)

        active_agents = self._active_episode_agents()
        max_ready_steps = len(active_agents) * 10  # Arbitrary large number of steps to wait for drones to be ready
        ready_steps = 0
        while not self._all_active_drones_ready():
            status = self.step_simulation()
            ready_steps += 1
            if status.has_ended:
                raise RuntimeError("Simulation ended before all drones received initial telemetry")
            if ready_steps >= max_ready_steps:
                raise RuntimeError("Timed out waiting for initial telemetry for all drones")

        tensordict_out = self._cached_reset_zero.clone()
        self._fill_observation(tensordict_out, self._observe_simulation())
        self._fill_done(tensordict_out, EndCause.NONE, [])
        self._fill_info(tensordict_out, EndCause.NONE, False)
        return tensordict_out

    def _all_active_drones_ready(self) -> bool:
        """Return True once every active agent has received initial telemetry."""
        return all(
            getattr(self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol, "ready", False)
            for agent in self._active_episode_agents()
        )
    
    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.reset(seed=seed)

    def _apply_actions(self, actions: torch.Tensor) -> None:
        active_agents = self._active_episode_agents()
        actions_cpu = actions.detach().cpu()

        for agent in active_agents:            
            agent_node = self.simulator.get_node(agent.node_id)
            protocol = agent_node.protocol_encapsulator.protocol
            flag = protocol.mobility_command_buffer['flag']
            
            ##### No not apply action if there is no Flag for that
            if flag == FlagMessage.NONE.value:
                continue 
            
            # Convert the int to a list of 2 floats - x and y
            idx = int(actions_cpu[agent.slot_index, 0].item())
            row, col = idx//self.action_map_size, idx%self.action_map_size
            # 0.5 to help approximation to land in the right cell in the protocol
            x = (row + 0.5)/self.action_map_size
            y = (col + 0.5)/self.action_map_size
            action = [x,y]

            mask = self._compute_valid_action_mask(agent)
            assert mask[idx], f"Agent {agent.slot_index} picked invalid action {idx} despite masking!"
            protocol.mobility_command(action, self.action_map_size)

            ##### Destination in the map coordinates
            destination_x = protocol.drone_position[0] + (x - 0.5)*self.action_map_size*self.distance_between_cells
            destination_y = protocol.drone_position[1] + (y - 0.5)*self.action_map_size*self.distance_between_cells

            self.immediate_reward_from_action[agent.slot_index] = -self.get_immediate_distance_penalty(agent, destination_x, destination_y)

    def _sample_dying_agents(self, stepped_agents: list[EpisodeAgentState]) -> list[EpisodeAgentState]:
        if self.agent_death_probability <= 0.0:
            return []
        return [
            agent for agent in stepped_agents
            if random.random() < self.agent_death_probability
        ]

    def _deactivate_agents(self, agents: list[EpisodeAgentState]) -> None:
        for agent in agents:
            if not agent.active or agent.node_id is None:
                continue
            protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
            protocol.die()
            agent.active = False

    def get_flag_from_simulation(self, agent: EpisodeAgentState) -> int:
        protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
        return protocol.mobility_command_buffer['flag']       

    def get_individual_patch_map_from_simulation(self, agent: EpisodeAgentState, observation_size: float) -> np.ndarray:
        protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
        return protocol.get_patched_map(observation_size)
        
    def get_individual_distances_from_map(self, agent:EpisodeAgentState, observation_size: float) -> np.ndarray:
        protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
        return protocol.get_spatial_distance_map(observation_size)
    
    def get_individual_agent_uncertainty_from_simulation(self, agent: EpisodeAgentState) -> float:
        protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
        mean, std = protocol.get_mean_and_std_deviation_uncertainty()
        return [mean, std]
    
    def get_individual_position_from_simulation(self, agent: EpisodeAgentState) -> np.ndarray:
        protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
        return protocol.get_normalized_drone_position()
    
    def _compute_valid_action_mask(self, agent: EpisodeAgentState) -> np.ndarray:
        """
        Return a (M*M,) bool array — True for actions whose target cell falls
        inside the map given this agent's current position. Mirrors the same
        geometry the protocol uses in `mobility_command` (cell-centered xy).
        """
        A = self.action_map_size
        protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
        current_x_cell, current_y_cell = protocol.get_current_cell()  

        mask = np.zeros(A * A, dtype=bool)
        for idx in range(A * A):
            row, col = idx // A, idx % A
            x = (row + 0.5) / A
            y = (col + 0.5) / A
            target_row = int(current_x_cell + (x - 0.5) * A)
            target_col = int(current_y_cell + (y - 0.5) * A)
            if 0 <= target_row < self.map_width and 0 <= target_col < self.map_height:
                mask[idx] = True
            if target_row == current_x_cell and target_col == current_y_cell:
                mask[idx] = False
        return mask

    def get_estimated_positions_and_time_from_simulation(self, agent: EpisodeAgentState) -> np.ndarray:
        protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
        
        return protocol.get_estimated_drone_destinations(self.NORMALIZE_TIME)

    def get_global_positions_from_simulation(self, agents: list[EpisodeAgentState]) -> list[np.ndarray]:
        positions = []
        for agent in agents:
            if agent.active is False:
                positions.append(np.array([-1.0, -1.0]))  # Placeholder for inactive agents
                continue
            protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
            positions.append(protocol.drone_position[:2])
        return positions
    
    def get_global_distance_penalty(self, agents: list[EpisodeAgentState]) -> float:
        positions = self.get_global_positions_from_simulation(agents)
        
        if len(positions) < 2:
            return 0.0
        pos_arr = np.array(positions)
        # pdist computes only the unique pairs
        dists = pdist(pos_arr)

        valid_dists = dists[dists < self.communication_range]
        penalty = np.sum(1 - valid_dists / self.communication_range)
        return penalty 
    
    def get_individual_uncertainty_from_simulation(self, agents: list[EpisodeAgentState]) -> list[float]:
        agents_uncertainty = []
        for agent in agents:
            if agent.active is False:
                continue
            protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
            mean, _ = protocol.get_mean_and_std_deviation_uncertainty()
            agents_uncertainty.append(mean)
        return agents_uncertainty
    
    
    def get_number_of_cells_with_high_uncertainty(self, agents: list[EpisodeAgentState]) -> list[int]:
        """ Computes the number of high uncertainty cells for each agent and globally.
        So it returns a list of integers for each agent: the first element with the number of cells with extreme high uncertainty (>=1.0)
        and the second one with the number of cells with high uncertainty (>=0.75). And the values for the global
        """
        
        individual_maps = []
        individual_high_cells = []
        for agent in agents:
            if agent.active is False:
                continue
            protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
            individual_map = protocol.map[:, :, 0] 

            individual_extreme_high = int(np.sum(individual_map >= 1.0))
            individual_high = int(np.sum(individual_map >= 0.75)) - individual_extreme_high
            individual_high_cells.append((individual_extreme_high, individual_high))

            individual_maps.append(protocol.map[:, :, 0])  # shape: (map_width, map_height)
             
        if not individual_maps:
            # No active agents — return a fully uncertain map. The actual values
            # don't matter because the episode is terminal, but the shape does.
            return np.ones((self.map_width, self.map_height), dtype=np.float32)
        
        ##### Take the lowest uncertainty value for each cell across all drones
        global_map = np.min(individual_maps, axis=0)

        global_extreme_high = int(np.sum(global_map >= 1.0))
        global_high = int(np.sum(global_map >= 0.75)) - global_extreme_high

        return [individual_high_cells, global_extreme_high, global_high]
    
    
    def get_immediate_distance_penalty(self, agent: EpisodeAgentState, destination_x: float, destination_y: float) -> float:
        total_penalty = 0.0

        agent_node = self.simulator.get_node(agent.node_id)
        protocol = agent_node.protocol_encapsulator.protocol

        for agent_id, agent_state in enumerate(protocol.drone_states):
            if agent_id == agent.node_id:
                continue  # Skip self            

            time_since_last_update = self.simulator._current_timestamp - agent_state['time_of_last_update']
            #print(f"Current time: {self.simulator._current_timestamp}, Agent {agent_id} last update: {agent_state['time_of_last_update']}")
            #print(f"Agent node: {agent.node_id} comparing with {agent_id} with time: {time_since_last_update}")
            
            if time_since_last_update > self.NORMALIZE_TIME:
                continue  # Skip if the information about the other agent is too old to be relevant

            other_x, other_y= agent_state['position'][:2]      
            distance_to_other = math.sqrt((destination_x - other_x) ** 2 + (destination_y - other_y) ** 2)
            if distance_to_other > self.communication_range:
                continue  # It is far way to consider a penalty
            norm_distance = distance_to_other / (self.action_map_size * self.distance_between_cells)
            
            total_penalty += 2*max(0, 1 - norm_distance)  # Closer means higher penalty, with a hard cutoff at the action map's diagonal distance
        
        #print(f"------------Total penalty of drone {agent.node_id} is: {total_penalty:.4f}")
        return total_penalty
    
    def get_global_map_from_simulation(self, agents: list[EpisodeAgentState]) -> np.ndarray:
        individual_maps = []
        for agent in agents:
            if agent.active is False:
                continue
            protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
            individual_maps.append(protocol.map[:, :, 0])  # shape: (map_width, map_height)
        
        if not individual_maps:
            # No active agents — return a fully uncertain map. The actual values
            # don't matter because the episode is terminal, but the shape does.
            return np.ones((self.map_width, self.map_height), dtype=np.float32)
        
        ##### Take the lowest uncertainty value for each cell across all drones
        return np.min(individual_maps, axis=0)

    def get_normalized_global_map_from_simulation(self, agents: list[EpisodeAgentState]) -> np.ndarray:
        global_map = self.get_global_map_from_simulation(agents)
        mean_uncertainty = np.mean(global_map)
        std_uncertainty = np.std(global_map)

        #normalized_global_map = (global_map - mean_uncertainty) / (std_uncertainty + 1e-4)
        normalized_global_map = (global_map - mean_uncertainty) 
        return np.clip(normalized_global_map, -5.0, 5.0)
    
    def get_normalized_global_uncertainty(self, agents: list[EpisodeAgentState]):
        global_map = self.get_global_map_from_simulation(agents)
        mean_uncertainty = np.clip(np.mean(global_map), 0.0, 5.0)
        std_uncertainty = np.clip(np.std(global_map), 0.0, 5.0)
        
        return [mean_uncertainty, std_uncertainty]


    def _update_stall(self, collected_before: list[bool], collected_after: list[bool]) -> None:
        if sum(collected_after) > sum(collected_before):
            self.stall_duration = 0
        else:
            self.stall_duration += self.algorithm_iteration_interval

        self.episode_duration += 1

    def _compute_individual_rewards(self, uncertainty_before, uncertainty_after, stepped_agents):
        rewards = {}
        
        for agent, u_before, u_after in zip(stepped_agents,uncertainty_before, uncertainty_after):
            ### Positive reward from reducing uncertainty
            reward_1 = 1000*(u_before - u_after)
            #print(f"u before: {u_before}. u after {u_after}")            

            ### reward for distance penalty:
            reward_2 = self.immediate_reward_from_action[agent.slot_index]
            # Reset this variable. It will be updated when the agent takes another action
            self.immediate_reward_from_action[agent.slot_index] = 0.0
            #print(f"Reward 1: {reward_1:.4f}")
            #print(f"Reward 2: {reward_2:.4f}")

            rewards[agent.slot_index] = reward_1 + reward_2
            #print(f"The final reward of agent {agent.slot_index} is {reward}")
        return rewards   

    def _compute_global_rewards(self, uncertainty_before: float, uncertainty_after: float, stepped_agents: list[EpisodeAgentState]) -> float:
        """Return the global reward based on the change in global uncertainty."""
        global_1 = 1000*(uncertainty_before - uncertainty_after)
        #print(f"Global 1: {global_1:.4f}")
        
        global_2 = -self.get_global_distance_penalty(stepped_agents)
        #print(f"Global 2 {global_2:.4f}")
        return global_1 + global_2


    def _reward_sum_update(self, individual_rewards: list[float], global_reward: float, stepped_agents: list[EpisodeAgentState]) -> None:
        if stepped_agents:
            self.global_reward += global_reward
            self.max_global_reward = max(self.max_global_reward, global_reward)

            for agent in stepped_agents:
                self.individual_reward[agent.slot_index] += individual_rewards[agent.slot_index]
                self.max_individual_reward = max(self.max_individual_reward, self.individual_reward[agent.slot_index])            

    def _observe_simulation(self) -> dict:
        """
        Extracts information from the simulation to form the observation for each agent. 
        Each agent's and global observations are computed based on the defined in the build_specs method. 
        Returns:
            dict: A dictionary containing observations for each agent.
        """
        existing_agents = self._existing_episode_agents()
        if not existing_agents:
            return {}

        individual_state = {}
        for agent_index, agent in enumerate(existing_agents):
            large_agent_patch_map = np.stack((self.get_individual_patch_map_from_simulation(agent, self.map_width), self.get_individual_distances_from_map(agent, self.map_width)), axis = 0)
            small_agent_patch_map = np.stack((self.get_individual_patch_map_from_simulation(agent, self.action_map_size), self.get_individual_distances_from_map(agent, self.action_map_size)), axis = 0)
            agent_uncertainty = self.get_individual_agent_uncertainty_from_simulation(agent)
            agent_position = self.get_individual_position_from_simulation(agent)
            estimated_positions_and_time = self.get_estimated_positions_and_time_from_simulation(agent)
            flag = self.get_flag_from_simulation(agent)
            
            individual_state[agent.name] = {
                "large_map_patch": large_agent_patch_map,
                "small_map_patch": small_agent_patch_map,
                "individual_map_uncertainty": agent_uncertainty,
                "position": agent_position,
                "estimated_positions_and_time": estimated_positions_and_time,
                "encounter_flag": flag,
                "valid_actions": self._compute_valid_action_mask(agent),
            }
        global_state = {}
        global_map = self.get_normalized_global_map_from_simulation(existing_agents)
        global_uncertainty = self.get_normalized_global_uncertainty(existing_agents)
        global_positions = self.get_global_positions_from_simulation(existing_agents)
        global_active = [agent.active for agent in self.episode_agents]

        global_state["full_map"] = global_map
        global_state["global_map_uncertainty"] = global_uncertainty
        global_state["all_positions"] = global_positions
        global_state["all_active"] = global_active

        return {
            "individual": individual_state,
            "global": global_state
        }
 

    def _fill_observation(
        self,
        td: TensorDictBase,
        observation_dict: dict,
        ) -> None:
        """Fill `td` with per-agent observations, the global (critic) state, and the agent mask."""

        agents_td = td.get("agents")
        agent_obs_td = agents_td.get("observation")

        large_agent_map_patch            = agent_obs_td.get("large_map_patch")
        small_agent_map_patch            = agent_obs_td.get("small_map_patch")
        agent_individual_map_uncertainty = agent_obs_td.get("individual_map_uncertainty")
        agent_position                   = agent_obs_td.get("position")
        estimated_positions_and_time_t   = agent_obs_td.get("estimated_positions_and_time") 
        agent_encounter_flag             = agent_obs_td.get("encounter_flag")
        agent_valid_actions              = agent_obs_td.get("valid_actions")


        global_td               = td.get("global_state")
        global_full_map         = global_td.get("full_map")
        global_map_uncertainty  = global_td.get("global_map_uncertainty")
        global_all_positions    = global_td.get("all_positions")
        global_all_active       = global_td.get("all_active")

        mask = agents_td.get("mask")

        # reset to defaults  
        large_agent_map_patch.zero_()
        small_agent_map_patch.zero_()
        agent_individual_map_uncertainty.zero_()
        agent_position.zero_()
        estimated_positions_and_time_t.fill_(-1.0)
        agent_encounter_flag.zero_()
        agent_valid_actions.zero_()

        global_full_map.zero_()
        global_map_uncertainty.zero_()
        global_all_positions.fill_(-1.0)
        global_all_active.zero_()
        mask.zero_()

        if not observation_dict:
            return

        individual_obs = observation_dict.get("individual", {})
        global_obs     = observation_dict.get("global", {})


        M = self.observation_map_size
        A = self.action_map_size
        
        # agent observations 
        for agent in self._existing_episode_agents():
            slot = agent.slot_index
            obs = individual_obs.get(agent.name)
            if obs is None:
                continue

            # map_patch -> take uncertainty channel, pad to (M, M)
            raw_patch = obs["large_map_patch"]
            assert raw_patch.shape == (2, M, M), (
                f"Expected patch shape (2, {M}, {M}) from protocol, got {raw_patch.shape}"
            )
            large_agent_map_patch[slot] = torch.as_tensor(
                obs["large_map_patch"], device=self.device, dtype=large_agent_map_patch.dtype
            )

            raw_patch = obs["small_map_patch"]            
            assert raw_patch.shape == (2, A, A), (
                f"Expected patch shape (2, {A}, {A}) from protocol, got {raw_patch.shape}"
            )
            small_agent_map_patch[slot] = torch.as_tensor(
                obs["small_map_patch"], device=self.device, dtype=small_agent_map_patch.dtype
            )

            # scalar uncertainty (normalized)
            agent_individual_map_uncertainty[slot] = torch.as_tensor(
                obs["individual_map_uncertainty"], 
                device=self.device, 
                dtype=agent_individual_map_uncertainty.dtype
            )

            # own (x, y) normalized
            agent_position[slot] = torch.as_tensor(
                (obs["position"]), device=self.device, dtype=agent_position.dtype
            )

            # estimated_positions_and_time 
            estimates = obs.get("estimated_positions_and_time")  # shape (max_num_agents-1, 3) - x, y, normalized time
            if estimates is not None:
                estimated_positions_and_time_t[slot, :] = torch.as_tensor(
                    estimates,
                    device=self.device,
                    dtype=estimated_positions_and_time_t.dtype
                )
            
            # encounter_flag (int) -> one-hot 
            agent_encounter_flag[slot, 0] = bool(obs["encounter_flag"])

            agent_valid_actions[slot] = torch.as_tensor(
                obs["valid_actions"], device=self.device, dtype=agent_valid_actions.dtype
            )


        ##### global / critic observation 
        half_w = self.map_width  * self.distance_between_cells / 2.0
        half_h = self.map_height * self.distance_between_cells / 2.0
        

        def _normalize_xy(p):
            # world [-half_w, +half_w] x [-half_h, +half_h]  ->  [0, 1] x [0, 1]
            return np.array([
                (float(p[0]) + half_w) / (2.0 * half_w),
                (float(p[1]) + half_h) / (2.0 * half_h),
            ], dtype=np.float32)
        

        if "full_map" in global_obs:
            global_full_map.copy_(
                torch.as_tensor(global_obs["full_map"], device=self.device, dtype=global_full_map.dtype)
            )

        if "global_map_uncertainty" in global_obs:
            global_map_uncertainty = torch.as_tensor(
                global_obs["global_map_uncertainty"],
                device=self.device,
                dtype=torch.float32  
            )

        for i, pos in enumerate(global_obs.get("all_positions", []) or []):
            if i >= self.max_num_agents:
                break
            pos_arr = np.asarray(pos, dtype=np.float32)
            # `_observe_simulation` already uses [-1, -1] as the placeholder for inactive
            if pos_arr.shape == (2,) and not (pos_arr[0] == -1.0 and pos_arr[1] == -1.0):
                global_all_positions[i] = torch.as_tensor(
                    _normalize_xy(pos_arr), device=self.device, dtype=global_all_positions.dtype
                )

        if "all_active" in global_obs and global_obs["all_active"] is not None:
            global_all_active.copy_(
                torch.as_tensor(global_obs["all_active"], device=self.device, dtype=global_all_active.dtype)
            )

        ##### mask: True only for currently active slots 
        active_agents = self._active_episode_agents()
        if active_agents:
            active_slots = self._agent_slot_tensor(active_agents)
            mask.index_fill_(0, active_slots, True)


    def _fill_rewards(
        self,
        td: TensorDictBase,
        individual_rewards: list[float],
        global_reward: float,
        rewarded_agents: list[EpisodeAgentState],
    ) -> None:
        """
        Write per-agent and global rewards into the dictionary.
        """
        agents_reward_t = td.get(("agents", "reward"))
        agents_reward_t.zero_()

        global_reward_t = td.get("global_reward")
        global_reward_t.zero_()

        for agent in rewarded_agents:
            agents_reward_t[agent.slot_index, 0] = float(individual_rewards[agent.slot_index])

        global_reward_t[0] = float(global_reward)

    def _fill_done(
        self,
        td: TensorDictBase,
        end_cause: EndCause,
        dying_agents: list[EpisodeAgentState],
    ) -> None:
        done = td.get(("agents", "done"))
        terminated = td.get(("agents", "terminated"))
        truncated = td.get(("agents", "truncated"))
        done.zero_()
        terminated.zero_()
        truncated.zero_()

        active_agents = self._active_episode_agents()
        dead_agents = [
            agent for agent in self._inactive_existing_episode_agents()
            if agent not in dying_agents
        ]
        inactive_agents = [agent for agent in self.episode_agents if not agent.exists]
        active_slots = self._agent_slot_tensor(active_agents)
        dying_slots = self._agent_slot_tensor(dying_agents)
        dead_slots = self._agent_slot_tensor(dead_agents)
        inactive_slots = self._agent_slot_tensor(inactive_agents)

        if end_cause != EndCause.NONE and active_slots.numel() > 0:
            terminated.index_fill_(0, active_slots, True)
            done.index_fill_(0, active_slots, True)
        if dying_slots.numel() > 0:
            terminated.index_fill_(0, dying_slots, True)
            done.index_fill_(0, dying_slots, True)

        # Previously dead real slots and padded slots are bookkeeping only.
        truncated.index_fill_(0, dead_slots, True)
        done.index_fill_(0, dead_slots, True)
        truncated.index_fill_(0, inactive_slots, True)
        done.index_fill_(0, inactive_slots, True)

        episode_done = end_cause != EndCause.NONE
        td.set("done", torch.tensor([episode_done], device=self.device))
        td.set("terminated", torch.tensor([episode_done], device=self.device))
        td.set("truncated", torch.tensor([False], device=self.device))

    def _fill_info(self, td: TensorDictBase, end_cause: EndCause, ended: bool) -> None:
        info_td = td.get(("agents", "info"))
        for key in self._all_info_keys:
            info_td.get(key).zero_()

        existing_agents = self._existing_episode_agents()
        if not ended or not existing_agents:
            return
        existing_slots = self._agent_slot_tensor(existing_agents)

        metrics = {
            "max_reward": self.max_global_reward,
            "reward": self.global_reward,
            "episode_duration": float(self.episode_duration),
            "cause": float(end_cause.value),
            "num_agents": float(len(existing_agents))
        }

        for key, value in metrics.items():
            info_td.get(key).index_fill_(0, existing_slots, value)

    def close(self, *, raise_if_closed: bool = True):
        super().close(raise_if_closed=raise_if_closed)
        self.finalize_simulation()
