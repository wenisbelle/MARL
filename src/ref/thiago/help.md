The Big Picture
The code wraps GrADySim inside a standard TorchRL environment interface. The training loop never speaks to GrADySim directly — it only sees a TorchRL EnvBase. The translation between "RL concepts" (observations, actions, rewards) and "simulation concepts" (nodes, positions, broadcasts) happens in the environment files.

How the Simulation is Created
The entry point is gradys_env.py:47 — reset_simulation().

At the start of every episode, _reset (data_collection.py:406) calls this, which does three things:

1. A SimulationBuilder is created, which is GrADySim's way of assembling a simulation before running it.

2. A custom GrADySHandler is injected (gradys_env.py:54). This is the critical synchronization mechanism — it sets _algorithm_iteration_finished = True and schedules itself to fire again after algorithm_iteration_interval seconds. This is how the environment knows when one "step" of simulation time has elapsed.

3. _build_simulation is called (data_collection.py:160), which adds:

A CommunicationHandler (defines radio range between nodes)
A MobilityHandler (moves nodes around the map)
A TimerHandler (lets protocols schedule their own timers)
N sensor nodes using SensorProtocol
M drone nodes using DroneProtocol
Each node gets a random position and a node_id stored in EpisodeAgentState.

The Two Protocols: What Lives Inside the Simulation
These are pure GrADySim objects — they react to simulator events.

SensorProtocol (protocols.py:12) is passive. It just waits. When a drone broadcasts a message and the sensor is within range, GrADySim delivers it via handle_packet, which flips has_collected = True. That's the entire data collection mechanic.

DroneProtocol (protocols.py:141) is active:

Every 0.1s it broadcasts an empty message (_collect_packets) — this is how sensors get "visited"
handle_telemetry receives the drone's current GPS position from the mobility engine and stores it in self.current_position
act() (protocols.py:156) is called externally (by the RL code), converts the agent's action into a GotoCoordsMobilityCommand and sends it to GrADySim
How a Step Works: Simulation Time vs. RL Time
This is the key part. GrADySim runs event-by-event (microseconds), but the RL agent acts every algorithm_iteration_interval seconds. The bridge is in step_simulation() (gradys_env.py:87):


_step() in data_collection.py:
  1. _apply_actions()  → calls DroneProtocol.act() for each active drone
  2. step_simulation() → runs the GrADySim event loop in a while-loop
                         until GrADySHandler fires and sets
                         _algorithm_iteration_finished = True
  3. read back state   → query simulator nodes for positions & has_collected
The while not self._algorithm_iteration_finished loop (gradys_env.py:94) is what "burns through" all the physics events (drone movements, broadcasts, packet deliveries) until the next RL decision point.

How Observations are Read from the Simulation
After each step, _observe_simulation() (data_collection.py:527) queries GrADySim directly:


# Read node positions from the simulator
self.simulator.get_node(sensor_id).position[:2]   # sensor coordinates
self.simulator.get_node(agent.node_id).position[:2] # drone coordinates

# Read protocol state
.protocol_encapsulator.protocol.has_collected     # was this sensor visited?
It then computes, for each drone, the K closest unvisited sensors and K closest other drones, normalizes their positions relative to the drone's position, and returns a dict that gets packed into tensors.

How Rewards are Computed
Rewards are purely based on sensor collection state read from the simulation:


reward = (sensors_collected_after - sensors_collected_before) × 10   # if new sensors collected
       = -(remaining_sensors / total_sensors)                         # otherwise (punishment)
This is in _compute_reward() (data_collection.py:508).

Summary Diagram

Training loop (algorithm.py)
        │
        │  batch of transitions
        ▼
DataCollector (collector.py)
        │
        │  calls env.step(actions)
        ▼
DataCollectionEnvironment._step()           ← TorchRL interface
        │
        ├─ _apply_actions()
        │       └─ DroneProtocol.act()      ← sends GotoCoords to GrADySim
        │
        ├─ step_simulation()                ← runs GrADySim event loop
        │       ├─ drones move (MobilityHandler)
        │       ├─ drones broadcast (CommunicationHandler)
        │       └─ sensors flip has_collected when reached
        │
        ├─ _observe_simulation()
        │       └─ simulator.get_node().position   ← read positions back
        │       └─ protocol.has_collected          ← read sensor state back
        │
        └─ returns tensordict (obs, reward, done, mask)
The simulation itself is a black box that advances physics. The environment is the translator that converts RL actions into simulation commands, advances time, and then reads the resulting state back out.