from torchrl.envs.utils import check_env_specs
from torchrl.envs.transforms import TransformedEnv, StepCounter
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.utils import RandomPolicy   
from env.mapping_environment import MappingEnvironment, MappingEnvironmentConfig

env = MappingEnvironment(MappingEnvironmentConfig(
    max_num_agents=3, min_num_agents=3, max_episode_length=200,
    ))

check_env_specs(env)         # validates reset/step output match the specs
td = env.reset()
print(td)                    # eyeball the structure
td = env.rand_step(td)       # uses your action_spec to sample random actions
print(td)