"""
REPLAY BUFFER

Only used when offline trainning is required, like DQN.
"""
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from typing import Optional
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        centralized_training: bool = False,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),  
            batch_size=batch_size,
        )
        self.centralized_training = centralized_training
        # if centrilized_training is True, we will store the global state in the buffer, otherwise we will store the local state of each agent


    def load_transitions(self, transitions: TensorDict):
        # `transitions` is a STACKED TD with batch_size=[K] (K rows from one collect).
        # `extend` inserts each row separately -> per-transition sampling works.        
        if transitions is None or transitions.batch_size[0] == 0:
            return 
        
        self.buffer.extend(transitions)
              
    def sample(self, batch_size=None) -> TensorDict:
        n = batch_size or self.batch_size
        return self.buffer.sample(n)
    
    def __len__(self):
        return len(self.buffer)

    