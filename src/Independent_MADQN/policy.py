import multiprocessing as mp
import torch
from torch import nn

from replay_buffer import ReplayBuffer
from actor import Actor

class RandomPolicy(nn.Module):
    def __init__(self, n_actions: int = 400):   # M*M, e.g. 20*20
        super().__init__()
        self.n_actions = n_actions

    def forward(self, per_agent_obs_td):
        q_values = torch.rand(self.n_actions)     # (400,) fake Q-values
        return q_values.argmax(keepdim=True)      # (1,) int64 — highest-value index
    
class DQNPolicy(nn.Module):

    def __init__(self,
                 max_num_agents: int,
                 action_dim: int = 400,          
                 map_channels: int = 1,          
                 vector_feature_dim: int = 64,
                 hidden_dim: int = 256,
                 large_map_key: str = "large_map_patch",
                 small_map_key: str = "small_map_patch",
                 position_key: str = "position",
                 uncertainty_key: str = "individual_map_uncertainty",
                 estimated_positions_key: str = "estimated_positions_and_time",
                 eps_init = 1.0,
                 ):
        super().__init__()
        self.actor = Actor(
            max_num_agents=max_num_agents,
            action_dim=action_dim,
            map_channels=map_channels,
            vector_feature_dim=vector_feature_dim,
            hidden_dim=hidden_dim,
            large_map_key=large_map_key,
            small_map_key=small_map_key,
            position_key=position_key,
            uncertainty_key=uncertainty_key,
            estimated_positions_key=estimated_positions_key,
        )
        self.register_buffer("eps", torch.tensor(eps_init, dtype=torch.float32))

    def forward(self, per_agent_obs_td):
        valid_mask = per_agent_obs_td["valid_actions"].bool()    # (action_dim,)

        if torch.rand(()).item() < self.eps.item():
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            pick = torch.randint(0, valid_indices.numel(), (1,))
            
            return valid_indices[pick].to(torch.int64).reshape(1)

        with torch.no_grad():
            q_values = self.actor(per_agent_obs_td)
        return q_values.argmax(dim=-1, keepdim=True)

    def update_epsilon(self, new_eps):
        self.eps.fill_(new_eps)