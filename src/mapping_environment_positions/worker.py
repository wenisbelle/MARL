"""
WORKER
Ther worker is responsible for running one instance of the environment and collecting the transitions.
These transitions are sent back to the main process via a multiprocessing.Queue.
The worker also listens for new policy weights and control signals (like "STOP") from the main process.
"""

import multiprocessing as mp
import time
import torch
from queue import Empty
from tensordict import TensorDict
from torchrl.envs.utils import step_mdp

from .simulation_orchestrator import AsyncMARLOrchestrator


def _worker_loop(
    worker_id: int,
    seed: int,
    env_fn,               
    policy_fn,             
    steps_per_batch: int,
    transition_queue: mp.Queue,
    weight_queue: mp.Queue,
    control_queue: mp.Queue,
    sync: bool = True, # sync or async
):
    """One worker process: env + orchestrator + local policy."""
    # Build everything inside the worker — the env/simulator does not pickle.
    torch.manual_seed(seed)
    import random; random.seed(seed)
    import numpy as np; np.random.seed(seed)

    env = env_fn()
    policy = policy_fn()
    policy.eval()
    SYNC = sync
    

    # no training inside the worker; just inference so I need to add the no_grad to the policy callable
    def policy_callable(per_agent_obs_td):
        with torch.no_grad(): 
            return policy(per_agent_obs_td)

    orch = AsyncMARLOrchestrator(env, policy_callable)
    td = orch.reset()

    while True:
        # shutdown?
        try:
            if control_queue.get_nowait() == "STOP":
                break
        except Empty:
            pass

        # new weights? drain everything, keep only the most recent.
        if SYNC:
            # Block until trainer sends the next policy version. Don't drain;
            # we want exactly one weight update per training iteration.
            new_state_dict = weight_queue.get()
            policy.load_state_dict(new_state_dict)
        else:
            # Async: drain to latest, skip if nothing new.
            new_state_dict = None
            try:
                while True:
                    new_state_dict = weight_queue.get_nowait()
            except Empty:
                pass
            if new_state_dict is not None:
                policy.load_state_dict(new_state_dict)
                
       #  run a chunk of simulation.
        for _ in range(steps_per_batch):
            td = orch.step(td)
            if td["next", "done"].item():
                td = orch.reset()
            else:
                td = step_mdp(td)

        # drain the orchestrator's transitions and ship them.
        if orch.agent_transitions:
            agent_td = torch.stack(orch.agent_transitions).contiguous()
            transition_queue.put(("agent", agent_td))
            orch.agent_transitions.clear()

        if orch.global_transitions:
            global_td = torch.stack(orch.global_transitions).contiguous()
            transition_queue.put(("global", global_td))
            orch.global_transitions.clear()

    env.close()