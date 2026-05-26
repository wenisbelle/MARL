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

from simulation_orchestrator import AsyncMARLOrchestrator

def _drain_latest(q: mp.Queue):
    """Pop everything available on `q` non-blocking; return the last item or None."""
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except Empty:
        pass
    return latest

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
    paused = False


    while True:
        # shutdown?
        try:
            msg = control_queue.get_nowait()
            if msg == "STOP":
                env.close()
                return
            elif msg == "PAUSE":
                paused = True
            elif msg == "CONTINUE":
                paused = False
        except Empty:
            pass


        # If paused, discard old-policy work and block 
        if paused:
            # Drop unshipped transitions — they belong to the old policy.
            orch.agent_transitions.clear()
            orch.global_transitions.clear()
            # Also reset to avoid mixed-policy partial SMDP transitions.
            # Reset the whole simulation
            ##### Carefull here, maybe is not the best option
            td = orch.reset()
 
            # Block on the control queue until CONTINUE or STOP arrives.
            while True:
                m = control_queue.get()  # blocking
                if m == "CONTINUE":
                    paused = False
                    break
                if m == "STOP":
                    env.close()
                    return
                # ignore other messages while paused
            # fall through to the weight drain so we pick up new weights
            # broadcast during the pause window
 
        # Apply latest weights (non-blocking drain, keep last)
        latest = _drain_latest(weight_queue)
        if latest is not None:
            policy.load_state_dict(latest)
 
        # Run a simulation 
        for _ in range(steps_per_batch):
            td = orch.step(td)
            if td["next", "done"].item():
                td = orch.reset()
            else:
                td = step_mdp(td)
 
        agent_td = (
            torch.stack(orch.agent_transitions).contiguous()
            if orch.agent_transitions else None
        )
        global_td = (
            torch.stack(orch.global_transitions).contiguous()
            if orch.global_transitions else None
        )
        transition_queue.put((agent_td, global_td))
        orch.agent_transitions.clear()
        orch.global_transitions.clear()