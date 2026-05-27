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
    new_batch_new_simulation: bool = True, # whether to reset the simulation at the start of each batch
):
    """One worker process: env + orchestrator + local policy."""
    # Build everything inside the worker — the env/simulator does not pickle.
    torch.manual_seed(seed)
    import random; random.seed(seed)
    import numpy as np; np.random.seed(seed)

    env = env_fn()
    policy = policy_fn()
    policy.eval()
    

    # no training inside the worker; just inference so I need to add the no_grad to the policy callable
    def policy_callable(per_agent_obs_td):
        with torch.no_grad(): 
            return policy(per_agent_obs_td)

    orch = AsyncMARLOrchestrator(env, policy_callable)
    td = orch.reset()


    while True:
        # shutdown?
        if new_batch_new_simulation:
            # give time to start get the message before keep going in the loop
            time.sleep(0.1) 
        try:
            msg = control_queue.get_nowait()
            print(f"[worker {worker_id}] got control msg: {msg!r}", flush=True)
        except Empty:
            msg = None

        if msg == "STOP":
                env.close()
                return

        if msg == "PAUSE":
            # Discard old-policy work + reset, so next chunk starts a clean episode.
            print(f"worker  >>> entering PAUSE block")
            orch.agent_transitions.clear()
            orch.global_transitions.clear()
            if new_batch_new_simulation:
                td = orch.reset()

            # Block until CONTINUE (or STOP). This consumes the CONTINUE that
            # arrived right after PAUSE — the race no longer matters.
            while True:
                m = control_queue.get()  # blocking
                print(f"[worker {worker_id}]     (paused) got: {m!r}", flush=True)
                if m == "CONTINUE":
                    break
                if m == "STOP":
                    env.close()
                    return
                # ignore any other control messages while paused

            # After unpausing, pick up the latest weights for the new iteration.
            latest = _drain_latest(weight_queue)
            if latest is not None:
                policy.load_state_dict(latest)
            continue  # restart the outer loop cleanly
            # If paused, discard old-policy work and block 
        
 
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