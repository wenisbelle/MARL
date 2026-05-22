import multiprocessing as mp
import time
import torch
from queue import Empty
from tensordict import TensorDict

from worker import _worker_loop

class WorkersOrchestrator:
    def __init__(
        self,
        num_workers: int,
        env_fn,
        policy_fn,
        agent_buffer,         # agent TensorDictReplayBuffer
        global_buffer,        # global TensorDictReplayBuffer
        steps_per_batch: int = 200,
        base_seed: int = 0,
        sync: bool = True, # sync or async
    ):
        ctx = mp.get_context("spawn")   # safest cross-platform; required on macOS/Windows
        self.transition_queue = ctx.Queue(maxsize=4 * num_workers)
        # need to provide a list because the first worker to grab the message would consume it, leaving the others without weights
        self.weight_queues   = [ctx.Queue(maxsize=2) for _ in range(num_workers)]
        self.control_queues  = [ctx.Queue() for _ in range(num_workers)]
    
        self.sync = sync
        self.policy_fn = policy_fn
        self.current_state_dict = None  
        self._update_weights = True  # whether the workers are currently running with the same weights   

        self.agent_buffer = agent_buffer
        self.global_buffer = global_buffer
        self.num_workers = num_workers

        self.workers = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(
                    i,
                    base_seed + i * 1000,
                    env_fn,
                    self.policy_fn,
                    steps_per_batch,
                    self.transition_queue,
                    self.weight_queues[i],
                    self.control_queues[i],
                    self.sync,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def collect(self, min_new_transitions: int = 1000, timeout: float = 30.0):
        """Drain the transition queue until enough new data is in the buffers."""
        new_count = 0
        deadline = time.time() + timeout
        if self.sync:
            if self.sync:
                pass 
        while new_count < min_new_transitions and time.time() < deadline:
            # I keep sending this 
            try:
                kind, td = self.transition_queue.get(timeout=0.5)
            except Empty:
                continue
            if kind == "agent":
                self.agent_buffer.extend(td)
                new_count += td.batch_size[0]
            elif kind == "global":
                self.global_buffer.extend(td)

        if self.sync:
            self._can_update.set()     # allow again

        return new_count

    def broadcast(self):
        """Push the pending state dict to all workers."""
        if self._pending_state_dict is None:
            return
        for q in self.weight_queues:
            try:
                q.put_nowait(self._pending_state_dict)
            except Exception:
                pass

    def shutdown(self, timeout: float = 5.0):
        for q in self.control_queues:
            try: q.put_nowait("STOP")
            except Exception: pass
        for p in self.workers:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
        # close queues
        for q in [self.transition_queue, *self.weight_queues, *self.control_queues]:
            q.close()

    def set_weights(self, state_dict):
        """Stage a new state dict for the next broadcast."""
        self._pending_state_dict = {
            k: v.detach().cpu().clone() for k, v in state_dict.items()
    }

    def __enter__(self):  return self
    def __exit__(self, *args):  self.shutdown()