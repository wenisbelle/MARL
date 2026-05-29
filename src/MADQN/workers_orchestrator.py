"""
WORKERS ORCHESTRATOR
--------------------
Trainer-side handle for a pool of worker processes.

Lifecycle in sync mode:
    1. __init__   -> spawn workers 
    2. set_weights(state_dict)
    3. broadcast()
    4. resume()                     # workers wake up and start collecting
    5. collect(min_new_transitions) # blocks until threshold; sends PAUSE at end
    6. ... train ...
    7. set_weights + broadcast + resume   (back to step 5)

Lifecycle in async mode:
    No PAUSE / CONTINUE — workers free-run from the moment they receive their
    first weight broadcast. The trainer drains transitions opportunistically.
"""

import multiprocessing as mp
import time
from queue import Empty

from worker import _worker_loop


class WorkersOrchestrator:
    def __init__(
        self,
        num_workers: int,
        env_fn,
        policy_fn,
        replay_buffer,
        steps_per_batch: int = 200,
        base_seed: int = 0,
        sync: bool = False,
        new_batch_new_simulation: bool = True,
    ):
        # "spawn" is required cross-platform 
        ctx = mp.get_context("spawn")
        self.transition_queue = ctx.Queue(maxsize=4 * num_workers)
        # One weight queue per worker — broadcasting via a shared queue would
        # let the first worker consume the message and leave the others stale.
        self.weight_queues  = [ctx.Queue(maxsize=2) for _ in range(num_workers)]
        self.control_queues = [ctx.Queue() for _ in range(num_workers)]

        self.sync = sync
        self.policy_fn = policy_fn
        self._pending_state_dict = None  # set by set_weights, consumed by broadcast

        self.replay_buffer = replay_buffer
        self.num_workers = num_workers
        self.new_batch_new_simulation = new_batch_new_simulation

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
                    self.new_batch_new_simulation,
                ),
                daemon=False, # in trainning it should be False
            )
            p.start()
            self.workers.append(p)


    def pause(self):
        """Tell workers to stop producing transitions and discard in-flight work."""
        for q in self.control_queues:
            try: q.put_nowait("PAUSE")
            except Exception: pass

    def resume(self):
        """Wake paused workers up. Workers will drain the latest weights, then run."""
        for q in self.control_queues:
            try: q.put_nowait("CONTINUE")
            except Exception: pass


    def set_weights(self, state_dict):
        """Stage a new state dict for the next broadcast. CPU-side, detached."""
        self._pending_state_dict = {
            k: v.detach().cpu().clone() for k, v in state_dict.items()
        }

    def broadcast(self):
        """Push the pending state dict to every worker. No-op if none staged."""
        if self._pending_state_dict is None:
            return
        for q in self.weight_queues:
            try:
                q.put_nowait(self._pending_state_dict)
            except Exception:
                # Queue full -> the worker hasn't drained its previous weights
                # yet. Safe to drop in async mode (latest wins); in sync mode
                # this shouldn't happen because workers drain on every iter.
                pass

    def collect(self, min_new_transitions: int = 1000, timeout: float = 300.0) -> int:
        """
        Drain the transition queue into the replay buffers until either:
          - at least `min_new_transitions` *agent* transitions have arrived, or
          - `timeout` seconds have elapsed.

        In sync mode, sends PAUSE to all workers at the end so they stop
        producing before the trainer starts its update.

        Returns the number of new agent transitions ingested.
        """
        new_count = 0
        deadline = time.time() + timeout
        while new_count < min_new_transitions and time.time() < deadline:
            try:
                agent_td = self.transition_queue.get(timeout=0.5)
            except Empty:
                continue
            
            if agent_td is not None:
                self.replay_buffer.load_transitions(agent_td)
                new_count += agent_td.batch_size[0]
            #print(f"Collected {new_count} transitions so far...")

        if self.sync:
            self.pause()
            print(f"Workers paused")

        return new_count


    def shutdown(self, timeout: float = 5.0):
        # If we're paused, workers are blocking on control_queue.get() — STOP
        # is delivered that way too, so this works in either state.
        for q in self.control_queues:
            try: q.put_nowait("STOP")
            except Exception: pass
        for p in self.workers:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
        for q in [self.transition_queue, *self.weight_queues, *self.control_queues]:
            q.close()

    def __enter__(self):  return self
    def __exit__(self, *args):  self.shutdown()