import pickle
import numpy as np

class SimulationRecorder:
    def __init__(self, simulation, num_drones, map_width, map_height, distance_between_cells):
        self.sim = simulation
        self.N, self.W, self.H, self.d = num_drones, map_width, map_height, distance_between_cells
        self.times = []
        self.maps      = {i: [] for i in range(num_drones)}   # own uncertainty map
        self.true_pos  = {i: [] for i in range(num_drones)}   # own (cx, cy) in cell coords
        self.beliefs   = {i: [] for i in range(num_drones)}   # (N, 3): cx, cy, age
        self.decisions = {}
        self.global_uncertainty = []
        self.unvisited = []
        self.masks = {i: [] for i in range(num_drones)}   # (A*A,) action mask per drone
        self.A = None                                     # action grid size, set on first snapshot

    def _to_cell(self, wx, wy):
        return (wx + self.W * self.d / 2) / self.d, (wy + self.H * self.d / 2) / self.d

    def snapshot(self):
        t = self.sim._current_timestamp
        self.times.append(t)
        unc_maps, visited = [], []
        for i in range(self.N):
            p = self.sim.get_node(i).protocol_encapsulator.protocol
            umap = p.map[:, :, 0].astype(np.float32)
            self.maps[i].append(umap.copy())
            unc_maps.append(umap)
            visited.append(p.is_cell_visited > 0)
            if p.drone_position is not None:
                self.true_pos[i].append(self._to_cell(p.drone_position[0], p.drone_position[1]))
            else:
                self.true_pos[i].append((np.nan, np.nan))
    
            # NEW — record the exact mask the protocol would use right now
            A = p.ACTION_MAP_SIZE
            if self.A is None:
                self.A = A
            if p.drone_position is not None:
                self.masks[i].append(p._compute_valid_action_mask().cpu().numpy().astype(bool))
            else:
                self.masks[i].append(np.zeros(A * A, dtype=bool))
    
            row = np.full((self.N, 3), np.nan, dtype=np.float32)
            for j, s in enumerate(p.drone_states):
                bx, by = self._to_cell(s["position"][0], s["position"][1])
                row[j] = (bx, by, t - s["time_of_last_update"])
            self.beliefs[i].append(row)
    
        global_map = np.min(unc_maps, axis=0)
        self.global_uncertainty.append(float(global_map.sum()))
        any_visited = np.any(visited, axis=0)
        self.unvisited.append(int(any_visited.size - any_visited.sum()))

    def save(self, path, results_aggregator):
        data = {
                "meta": {"num_drones": self.N, "map_width": self.W, "map_height": self.H,
                "times": np.array(self.times),
                "action_map_size": self.A,                              # NEW
                "global_uncertainty": np.array(self.global_uncertainty),
                "unvisited": np.array(self.unvisited)},
                "states": {i: {
                    "maps":          np.stack(self.maps[i]),
                    "true_pos_cell": np.array(self.true_pos[i], dtype=np.float32),
                    "beliefs":       np.stack(self.beliefs[i]),
                    "masks":         np.stack(self.masks[i]),                    # NEW (T, A*A) bool
                } for i in range(self.N)},
            "decisions": results_aggregator.get("decisions", {}),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)