import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

class FitnessEvaluator:
    def __init__(self, map_width: int,
                 map_height: int,
                 distance_between_cells: int,
                 camera_angle:float,
                 distance_norm: float,
                 distance_between_drone_norm: float,
                 number_of_cells_x_y: int = 10):

        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells
        self.distance_norm = distance_norm
        self.distance_between_drone_norm = distance_between_drone_norm
        self.NUMBER_OF_CELLS_X_Y = number_of_cells_x_y

     

    def get_cells_visited_in_trajectory(self, drone_altitude: float, initial_cell: Tuple[int, int], final_cell: Tuple[int, int]) -> list:
        
        x0, y0 = initial_cell
        x1, y1 = final_cell

        radius_coverage = drone_altitude * np.tan(self.camera_angle)

        x_min, x_max = min(x0, x1) , max(x0, x1) 
        y_min, y_max = min(y0, y1) , max(y0, y1) 

        X, Y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))

        ### Equation distance between point and line
        A = y1 - y0
        B = x0 - x1
        C = x1 * y0 - y1 * x0
        denominator = np.sqrt(A**2 + B**2)

        if denominator == 0:
            # Drone is evaluating its exact current cell
            return [(x0, y0)]
        
        distances = np.abs(A * X + B * Y + C) / denominator
        map_size_d = distances * self.distance_between_cells

        # Filter cells within the camera radius
        mask = map_size_d <= radius_coverage
        
        # Extract the valid coordinates
        valid_x = X[mask]
        valid_y = Y[mask]
        
        # Filter out coordinates that fall outside the actual map boundaries
        bounds_mask = (valid_x >= 0) & (valid_x < self.map_width) & (valid_y >= 0) & (valid_y < self.map_height)
        valid_x = valid_x[bounds_mask]
        valid_y = valid_y[bounds_mask]
        
        # Create the final list of absolute coordinates
        cells_within_trajectory = list(zip(valid_x, valid_y))

        return cells_within_trajectory       

    def cells_priority(self, map_data: np.array, drone_position: Tuple[float, float, float], map_center_offset: float) -> list:
        """
        Batches inputs and uses the Interpolator instead of skfuzzy.compute
        """
        map_data = map_data.copy()
        drone_x, drone_y, _ = drone_position
        
        current_i = int((drone_x + map_center_offset)/self.distance_between_cells)
        current_j = int((drone_y + map_center_offset)/self.distance_between_cells)

        fitness_scores = []

        rows, cols = map_data.shape

        min_x_cell = max(0, current_i - self.NUMBER_OF_CELLS_X_Y//2)
        max_x_cell = min(rows, current_i + self.NUMBER_OF_CELLS_X_Y//2)
        min_y_cell = max(0, current_j - self.NUMBER_OF_CELLS_X_Y//2)
        max_y_cell = min(cols, current_j + self.NUMBER_OF_CELLS_X_Y//2)
       
        for i in range(min_x_cell, max_x_cell):
            for j in range(min_y_cell, max_y_cell):
                # Distance
                x_cell = self.distance_between_cells*i - map_center_offset
                y_cell = self.distance_between_cells*j - map_center_offset
                dist = np.sqrt((x_cell - drone_x) ** 2 + (y_cell - drone_y) ** 2)

                trajectory_cells = self.get_cells_visited_in_trajectory(
                    drone_altitude=drone_position[2],
                    initial_cell=(current_i, current_j),
                    final_cell=(i, j)
                )
                average_trajectory_cells = sum([map_data[cell[0], cell[1]] for cell in trajectory_cells])/len(trajectory_cells) if trajectory_cells else 0.0

                ##### Final fitness #####
                cell_fitness = average_trajectory_cells - dist/self.distance_norm
                fitness_scores.append((cell_fitness, (i, j)))

        return fitness_scores

    def both_cells_priority(self, map_data: np.array, first_drone_pos, second_drone_pos, map_center_offset) -> list:
        """
        Fully vectorized fuzzy inference for two drones.
        """
        # Get individual priorities (using the fast method above)
        list1 = self.cells_priority(map_data, first_drone_pos, map_center_offset)
        list2 = self.cells_priority(map_data, second_drone_pos, map_center_offset)

        if not list1 or not list2:
            return []

        # Convert to arrays for vectorization
        p1_vals = np.array([x[0] for x in list1])
        p1_coords = np.array([x[1] for x in list1]) # Shape (N, 2)

        p2_vals = np.array([x[0] for x in list2])
        p2_coords = np.array([x[1] for x in list2]) # Shape (M, 2)

        #Vectorized Sum of Priorities
        # Shape (N, 1) + (1, M) -> (N, M)
        sum_p_matrix = p1_vals[:, np.newaxis] + p2_vals[np.newaxis, :]

        #Vectorized Distance Calculation
        # Convert grid indices to physical coordinates
        phys_p1 = (p1_coords * self.distance_between_cells) - map_center_offset
        phys_p2 = (p2_coords * self.distance_between_cells) - map_center_offset

        # Broadcasting distance: (N, 1, 2) - (1, M, 2)
        diff = phys_p1[:, np.newaxis, :] - phys_p2[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2)) # Shape (N, M)

        # Normalize distance and combine with priorities
        dist_parameter = dist_matrix / self.distance_between_drone_norm

        total_fitness = sum_p_matrix + dist_parameter

        N = p1_coords.shape[0]
        M = p2_coords.shape[0]
        
        # Flatten the fitness matrix to a 1D array
        scores = total_fitness.ravel()
        
        # Repeat p1 coordinates M times for each element (e.g., A, A, A, B, B, B)
        p1_c = np.repeat(p1_coords, M, axis=0)
        
        # Tile p2 coordinates N times (e.g., X, Y, Z, X, Y, Z)
        p2_c = np.tile(p2_coords, (N, 1))
        
        # Combine them quickly without Python-level loops
        # Returns a list of tuples: (fitness_float, array([x1, y1]), array([x2, y2]))
        return list(zip(scores, p1_c, p2_c))
    
    def choose_one_cell(self, fitness_scores: list) -> Tuple[float, float]:
        if not fitness_scores:
            return None
        
        best_cell = max(fitness_scores, key=lambda x: x[0])
        # Return the coordinates
        return [best_cell[1], best_cell[0]]
    
    def choose_two_cells(self, fitness_scores: list) ->  List[Tuple[float, float]]:
        if not fitness_scores:
            return None
        
        fitness_scores = max(fitness_scores, key=lambda x: x[0])
        best_1 = (fitness_scores[1])
        best_2 = (fitness_scores[2])

        return [[best_1, best_2], fitness_scores[0]]

