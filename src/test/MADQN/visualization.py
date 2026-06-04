import matplotlib.pyplot as plt
import numpy as np
import logging

class MapVisualizer:
    """
    Handles the real-time visualization of drone maps using Matplotlib.
    """
    def __init__(self, num_drones: int, map_width: int = 10, map_height: int = 10, distance_between_cells: int = 10):
        try:
            self.distance_between_cells = distance_between_cells
            plt.ion()
            
            self.fig, self.axes = plt.subplots(1, num_drones, figsize=(5 * num_drones, 10), squeeze=False)

            self.map_shape = (map_height*distance_between_cells, map_width*distance_between_cells)
            self.images = [] 
            self.drone_markers = []
            
            for i in range(num_drones):
                initial_map_data = np.ones(self.map_shape)

                # This is now perfectly safe because of squeeze=False
                ax_top = self.axes[0, i]
                im_top = ax_top.imshow(initial_map_data, cmap='gray_r', vmin=0, vmax=1, origin='lower')
                ax_top.set_title(f"Drone {i} Map")
                ax_top.set_xticks([])
                ax_top.set_yticks([])
                self.images.append(im_top)

                marker, = ax_top.plot([], [], 'ro', markersize=8, label="Drone Position") 
                self.drone_markers.append(marker)
                
            self.fig.tight_layout(pad=2.0)
            plt.show()
            
        except Exception as e:
            pass
        #    # If this triggers, your drone maps will not work!
        #    #logging.error(f"Error initializing visualizer: {e}")
        #    raise # It's usually better to raise the error so you know it failed

    def update_map(self, drone_id: int, map_data: np.ndarray, drone_position: tuple = None):
        """
        Updates the map visualization for a specific drone.
        """
        try:
            plot_index = drone_id
            
            map_view = map_data.copy().T
            
            map_view = np.clip(map_view, 0, 1)
            
            self.images[plot_index].set_data(map_view)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            x, y = drone_position
            scaled_x = (x * self.distance_between_cells) + (self.distance_between_cells / 2) - 0.5
            scaled_y = (y * self.distance_between_cells) + (self.distance_between_cells / 2) - 0.5
            self.drone_markers[plot_index].set_data([scaled_x], [scaled_y])
            
        except Exception as e:
            pass
        #    logging.warning(f"Could not update map for drone {drone_id}: {e}")
        
    def close(self):
        """Closes the Matplotlib window."""
        plt.ioff()
        plt.close(self.fig)