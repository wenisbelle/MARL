import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import glob # Used to easily find files matching a pattern

def parse_uav_log(log_file_path):
    """
    Parses a SINGLE UAV log file to extract data for each UAV.
    (This function is unchanged)
    """
    uav_data_lists = defaultdict(lambda: [[], [], [], []]) # Each UAV ID maps to 4 lists: uncertainty, unvisited, accomulated uncertainty, accomulated distance
    uncertainty_pattern = re.compile(r"At time: ([\d\.]+), node (\d+) map has total uncertainty of ([\d\.]+)")
    unvisited_pattern = re.compile(r"At time: ([\d\.]+), the node (\d+) has ([\d\.]+) unvisited cells")
    accomulated_uncertainty_pattern = re.compile(r"At time: ([\d\.]+), node (\d+) map has a accomulated uncertainty of ([\d\.]+)")
   
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match_uncertainty = uncertainty_pattern.match(line)
                match_unvisited = unvisited_pattern.match(line)
                match_accomulated = accomulated_uncertainty_pattern.match(line)

                if match_uncertainty:
                    time, uav_id, value = match_uncertainty.groups()
                    uav_data_lists[int(uav_id)][0].append([float(value), float(time)])
                elif match_unvisited:
                    time, uav_id, value = match_unvisited.groups()
                    uav_data_lists[int(uav_id)][1].append([float(value), float(time)])
                elif match_accomulated:
                    time, uav_id, value = match_accomulated.groups()
                    uav_data_lists[int(uav_id)][2].append([float(value), float(time)])

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return None

    processed_uav_data = {}
    for uav_id, data_lists in sorted(uav_data_lists.items()):
        uncertainty_arr = np.array(data_lists[0]).T if data_lists[0] else np.empty((2, 0))
        unvisited_arr = np.array(data_lists[1]).T if data_lists[1] else np.empty((2, 0))
        accomulated_uncertainty_arr = np.array(data_lists[2]).T if data_lists[2] else np.empty((2, 0))
        processed_uav_data[uav_id] = [uncertainty_arr, unvisited_arr, accomulated_uncertainty_arr]

    return processed_uav_data

def plot_uncertainty_comparison_stacked_triple(
    labeled_set_1, 
    labeled_set_2, 
    labeled_set_3, 
    title_1="Dataset 1", 
    title_2="Dataset 2",
    title_3="Dataset 3"
):
    """
    Plots the 'Total Map Uncertainty' for three different labeled_datasets,
    stacked vertically.

    Args:
        labeled_set_1 (dict): The first dictionary of labeled datasets.
        labeled_set_2 (dict): The second dictionary of labeled datasets.
        labeled_set_3 (dict): The third dictionary of labeled datasets.
        title_1 (str): A title for the top chart (representing set 1).
        title_2 (str): A title for the middle chart (representing set 2).
        title_3 (str): A title for the bottom chart (representing set 3).
    """
    
    # --- 1. Setup the Figure ---
    # 3 rows, 1 column. Share the x-axis.
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    fig.suptitle('Comparison of Total Map Uncertainty', fontsize=18)
    
    # --- 2. Define the specific metric we care about ---
    metric_info = {'title': 'Total Map Uncertainty', 'ylabel': 'Uncertainty Value'}
    metric_idx = 0 # This is the index for 'Uncertainty'
    dataset_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Bundle datasets and titles for easy looping
    datasets_to_plot = [
        (labeled_set_1, title_1),
        (labeled_set_2, title_2),
        (labeled_set_3, title_3)
    ]

    # --- 3. Loop through each dataset and its corresponding axis ---
    for ax, (labeled_dataset, plot_title) in zip(axes, datasets_to_plot):
        if not labeled_dataset:
            print(f"No data provided for {plot_title}")
            ax.set_title(f"{metric_info['title']} ({plot_title}) - NO DATA", fontsize=14)
            continue

        for color_idx, (label, all_uav_data) in enumerate(labeled_dataset.items()):
            color = dataset_colors[color_idx % len(dataset_colors)]
            
            all_times, all_uav_metric_data = [], []
            if not all_uav_data:
                continue

            for uav_id in all_uav_data:
                # Get the metric data for this UAV (metric_idx = 0)
                uav_data = all_uav_data[uav_id][metric_idx] 
                if uav_data.size > 0:
                    all_times.extend(uav_data[1])
                    all_uav_metric_data.append(uav_data)
            
            if not all_times:
                continue # Skip if this dataset has no data for this metric
            
            # Handle cases where min==max
            min_time = min(all_times)
            max_time = max(all_times)
            if min_time == max_time:
                common_times = np.array([min_time])
            else:
                 common_times = np.linspace(min_time, max_time, num=200)

            interpolated_values = [np.interp(common_times, d[1], d[0]) for d in all_uav_metric_data]

            if interpolated_values:
                mean_values = np.mean(interpolated_values, axis=0)
                std_dev = np.std(interpolated_values, axis=0)

                # Plot on the current axis
                ax.plot(common_times, mean_values, color=color, label=label, linewidth=2)
                ax.fill_between(common_times, mean_values - std_dev, mean_values + std_dev,
                                color=color, alpha=0.15)
    
        # Finalize the current subplot
        ax.set_title(f"{metric_info['title']} ({plot_title})", fontsize=14)
        ax.set_ylabel(metric_info['ylabel'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
    
    # --- 4. Finalize the entire figure ---
    axes[-1].set_xlabel('Time (s)') # Set x-label only on the bottom plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("stacked_uncertainty_comparison_triple.png") # Give it a new name
    plt.show()

def parse_and_average_logs_from_folder(folder_path):
    """
    Parses all .log files in a given folder, averages the results across
    all simulations FOR EACH UAV individually, and returns a single processed 
    data dictionary.
    """
    log_files = glob.glob(os.path.join(folder_path, 'simulation*.log'))
    if not log_files:
        print(f"No simulation log files found in '{folder_path}'")
        return None

    print(f"Found {len(log_files)} log files to process.")

    # 1. Collect data from all simulation runs
    all_simulations_data = []
    for log_file in log_files:
        single_sim_data = parse_uav_log(log_file)
        if single_sim_data:
            all_simulations_data.append(single_sim_data)

    if not all_simulations_data:
        print("No data could be parsed from any log file.")
        return None

    # 2. Average the data across all collected simulations PER UAV
    averaged_data = defaultdict(list)
    
    # Find all unique UAV IDs across all simulations (e.g., Drones 0, 1, 2, 3, 4)
    all_uav_ids = set()
    for sim_data in all_simulations_data:
        all_uav_ids.update(sim_data.keys())

    for uav_id in sorted(list(all_uav_ids)):
        for metric_idx in range(3): # 0: uncertainty, 1: unvisited, 2: accomulated uncertainty
            
            # Gather data for this specific UAV and metric from every simulation
            all_runs_for_metric = []
            all_times_for_metric = []
            for sim_data in all_simulations_data:
                if uav_id in sim_data and sim_data[uav_id][metric_idx].size > 0:
                    metric_data = sim_data[uav_id][metric_idx]
                    
                    # Sort timestamps chronologically ---
                    # metric_data[0] is values, metric_data[1] is times
                    sorted_indices = np.argsort(metric_data[1])
                    sorted_times = metric_data[1][sorted_indices]
                    sorted_values = metric_data[0][sorted_indices]
                    
                    # Store the cleaned, sorted run
                    all_runs_for_metric.append((sorted_values, sorted_times))
                    all_times_for_metric.extend(sorted_times)

            if not all_runs_for_metric:
                averaged_data[uav_id].append(np.empty((2, 0)))
                continue

            # Create a common time axis for this specific UAV across all runs
            common_times = np.linspace(min(all_times_for_metric), max(all_times_for_metric), num=200)
            
            interpolated_runs = []
            for run_vals, run_times in all_runs_for_metric:
                
                # If calculating 'Accumulated Uncertainty' (idx 2), force it to start at 0.0
                # before the drone's first logged timestamp. 
                if metric_idx == 2:
                    interp_vals = np.interp(common_times, run_times, run_vals, left=0.0)
                else:
                    # For total uncertainty or unvisited cells, default padding is safer
                    interp_vals = np.interp(common_times, run_times, run_vals)
                
                interpolated_runs.append(interp_vals)

            # Calculate the mean across the simulations for THIS specific UAV
            mean_values = np.mean(interpolated_runs, axis=0)
            
            # Store in the final format: [[values...], [times...]]
            final_metric_array = np.array([mean_values, common_times])
            averaged_data[uav_id].append(final_metric_array)

    return dict(averaged_data)

def plot_swarm_comparison(labeled_datasets):
    """
    Generates a comparison plot for multiple swarm data sets.

    Args:
        labeled_datasets (dict): A dictionary where keys are string labels 
                                 (e.g., "Decentralized") and values are the 
                                 'all_uav_data' dictionaries to be plotted.
    """
    if not labeled_datasets:
        print("No datasets were provided for plotting.")
        return

    # --- 1. Setup the Figure ---
    fig, axes = plt.subplots(3 ,1, figsize=(14, 18), sharex=True)
    fig.suptitle('Comparison of Swarm Performance Metrics', fontsize=18)

    # --- 2. Define Metrics and Colors for different datasets ---
    metrics = [
        {'title': 'Total Map Uncertainty', 'ylabel': 'Uncertainty Value'},
        {'title': 'Number of Unvisited Cells', 'ylabel': 'Cell Count'},
        {'title': 'Accomulated Uncertainty', 'ylabel': 'Accomulated Uncertainty Value'},
    ]
    # Define a list of colors to cycle through for each dataset
    dataset_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # --- 3. Loop through each METRIC (Uncertainty, Unvisited, etc.) ---
    for metric_idx, metric_info in enumerate(metrics):
        ax = axes[metric_idx]

        # --- 4. Loop through each DATASET provided in the dictionary ---
        for color_idx, (label, all_uav_data) in enumerate(labeled_datasets.items()):
            color = dataset_colors[color_idx % len(dataset_colors)]
            
            # --- This is the same logic as before, but now applied to each dataset ---
            all_times, all_uav_metric_data = [], []
            if not all_uav_data:
                continue

            for uav_id in all_uav_data:
                uav_data = all_uav_data[uav_id][metric_idx]
                if uav_data.size > 0:
                    all_times.extend(uav_data[1])
                    all_uav_metric_data.append(uav_data)
            
            if not all_times:
                continue # Skip if this dataset has no data for this metric

            common_times = np.linspace(min(all_times), max(all_times), num=200)
            interpolated_values = [np.interp(common_times, d[1], d[0]) for d in all_uav_metric_data]

            if interpolated_values:
                mean_values = np.mean(interpolated_values, axis=0)
                std_dev = np.std(interpolated_values, axis=0)

                # --- 5. Plot the data for the current dataset with its label and color ---
                ax.plot(common_times, mean_values, color=color, label=label, linewidth=2)
                ax.fill_between(common_times, mean_values - std_dev, mean_values + std_dev,
                                color=color, alpha=0.15)
        
        # --- 6. Finalize each subplot ---
        ax.set_title(metric_info['title'], fontsize=14)
        ax.set_ylabel(metric_info['ylabel'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    # --- 7. Finalize the entire figure ---
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    first_label = list(labeled_datasets.keys())[0] if labeled_datasets else "plot"
    plt.savefig(f"swarm_comparison_performance_{first_label}.png")
    plt.show()

def plot_swarm_uncertainty(labeled_datasets):
    """
    Generates a comparison plot for ONLY the 'Total Map Uncertainty' metric
    across multiple swarm data sets.

    Args:
        labeled_datasets (dict): A dictionary where keys are string labels 
                                 (e.g., "Decentralized") and values are the 
                                 'all_uav_data' dictionaries to be plotted.
    """
    if not labeled_datasets:
        print("No datasets were provided for plotting.")
        return

    # --- 1. Define the specific metric index we want ---
    metric_idx = 0 # 0 for 'Total Map Uncertainty'

    # --- 2. Setup the Figure ---
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Define a list of colors to cycle through for each dataset
    dataset_colors = ["#2d86c5", '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # --- 3. Loop through each DATASET provided in the dictionary ---
    for color_idx, (label, all_uav_data) in enumerate(labeled_datasets.items()):
        color = dataset_colors[color_idx % len(dataset_colors)]
        
        all_times, all_uav_metric_data = [], []
        if not all_uav_data:
            continue

        for uav_id in all_uav_data:
            # Check if UAV data exists and has the metric
            if uav_id not in all_uav_data or not all_uav_data[uav_id] or len(all_uav_data[uav_id]) <= metric_idx:
                continue
                
            uav_data = all_uav_data[uav_id][metric_idx] # This is a numpy array
            
            # --- THIS IS THE FIX ---
            # Check if the numpy array returned by the parser has any data
            if uav_data.size > 0:
                all_times.extend(uav_data[1])
                all_uav_metric_data.append(uav_data)
            # --- END OF FIX ---

        if not all_times:
            print(f"Warning: Dataset '{label}' had no valid data for metric index {metric_idx}.")
            continue # Skip if this dataset has no data for this metric
        
        # Handle cases where min==max
        min_time = min(all_times)
        max_time = max(all_times)
        if min_time == max_time:
            common_times = np.array([min_time])
        else:
            common_times = np.linspace(min_time, max_time, num=200)
        
        # Filter out any empty or invalid data before interpolation
        valid_uav_metric_data = [d for d in all_uav_metric_data if d.size > 0]
        if not valid_uav_metric_data:
            continue
            
        interpolated_values = [np.interp(common_times, d[1], d[0]) for d in valid_uav_metric_data]

        if interpolated_values:
            mean_values = np.mean(interpolated_values, axis=0)
            std_dev = np.std(interpolated_values, axis=0)

            # --- 4. Plot the data for the current dataset ---
            ax.plot(common_times, mean_values, color=color, label=label, linewidth=2)
            ax.fill_between(common_times, mean_values - std_dev, mean_values + std_dev,
                            color=color, alpha=0.15)
    
    # --- 5. Finalize the subplot ---
    ax.set_ylabel('Uncertainty Value')
    ax.set_xlabel('Time (s)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Only add legend if there are labels to show
    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    # --- 6. Finalize the entire figure ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Give each plot a unique name based on the first label
    # to avoid overwriting the same file.
    first_label = list(labeled_datasets.keys())[0] if labeled_datasets else "plot"
    plt.savefig(f"swarm_uncertainty_comparison_{first_label}.png")
    plt.show()


MARL = parse_and_average_logs_from_folder("/MARL/src/test/MADQN/logs/MARL")

Greed = parse_and_average_logs_from_folder("/MARL/src/test/MADQN/logs/greed")

#eGreed = parse_and_average_logs_from_folder("/FuzzyGA/src/definitive_system/coordination_only/3_drones/e_greed/logs")

#labeled_datasets1 = {
#    "First and Second Best Individual": Fuzzy_first_second_best_individual,
#    "Third Best Individual": Fuzzy_third_best_individual,
#    "Fuzzy modified": Fuzzy_modified_individual
#}

labeled_datasets2 = {
    "MARL": MARL,
    "Greed": Greed
}

plot_swarm_uncertainty(labeled_datasets2)
plot_swarm_comparison(labeled_datasets2)

