import os
import json

class Logger:
    def __init__(self, log_dir="output/logs"):
        """
        Initializes the logger, creating the directory if it doesn't exist.
        It also clears out any old log files from previous runs to ensure clean data.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Define the algorithms we expect to log
        algos = ["ValueIteration", "PolicyIteration", "MonteCarlo", "SARSA", "TDLearning", "QLearning"]
        
        # Clear old log files
        for algo in algos:
            filepath = os.path.join(self.log_dir, f"{algo}_log.jsonl")
            # Opening in 'w' mode truncates the file to 0 bytes if it exists, 
            # or creates an empty file if it doesn't.
            open(filepath, 'w').close()

    def _stringify_keys(self, data):
        """
        Recursively converts dictionary keys to strings.
        JSON requires string keys, but GridWorld states are often tuples like (0, 1).
        """
        if isinstance(data, dict):
            return {str(k): self._stringify_keys(v) for k, v in data.items()}
        return data

    def record(self, algo_name, step, **kwargs):
        """
        Appends a single step/iteration of data to the algorithm's log file.
        
        Args:
            algo_name (str): The name of the algorithm (e.g., "SARSA").
            step (int): The current iteration or episode number.
            **kwargs: Arbitrary metrics to log (e.g., V, Q, policy, episode_return, delta).
        """
        log_entry = {"step": step}
        
        # Process all passed keyword arguments
        for key, value in kwargs.items():
            log_entry[key] = self._stringify_keys(value)

        filepath = os.path.join(self.log_dir, f"{algo_name}_log.jsonl")
        
        # Append the new entry as a single JSON string followed by a newline
        with open(filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

import os
import json
import ast
import numpy as np

# Set the backend to 'Agg' BEFORE importing pyplot to prevent Tkinter errors on headless/Linux setups
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, log_dir="output/logs", plot_dir="output/plots", config=None):
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.config = config # Store config for fallback variables if needed
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _destringify_keys(self, data):
        """Safely converts JSON string keys back to original Python types."""
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                try:
                    parsed_key = ast.literal_eval(k)
                except (ValueError, SyntaxError):
                    parsed_key = k
                new_dict[parsed_key] = self._destringify_keys(v)
            return new_dict
        return data

    def _load_data(self, algo_name):
        """Reads the .jsonl file for a given algorithm."""
        filepath = os.path.join(self.log_dir, f"{algo_name}_log.jsonl")
        data = []
        if not os.path.exists(filepath):
            print(f"Warning: No log file found for {algo_name} at {filepath}")
            return data
            
        with open(filepath, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                data.append(self._destringify_keys(entry))
        return data

    def plot_convergence(self, algos=["ValueIteration", "PolicyIteration"]):
        """Plots the Delta drop for planning algorithms."""
        plt.figure(figsize=(10, 5))
        
        for algo in algos:
            data = self._load_data(algo)
            if not data: continue
            
            steps = [entry.get("step") for entry in data if "delta" in entry]
            deltas = [entry.get("delta") for entry in data if "delta" in entry]
            
            if deltas:
                plt.plot(steps, deltas, label=algo, marker='o', markersize=4)

        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Max Delta ($\Delta$)')
        plt.title('Algorithm Convergence (Log Scale)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        save_path = os.path.join(self.plot_dir, "convergence_plot.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved convergence plot to {save_path}")

    def plot_learning_curve(self, algos=["MonteCarlo"], window=10):
        """Plots the moving average of episode returns for learning algorithms."""
        plt.figure(figsize=(10, 5))
        
        for algo in algos:
            data = self._load_data(algo)
            if not data: continue
            
            returns = [entry.get("episode_return") for entry in data if "episode_return" in entry]
            
            if len(returns) > 0:
                if len(returns) >= window:
                    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
                    plt.plot(range(window, len(returns) + 1), smoothed, label=f"{algo} (MA-{window})")
                else:
                    plt.plot(returns, label=algo, alpha=0.5)

        plt.xlabel('Episodes')
        plt.ylabel('Episode Return')
        plt.title('Agent Learning Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = os.path.join(self.plot_dir, "learning_curve.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved learning curve to {save_path}")

    def plot_grid_visuals(self, algo_name, grid_shape, state_mapper=None, action_mapper=None):
        """
        A completely generic grid plotter. 
        
        Args:
            grid_shape (tuple): (rows, cols)
            state_mapper (callable): A function that takes a raw `state` and returns `(row, col)`.
            action_mapper (callable): A function that takes `(state, action)` and returns `(dr, dc)` for the arrow.
        """
        data = self._load_data(algo_name)
        if not data: return
        
        final_entry = data[-1]
        V = final_entry.get("V", {})
        policy = final_entry.get("policy", {})

        if not V:
            print(f"No Value function found for {algo_name}.")
            return

        rows, cols = grid_shape
        V_matrix = np.full(grid_shape, -np.inf)
        U, V_vec = np.zeros(grid_shape), np.zeros(grid_shape)
        
        # Track the "best" original state for each (r, c) cell so we can query the policy correctly
        best_state_for_cell = {}

        # --- Default Mappers (if none are provided) ---
        if state_mapper is None:
            # Assumes state is either (r, c, ...), (r, c), or a flat int.
            def state_mapper(s):
                if isinstance(s, tuple) and len(s) >= 2: return s[0], s[1]
                if isinstance(s, int): return divmod(s, cols)
                return None

        if action_mapper is None:
            # Assumes integer actions 0:Up, 1:Right, 2:Down, 3:Left
            default_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            def action_mapper(s, a):
                return default_map.get(a, (0, 0))

        # 1. Project Values onto the 2D Grid
        for state, val in V.items():
            rc = state_mapper(state)
            if rc is None or not (0 <= rc[0] < rows and 0 <= rc[1] < cols):
                continue
                
            r, c = rc
            # If multiple states map to the same cell (e.g. diff orientations), take the MAX value
            if val > V_matrix[r, c]:
                V_matrix[r, c] = val
                best_state_for_cell[(r, c)] = state

        # Clean up unseen states
        V_matrix[V_matrix == -np.inf] = np.nanmin(V_matrix) if not np.all(V_matrix == -np.inf) else 0

        # 2. Build Policy Arrows based on the best state in each cell
        for r in range(rows):
            for c in range(cols):
                best_state = best_state_for_cell.get((r, c))
                if best_state is not None and best_state in policy:
                    action = policy[best_state]
                    dr, dc = action_mapper(best_state, action)
                    U[r, c] = dc      # X vector
                    V_vec[r, c] = -dr # Y vector (inverted for matplotlib grid)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(V_matrix, cmap='viridis')
        fig.colorbar(cax, label='State Value $V(s)$', fraction=0.046, pad=0.04)

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        ax.quiver(x, y, U, V_vec, color='white', scale=15, pivot='mid')
        
        ax.set_title(f'{algo_name}: Value Heatmap & Optimal Policy')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        
        save_path = os.path.join(self.plot_dir, f"{algo_name}_grid_visual.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved generic grid visual to {save_path}")