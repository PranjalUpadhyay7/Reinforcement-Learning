import time
import tracemalloc
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Adjust these imports based on your actual project structure
from src.config import Q1Config
from src.environment import DifferentialDriveRobot
from src.utils import Logger, Visualizer
from src.solvers import Valueiterations, Policyiterations, MonteCarlo 
# (Assuming you saved the solver classes in a file named solvers.py)

def run_and_profile(solver_class, env, logger, algo_name):
    """
    Initializes a solver, runs it, and profiles its time and memory footprint.
    """
    print(f"\n{'='*40}")
    print(f"Starting {algo_name}...")
    print(f"{'='*40}")
    
    # Initialize solver
    solver = solver_class(environment=env, logger=logger)
    
    # Start tracking memory and time
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Run the RL algorithm
    solver.solve()
    
    # Stop tracking
    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate stats
    execution_time = end_time - start_time
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    print(f"[{algo_name} Finished]")
    print(f"Total Iterations : {solver.iterations}")
    print(f"Execution Time   : {execution_time:.4f} seconds")
    print(f"Peak Memory Usage: {peak_mem_mb:.4f} MB")
    
    return execution_time, peak_mem_mb

def main():
    # 1. Initialize Configuration and Environment
    base_config = Q1Config()
    env = DifferentialDriveRobot(base_config)
    
    # 2. Initialize the Logger (this will clear old logs)
    logger = Logger(log_dir="output/logs")
    
    # 3. Run and Profile Solvers
    # (Uncomment SARSA/Q-Learning when you write them!)
    solvers_to_run = [
        (Valueiterations, "ValueIteration"),
        (Policyiterations, "PolicyIteration"),
        (MonteCarlo, "MonteCarlo")
    ]
    
    stats = {}
    for solver_class, name in solvers_to_run:
        # Reset the environment between runs if necessary, 
        # though solvers usually handle their own V/Q initializations.
        time_taken, mem_used = run_and_profile(solver_class, env, logger, name)
        stats[name] = {"time": time_taken, "memory": mem_used}
        
    # 4. Print Summary Comparison Table
    print("\n\n" + "="*50)
    print("PERFORMANCE SUMMARY".center(50))
    print("="*50)
    print(f"{'Algorithm':<20} | {'Time (s)':<12} | {'Peak Mem (MB)':<12}")
    print("-" * 50)
    for name, data in stats.items():
        print(f"{name:<20} | {data['time']:<12.4f} | {data['memory']:<12.4f}")
    print("="*50 + "\n")

    # 5. Visualize the Results
    print("Generating visualizations...")
    visualizer = Visualizer(log_dir="output/logs", plot_dir="output/plots") # <--- Update this line
    
    # Plot Convergence (Delta drops) for Planning Algos
    visualizer.plot_convergence(algos=["ValueIteration", "PolicyIteration"])
    
    # Plot Learning Curve (Episode Returns) for Learning Algos
    visualizer.plot_learning_curve(algos=["MonteCarlo"], window=10)
    
    # Extract grid shape from your config for the heatmaps.
    # NOTE: Change 'grid_width'/'grid_height' to match whatever variables 
    # you actually used in Q1Config to define the environment's dimensions.
    try:
        grid_shape = (base_config.grid_size[0], base_config.grid_size[1])
        my_state_mapper = lambda state: (state[0], state[1])
    
        def my_action_mapper(state, action):
            if action != "Forward":
                return (0, 0)

            theta = state[2]
            # Mapping: 0=Up, 1=Right, 2=Down, 3=Left (Adjust according to your env rules)
            if theta == 0: return (-1, 0)
            if theta == 1: return (0, 1)
            if theta == 2: return (1, 0)
            if theta == 3: return (0, -1)
            return (0, 0)

        visualizer.plot_grid_visuals(
            algo_name="ValueIteration", 
            grid_shape=grid_shape,
            state_mapper=my_state_mapper,
            action_mapper=my_action_mapper
        )
    except AttributeError:
        print("Could not find grid dimensions in base_config. Skipping heatmaps.")
        print("Update the 'grid_shape' variable in main.py to plot the grid visuals.")

if __name__ == "__main__":
    main()