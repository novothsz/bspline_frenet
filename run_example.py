"""Main example: formation control with obstacle avoidance.

Replaces ACC_example_2.py with the clean rewritten codebase.
"""

import time
import math
import os

from src.config import Config
from src.formation import Group


def run(config=None):
    if config is None:
        config = Config()

    # Create group
    start_position = [0.0, 0.0, 0.0]
    goal_position = [0.0, 0.0, 0.0]

    group = Group(
        n_vehicles=config.n_vehicles,
        config=config,
        start_position=start_position,
        goal_position=goal_position,
    )

    # Generate and add obstacles
    obstacles = group.generate_obstacles()
    group.add_obstacles(obstacles)
    group.setup_neighbours()

    # Prepare solvers
    print("Building NLP solvers...")
    group.prepare()
    print("Solvers built.")

    # Run simulation
    n_steps = math.floor(1 / config.t_step)
    iteration_times = []

    for step in range(n_steps):
        t_iter = time.time()

        # Generate waypoints (warm-start)
        group.generate_waypoints()

        # Run ADMM iterations
        for _ in range(config.n_admm_iterations):
            group.solve_step()
            # Disable shifting for subsequent ADMM iterations in same step
            for v in group.vehicles:
                v.shift_enabled = False

        elapsed = time.time() - t_iter
        iteration_times.append(elapsed)
        print(f"Step {step}/{n_steps}: {elapsed:.3f}s")

        # Advance simulation
        group.simulation_step()

    # Print summary
    print(f"\nCompleted {n_steps} steps")
    print(f"Average iteration time: {sum(iteration_times) / len(iteration_times):.3f}s")
    print(f"Total time: {sum(iteration_times):.1f}s")

    # Print solver stats
    for v in group.vehicles:
        n_success = sum(1 for s in v.history['feasibility'] if s == 'Solve_Succeeded')
        print(f"Vehicle {v.id}: {n_success}/{len(v.history['feasibility'])} succeeded")

    return group, iteration_times


if __name__ == "__main__":
    group, times = run()

    # Plot results
    try:
        from src.visualization import plot_frenet_view
        plot_frenet_view(group, "./frenet_view.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
