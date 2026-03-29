"""Centralized configuration for the formation control system."""

from dataclasses import dataclass, field
import math


@dataclass
class Config:
    """All hyperparameters in one place."""

    # Dimensions
    n_dimensions: int = 3          # p, q, phi
    n_dimensions_pos: int = 2      # p, q only
    state_degree: int = 3
    knot_intervals: int = 5

    # Time
    t_step: float = 0.01
    t_window_size: float = 0.12

    # ADMM
    rho: float = 50.0
    rho_input: float = 200.0
    rho_final_value: float = 5000.0
    n_admm_iterations: int = 1

    # Vehicle
    radius: float = 0.08
    slack: float = 1e-5
    epsilon: float = 0.05
    safety_weight: float = 100.0

    # Solver
    ipopt_max_iter: int = 10000
    ipopt_print_level: int = 0
    ipopt_max_cpu_time: float = 100.0

    # Formation warm-start
    n_waypoints: int = 5
    back_scaling_factor: float = 0.3
    back_rotation_factor: float = 0.4

    # Obstacle spline
    obstacle_degree: int = 3
    obstacle_knot_intervals: int = 5
    obstacle_fit_knot_intervals: int = 20

    # Formation
    n_vehicles: int = 4

    # Bounds on state
    y_min: list = field(default_factory=lambda: [-2, -2, -math.pi * 2])
    y_max: list = field(default_factory=lambda: [2, 2, math.pi * 2])

    # Resolution
    t_resolution_length: int = 6  # knot_intervals + 1

    @property
    def ipopt_options(self):
        return {
            'print_time': False,
            'ipopt': {
                'print_level': self.ipopt_print_level,
                'max_iter': self.ipopt_max_iter,
                'max_cpu_time': self.ipopt_max_cpu_time
            }
        }

    @property
    def state_len(self):
        """Total state length: [pos, vel] = 2 * n_dimensions."""
        return self.n_dimensions * 2
