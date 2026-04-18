"""X-update NLP: trajectory optimization for a single vehicle.

This constructs and solves the prediction step of the ADMM algorithm.
Each vehicle optimizes its own trajectory subject to:
- Initial position/velocity constraints
- Waypoint constraints (from warm-starter)
- Obstacle avoidance (separating hyperplane)
- ADMM consensus cost (y_i - z_i, y_i - z_ji)
- Acceleration minimization cost
"""

import math
import time
import numpy as np
from casadi import MX, vertcat, dot, nlpsol, cos, sin

from ..bspline import BSpline, make_basis, definite_integral
from .nlp_builder import NLPBuilder


class XProblem:
    """Builds and caches the CasADi NLP for the x-update."""

    def __init__(self, config, n_obstacles, n_neighbours, obstacle_radii):
        """
        Args:
            config: Config object.
            n_obstacles: Number of obstacles.
            n_neighbours: Number of neighbouring vehicles.
            obstacle_radii: List of max_dist_from_center for each obstacle.
        """
        self.config = config
        self.solver = None
        self.arg_template = None
        self._build(config, n_obstacles, n_neighbours, obstacle_radii)

    def _build(self, config, n_obstacles, n_neighbours, obstacle_radii):
        """Construct the NLP symbolically (done once at setup)."""
        b = NLPBuilder()
        basis = make_basis(degree=config.state_degree, knot_intervals=config.knot_intervals)
        obs_basis = make_basis(degree=config.obstacle_degree, knot_intervals=config.obstacle_knot_intervals)

        # --- Parameters ---
        x0 = b.add_scalar_param('x0', config.state_len)
        xf = b.add_scalar_param('xf', config.state_len)

        # --- Decision variables: position splines (p, q, phi) ---
        y = b.add_spline_var('y', basis, config.n_dimensions,
                             lb=config.y_min, ub=config.y_max,
                             init=[[0, 0]] * config.n_dimensions)
        p, q, phi = y[0], y[1], y[2]
        y_dot = [s.derivative() for s in y]
        y_ddot = [s.derivative() for s in y_dot]

        # --- Initial constraints ---
        b.add_constraint_initial_param(y, x0[:config.n_dimensions], x0[:config.n_dimensions])
        b.add_constraint_initial_param(y_dot, x0[config.n_dimensions:], x0[config.n_dimensions:])

        # --- Waypoint constraints (MPC_param mode) ---
        n_wp = config.n_waypoints
        for i in range(n_wp):
            x_wp = b.add_scalar_param('x_intermediate', config.n_dimensions)
            t_wp = b.add_scalar_param('t_intermediate', 1)

            # Position must be near waypoint
            b.add_constraint_eval(
                [(p(t_wp) - x_wp[0]) ** 2, (q(t_wp) - x_wp[1]) ** 2],
                [0, 0],
                [config.radius ** 2, config.radius ** 2]
            )
            # Rotation must be near waypoint
            b.add_constraint_eval(
                [(phi(t_wp) - x_wp[2]) ** 2],
                [0],
                [5 / 360 * math.pi * 2]
            )

        # Keep lateral speed zero at the horizon end.
        q_dot = y_dot[1]
        b.add_constraint_final([q_dot], [0], [0])

        # --- Obstacle avoidance ---
        for i in range(n_obstacles):
            # Corner splines as parameters
            corners = []
            for j in range(4):
                corner = b.add_spline_param('obst', obs_basis, 2)
                corners.append(corner)

            # Center spline as parameter
            b.add_spline_param('obst_center', obs_basis, 2)

            b.add_hyperplane_avoidance(
                [p, q], corners,
                radius=0.0,
                center_circle=None,
                config=config
            )

        # --- Acceleration cost ---
        acc_cost = sum(dd ** 2 for dd in y_ddot)
        b.cost += config.rho_input * definite_integral(acc_cost, 0, 1)

        # --- ADMM consensus cost: (y_i - z_i) ---
        z_i = b.add_spline_param('z_i', basis, config.n_dimensions)
        lambda_i = b.add_spline_param('lambda_i', basis, config.n_dimensions)

        for i in range(len(y)):
            b.cost += dot(lambda_i[i].coeffs, y[i].coeffs - z_i[i].coeffs)
            b.cost += config.rho * dot(
                np.ones(y[i].coeffs.shape[0]),
                (y[i].coeffs - z_i[i].coeffs) ** 2
            )

        # --- ADMM consensus cost: (y_i - z_ji) for each neighbour ---
        for n in range(n_neighbours):
            z_ji = b.add_spline_param('z_ji', basis, config.n_dimensions)
            lambda_ji = b.add_spline_param('lambda_ji', basis, config.n_dimensions)

            for i in range(len(y)):
                b.cost += dot(lambda_ji[i].coeffs, y[i].coeffs - z_ji[i].coeffs)
                b.cost += config.rho * dot(
                    np.ones(y[i].coeffs.shape[0]),
                    (y[i].coeffs - z_ji[i].coeffs) ** 2
                )

        # --- Build solver ---
        self.solver, self.arg_template = b.build_solver('x_solver', config.ipopt_options)
        self._n_y = len(basis) * config.n_dimensions
        self._n_a = len(basis) * 2 * n_obstacles
        self._n_b = len(basis) * n_obstacles
        self._n_dtau = len(basis) * n_obstacles

    def solve(self, x0_init, params):
        """Solve the x-update NLP.

        Args:
            x0_init: Initial guess for decision variables (flat array).
            params: Parameter values (flat array).

        Returns:
            (solution_flat, stats) tuple.
        """
        arg = dict(self.arg_template)
        arg['x0'] = x0_init
        arg['p'] = params

        start = time.time()
        sol = self.solver.call(arg)
        elapsed = time.time() - start

        stats = self.solver.stats()
        sol_flat = sol['x'].full().flatten()

        # Retry on failure
        if stats['return_status'] != 'Solve_Succeeded':
            arg['x0'] = sol_flat
            sol = self.solver.call(arg)
            stats = self.solver.stats()
            sol_flat = sol['x'].full().flatten()

        return sol_flat, stats, elapsed
