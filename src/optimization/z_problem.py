"""Z-update NLP: formation consensus optimization.

This constructs and solves the coordination step of the ADMM algorithm.
Each vehicle updates its consensus variables z_i, z_ij to satisfy:
- Formation constraint (cross product = 0, parallel relative positions)
- Trajectory tracking (mean position = 0 in Frenet frame)
- Phi equality (all vehicles share the same rotation angle)
"""

import math
import time
import numpy as np
from casadi import MX, vertcat, dot, nlpsol, cos, sin

from ..bspline import BSpline, make_basis
from .nlp_builder import NLPBuilder


class ZProblem:
    """Builds and caches the CasADi NLP for the z-update."""

    def __init__(self, config, n_neighbours, neighbour_xf, own_x0, own_xf):
        """
        Args:
            config: Config object.
            n_neighbours: Number of neighbouring vehicles.
            neighbour_xf: List of xf (final positions) for each neighbour.
            own_x0: Own initial state [p0, q0, phi0, dp0, dq0, dphi0].
            own_xf: Own final state.
        """
        self.config = config
        self.solver = None
        self.arg_template = None
        self._build(config, n_neighbours, neighbour_xf, own_x0, own_xf)

    def _build(self, config, n_neighbours, neighbour_xf, own_x0, own_xf):
        b = NLPBuilder()
        basis = make_basis(degree=config.state_degree, knot_intervals=config.knot_intervals)
        nd = config.n_dimensions
        nd_old = config.n_dimensions_pos  # 2 (p, q only)
        slack = config.slack
        t_res = config.t_resolution_length

        # --- Parameters ---
        # y_i (own trajectory, from x-update)
        y = b.add_spline_param('y', basis, nd)

        # z_i (own consensus variable -- decision)
        z_i = b.add_spline_var('z_i', basis, nd,
                               lb=config.y_min, ub=config.y_max,
                               init=[[own_x0[i], own_xf[i]] for i in range(nd)])

        # lambda_i
        lambda_i = b.add_spline_param('lambda_i', basis, nd)

        # ADMM cost: y_i - z_i
        for i in range(nd):
            b.cost += dot(lambda_i[i].coeffs, y[i].coeffs - z_i[i].coeffs)
            b.cost += config.rho * dot(
                np.ones(y[i].coeffs.shape[0]),
                (y[i].coeffs - z_i[i].coeffs) ** 2
            )

        # Tracking: sum of positions should be at origin
        p_sum = z_i[0]
        q_sum = z_i[1]

        # --- For each neighbour ---
        for n in range(n_neighbours):
            y_j = b.add_spline_param('y_j', basis, nd)
            z_ij = b.add_spline_var('z_ij', basis, nd,
                                     lb=config.y_min, ub=config.y_max,
                                     init=[[neighbour_xf[n][i], neighbour_xf[n][i]]
                                           for i in range(nd)])
            lambda_ij = b.add_spline_param('lambda_ij', basis, nd)

            # ADMM cost: y_j - z_ij
            for i in range(nd):
                b.cost += dot(lambda_ij[i].coeffs, y_j[i].coeffs - z_ij[i].coeffs)
                b.cost += config.rho * dot(
                    np.ones(y_j[i].coeffs.shape[0]),
                    (y_j[i].coeffs - z_ij[i].coeffs) ** 2
                )

            # Formation constraint: cross product of relative positions
            # (z_i - z_ij) x R(phi) * x_ref = 0
            vec1 = z_i - z_ij  # current relative position
            vec2 = np.array(own_xf[:nd_old]) - np.array(neighbour_xf[n][:nd_old])  # reference

            def cross_product(s1, s2):
                return s1[0] * s2[1] - s1[1] * s2[0]

            def vector_rotation(spline_vec, alpha_spline, t):
                a1, a2 = spline_vec[0], spline_vec[1]
                return (a1 * cos(alpha_spline(t)) - a2 * sin(alpha_spline(t)),
                        a1 * sin(alpha_spline(t)) + a2 * cos(alpha_spline(t)))

            for t in np.linspace(0, 1, t_res):
                rotated = vector_rotation(vec2, z_i[2], t)
                cross_val = cross_product(vec1, rotated)(t)
                b.add_constraint_eval([cross_val], [-slack], [slack])

            # Phi equality: z_i[2] == z_ij[2]
            phi_diff = z_i[2] - z_ij[2]
            b.add_constraint_coeffs([phi_diff], [0], [0])

            # Accumulate position sums
            p_sum = p_sum + z_ij[0]
            q_sum = q_sum + z_ij[1]

        # Trajectory tracking: mean position = 0
        b.add_constraint_coeffs([p_sum, q_sum], [-slack, -slack], [slack, slack])

        # Build solver
        self.solver, self.arg_template = b.build_solver('z_solver', config.ipopt_options)

    def solve(self, x0_init, params):
        """Solve the z-update NLP.

        Args:
            x0_init: Initial guess (flat array of z_i, z_ij coefficients).
            params: Parameter values (y_i, lambda_i, y_j's, lambda_ij's).

        Returns:
            (solution_flat, stats, elapsed) tuple.
        """
        arg = dict(self.arg_template)
        arg['x0'] = x0_init
        arg['p'] = params

        start = time.time()
        sol = self.solver.call(arg)
        elapsed = time.time() - start

        stats = self.solver.stats()
        sol_flat = sol['x'].full().flatten()

        return sol_flat, stats, elapsed
