"""Vehicle: holds state and manages the per-vehicle optimization."""

import numpy as np
import copy
from ..config import Config
from ..bspline import BSpline, make_basis
from ..bspline.operations import extrapolate, shift_spline
from ..optimization.admm import ADMMState
from ..optimization.x_problem import XProblem
from ..optimization.z_problem import ZProblem


class Vehicle:
    """A single vehicle in the formation."""

    def __init__(self, vehicle_id, config):
        self.id = vehicle_id
        self.config = config

        self.x0 = [0.0] * config.state_len     # [p, q, phi, dp, dq, dphi]
        self.xf = [0.0] * config.state_len
        self.current_position = [0.0, 0.0, 0.0]  # [p, q, phi] in Frenet

        self.neighbours = []    # List of Vehicle references
        self.obstacles = []     # List of Obstacle instances

        self.t_start = 0.0
        self.t_end = config.t_window_size
        self.shift_enabled = False

        # ADMM state (initialized in prepare())
        self.admm = None
        self.x_problem = None
        self.z_problem = None

        # History for analysis
        self.history = {
            'y': [],
            't_start': [],
            't_end': [],
            'x_update_time': [],
            'z_update_time': [],
            'solver_stats': [],
            'feasibility': [],
        }

        # Waypoint data (set by warm-starter)
        self.waypoint_positions = []
        self.waypoint_times = []

    def set_position(self, position, position_type):
        """Set initial or final position."""
        assert len(position) == self.config.n_dimensions
        if position_type == 'initial':
            self.x0 = position + [0.0] * len(position)
            self.current_position = list(position)
        elif position_type == 'final':
            self.xf = position + [0.0] * len(position)
        else:
            raise ValueError(f"Unknown position_type: {position_type}")

    def prepare(self):
        """Build NLP solvers after neighbours and obstacles are set."""
        basis = make_basis(self.config.state_degree, self.config.knot_intervals)
        self.admm = ADMMState(
            n_coeffs=len(basis),
            n_dimensions=self.config.n_dimensions,
            n_neighbours=len(self.neighbours),
            n_obstacles=len(self.obstacles)
        )

        obstacle_radii = [o.max_dist_from_center for o in self.obstacles]

        self.x_problem = XProblem(
            self.config,
            n_obstacles=len(self.obstacles),
            n_neighbours=len(self.neighbours),
            obstacle_radii=obstacle_radii
        )

        neighbour_xf = [n.xf for n in self.neighbours]
        self.z_problem = ZProblem(
            self.config,
            n_neighbours=len(self.neighbours),
            neighbour_xf=neighbour_xf,
            own_x0=self.x0,
            own_xf=self.xf
        )

    def x_update(self, obstacle_params, obstacle_center_params):
        """Run the x-update (trajectory optimization)."""
        # Pack parameters
        params = self.admm.pack_x_params(
            self.x0, self.xf,
            obstacle_params, obstacle_center_params,
            self.waypoint_positions, self.waypoint_times
        )

        # Initial guess
        x0_init = self.admm.pack_x_decision()

        # Solve
        sol, stats, elapsed = self.x_problem.solve(x0_init, params)
        self.admm.unpack_x_decision(sol)

        self.history['y'].append(list(self.admm.y))
        self.history['t_start'].append(self.t_start)
        self.history['t_end'].append(self.t_end)
        self.history['x_update_time'].append(elapsed)
        self.history['solver_stats'].append(stats)
        self.history['feasibility'].append(stats.get('return_status', 'unknown'))

    def z_update(self):
        """Run the z-update (formation consensus)."""
        params = self.admm.pack_z_params()
        x0_init = self.admm.pack_z_decision()

        sol, stats, elapsed = self.z_problem.solve(x0_init, params)
        self.admm.unpack_z_decision(sol)
        self.history['z_update_time'].append(elapsed)

    def shift_splines(self):
        """Shift all spline coefficients forward by t_step after a simulation step."""
        if not self.shift_enabled:
            return

        basis = make_basis(self.config.state_degree, self.config.knot_intervals)
        t_shift = self.config.t_step
        nc = len(basis)
        nd = self.config.n_dimensions

        # Extract new x0 from current solution
        sol = self.admm.y
        coeffs_per_dim = [sol[nc * i: nc * (i + 1)] for i in range(nd)]
        y_splines = [BSpline(basis, np.array(c)) for c in coeffs_per_dim]
        y_dot = [s.derivative() for s in y_splines]

        t_eval = t_shift / self.config.t_window_size
        y0 = [float(s(t_eval)) for s in y_splines]
        y_dot0 = [float(s(t_eval)) for s in y_dot]
        self.x0 = y0 + y_dot0

        # Shift coefficient arrays
        self.admm.y = self._shift_flat(self.admm.y, nd, basis, t_shift)
        self.admm.z_i = self._shift_flat(self.admm.z_i, nd, basis, t_shift)
        self.admm.lambda_i = self._shift_flat(self.admm.lambda_i, nd, basis, t_shift)

        n_obs = self.config.n_dimensions_pos  # a has 2 components per obstacle
        self.admm.a = self._shift_flat(self.admm.a, 2 * len(self.obstacles), basis, t_shift)
        self.admm.b = self._shift_flat(self.admm.b, len(self.obstacles), basis, t_shift)
        self.admm.d_tau = self._shift_flat(self.admm.d_tau, len(self.obstacles), basis, t_shift)

        nn = len(self.neighbours)
        self.admm.z_ij = self._shift_flat(self.admm.z_ij, nd * nn, basis, t_shift)
        self.admm.z_ji = self._shift_flat(self.admm.z_ji, nd * nn, basis, t_shift)
        self.admm.lambda_ij = self._shift_flat(self.admm.lambda_ij, nd * nn, basis, t_shift)
        self.admm.lambda_ji = self._shift_flat(self.admm.lambda_ji, nd * nn, basis, t_shift)

    def _shift_flat(self, flat_coeffs, n_splines, basis, t_shift):
        """Shift flat coefficient array using extrapolate+crop+transform."""
        nc = len(basis)
        result = []
        default_basis = make_basis(basis.degree, len(basis) - basis.degree)

        for i in range(n_splines):
            c = np.array(flat_coeffs[nc * i: nc * (i + 1)])
            spline = BSpline(basis, c)

            # Extrapolate
            new_basis, new_coeffs = extrapolate(spline.coeffs, t_shift, spline.basis)
            # Shift
            shifted_basis, shifted_coeffs = shift_spline(new_coeffs, t_shift, new_basis)
            shifted = BSpline(shifted_basis, shifted_coeffs)
            shifted = shifted.scale(1, -t_shift)
            # Transform back
            final_coeffs = default_basis.transform(shifted.basis).dot(shifted.coeffs)
            result.extend(final_coeffs.tolist())

        return result

    def advance_time(self):
        """Move the time window forward."""
        self.t_start += self.config.t_step
        self.t_end = self.t_start + self.config.t_window_size
        self.t_end = min(self.t_end, 1.0)
        self.t_start = min(self.t_start, 1.0)
        self.shift_enabled = True
