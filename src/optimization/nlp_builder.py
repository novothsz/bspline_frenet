"""NLP construction helpers for CasADi/IPOPT optimization.

Extracted and cleaned from VehicleBasis. Accumulates decision variables,
parameters, constraints, and cost for building CasADi NLP problems.
"""

import math
import random
import numpy as np
from casadi import MX, vertcat, dot, nlpsol, cos, sin

from ..bspline import BSpline, BSplineBasis, make_basis, definite_integral


class NLPBuilder:
    """Accumulates CasADi MX variables, constraints, and cost."""

    def __init__(self):
        self.w = []       # decision variables (MX)
        self.lbw = []     # lower bounds
        self.ubw = []     # upper bounds
        self.w0 = []      # initial values
        self.w_labels = []  # labels for tracking

        self.g = []       # constraints (MX)
        self.lbg = []     # constraint lower bounds
        self.ubg = []     # constraint upper bounds

        self.P = []       # parameters (MX)
        self.P_labels = []  # parameter labels
        self.P0 = []      # default parameter values

        self.cost = 0     # cost expression

    def add_spline_var(self, name, basis, n_splines, lb, ub, init=None):
        """Create n_splines BSpline decision variables.

        Args:
            name: Name prefix for the MX symbols.
            basis: BSplineBasis to use.
            n_splines: Number of splines.
            lb: Lower bounds (list, one per spline).
            ub: Upper bounds (list, one per spline).
            init: Initial values. None=random, list of [start,end] pairs=linspace,
                  list of full coefficient arrays=use directly.

        Returns:
            numpy array of BSpline objects with MX coefficients.
        """
        splines = []
        for k in range(n_splines):
            coeffs = MX.sym(name, len(basis))
            self.w.append(coeffs)
            self.w_labels.extend([name] * len(basis))

            # Initial values
            if init and len(init) > k:
                iv = init[k]
                if len(iv) == len(basis):
                    self.w0.extend(iv)
                elif iv[0] is not None and iv[1] is not None:
                    w0 = np.linspace(iv[0], iv[1], len(basis))
                    w0 += w0 * 0.05 * np.array([random.uniform(-0.5, 0.5) for _ in range(len(basis))])
                    self.w0.extend(w0.tolist())
                else:
                    self.w0.extend([random.uniform(-0.5, 0.5) for _ in range(len(basis))])
            else:
                self.w0.extend([random.uniform(-0.5, 0.5) for _ in range(len(basis))])

            # Bounds
            self.lbw.extend([lb[k]] * len(basis))
            self.ubw.extend([ub[k]] * len(basis))

            splines.append(BSpline(basis, coeffs))
        return np.array(splines)

    def add_spline_param(self, name, basis, n_splines):
        """Create n_splines BSpline parameters (not optimized).

        Returns:
            numpy array of BSpline objects with MX coefficients (as parameters).
        """
        splines = []
        for k in range(n_splines):
            coeffs = MX.sym(name, len(basis))
            self.P.append(coeffs)
            self.P_labels.extend([name] * len(basis))
            self.P0.extend([0] * len(basis))
            splines.append(BSpline(basis, coeffs))
        return np.array(splines)

    def add_scalar_param(self, name, size):
        """Add a scalar/vector parameter.

        Returns:
            MX symbol.
        """
        p = MX.sym(name, size)
        self.P.append(p)
        self.P_labels.extend([name] * size)
        self.P0.extend([0] * size)
        return p

    def add_constraint_coeffs(self, splines, lb, ub):
        """Add coefficient-level constraints (bounds on all coefficients)."""
        for i, s in enumerate(splines):
            for j in range(s.coeffs.shape[0]):
                self.g.append(s.coeffs[j])
                self.lbg.append(lb[i])
                self.ubg.append(ub[i])

    def add_constraint_initial(self, splines, lb, ub):
        """Constrain the first coefficient of each spline."""
        for i, s in enumerate(splines):
            self.g.append(s.coeffs[0])
            self.lbg.append(lb[i])
            self.ubg.append(ub[i])

    def add_constraint_final(self, splines, lb, ub):
        """Constrain the last coefficient of each spline."""
        for i, s in enumerate(splines):
            self.g.append(s.coeffs[-1])
            self.lbg.append(lb[i])
            self.ubg.append(ub[i])

    def add_constraint_initial_param(self, splines, lb_param, ub_param):
        """Constrain first coefficient to match a parameter value (as inequality)."""
        for i in range(lb_param.shape[0]):
            self.g.append(splines[i].coeffs[0] - lb_param[i])
            self.lbg.append(0)
            self.ubg.append(math.inf)
        for i in range(ub_param.shape[0]):
            self.g.append(splines[i].coeffs[0] - ub_param[i])
            self.lbg.append(-math.inf)
            self.ubg.append(0)

    def add_constraint_final_param(self, splines, lb_param, ub_param):
        """Constrain last coefficient to match a parameter value (as inequality)."""
        for i in range(lb_param.shape[0]):
            self.g.append(splines[i].coeffs[-1] - lb_param[i])
            self.lbg.append(0)
            self.ubg.append(math.inf)
        for i in range(ub_param.shape[0]):
            self.g.append(splines[i].coeffs[-1] - ub_param[i])
            self.lbg.append(-math.inf)
            self.ubg.append(0)

    def add_constraint_eval(self, expr_list, lb, ub):
        """Add point-evaluation constraints (scalar MX expressions)."""
        for i, expr in enumerate(expr_list):
            self.g.append(expr)
            for j in range(expr.shape[0]):
                self.lbg.append(lb[i])
                self.ubg.append(ub[i])

    def add_hyperplane_avoidance(self, vehicle_pq, obstacle_corners, radius,
                                  center_circle, config):
        """Separating hyperplane obstacle avoidance.

        Creates decision variables a (normal), b (offset), d_tau (slack)
        and adds the three constraint groups from the paper.

        Args:
            vehicle_pq: [p_spline, q_spline] vehicle position splines.
            obstacle_corners: list of [corner_x_spline, corner_y_spline] for each corner.
            radius: vehicle safety radius.
            center_circle: [center_splines, circle_radius] or empty.
            config: Config object with knot_intervals, epsilon, safety_weight.

        Returns:
            a: hyperplane normal splines (for warm-starting).
        """
        basis = make_basis(degree=3, knot_intervals=config.knot_intervals)

        # Hyperplane normal vector a (2D)
        a = self.add_spline_var('a', basis, 2,
                                lb=[-math.inf, -math.inf],
                                ub=[math.inf, math.inf],
                                init=[[1, -1], [0, 0]])

        # Hyperplane offset b (scalar)
        b = self.add_spline_var('b', basis, 1,
                                lb=[-math.inf], ub=[math.inf])

        # Slack variable d_tau (non-negative)
        d_tau = self.add_spline_var('d_tau', basis, 1,
                                    lb=[0], ub=[math.inf])

        # Constraint 1: a^T * vehicle_pos - b <= -radius (vehicle on correct side)
        const1 = a[0] * vehicle_pq[0] + a[1] * vehicle_pq[1] - b[0]
        self.add_constraint_coeffs([const1], [-math.inf], [-radius])

        # Also ensure vehicle is outside center circle
        if center_circle:
            self.add_constraint_coeffs([const1], [-math.inf], [center_circle[1]])

        # Constraint 2: a^T * corner - b - d_tau >= 0 (obstacle on other side)
        for corner in obstacle_corners:
            c2 = a[0] * corner[0] + a[1] * corner[1] - b[0] - d_tau[0]
            self.add_constraint_coeffs([c2], [0], [math.inf])

        # Also for center circle
        if center_circle:
            c2_center = a[0] * center_circle[0][0] + a[1] * center_circle[0][1] - b[0] - d_tau[0]
            self.add_constraint_coeffs([c2_center], [0], [math.inf])

        # Constraint 3: ||a||^2 <= 1
        const3 = a[0] * a[0] + a[1] * a[1]
        self.add_constraint_coeffs([const3], [0.0], [1.0])

        # Cost: penalize slack to encourage tight avoidance
        self.cost += config.safety_weight * definite_integral(
            (config.epsilon - d_tau[0]) ** 2, 0, 1)

        return a

    def build_solver(self, name, options):
        """Build the CasADi NLP solver.

        Returns:
            (solver, arg_template) tuple.
        """
        prob = {
            'f': self.cost,
            'x': vertcat(*self.w),
            'g': vertcat(*self.g),
            'p': vertcat(*self.P)
        }
        solver = nlpsol(name, 'ipopt', prob, options)
        arg = {
            'x0': self.w0,
            'lbx': self.lbw,
            'ubx': self.ubw,
            'lbg': self.lbg,
            'ubg': self.ubg,
            'p': self.P0,
        }
        return solver, arg
