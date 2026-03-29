"""Spline fitting using CasADi/IPOPT.

Fits B-splines to sampled data by minimizing an L4 loss via nonlinear optimization.
"""

import numpy as np
from casadi import MX, vertcat, nlpsol

from ..bspline import BSpline, BSplineBasis, make_basis


class SplineFitter:
    """Fits B-splines to data using IPOPT."""

    def __init__(self, knot_intervals=10):
        self.knot_intervals = knot_intervals
        self._solver_opts = {
            'print_time': False,
            'ipopt': {'print_level': 0, 'max_iter': 1000, 'max_cpu_time': 100}
        }

    def fit(self, data, y_min=-20, y_max=20, degree=3):
        """Fit B-splines to multi-dimensional sampled data.

        Args:
            data: List of lists, shape [n_dimensions][n_samples].
                  Each inner list contains the sampled values for one dimension.
            y_min: Lower bound on spline coefficients (scalar or list).
            y_max: Upper bound on spline coefficients (scalar or list).
            degree: Degree of the B-spline basis.

        Returns:
            List of BSpline objects, one per dimension.
        """
        n_dims = len(data)
        n_samples = len(data[0])

        if isinstance(y_min, (int, float)):
            y_min = [y_min] * n_dims
        if isinstance(y_max, (int, float)):
            y_max = [y_max] * n_dims

        basis = make_basis(degree=degree, knot_intervals=self.knot_intervals)

        # Decision variables: spline coefficients for each dimension
        w, lbw, ubw = [], [], []
        splines = []
        for k in range(n_dims):
            coeffs = MX.sym(f'y_{k}', len(basis))
            w.append(coeffs)
            lbw.extend([y_min[k]] * len(basis))
            ubw.extend([y_max[k]] * len(basis))
            splines.append(BSpline(basis, coeffs))

        # Cost: L4 loss at sample points
        cost = 0
        for i, t in enumerate(np.linspace(0, 1, n_samples)):
            for j in range(n_dims):
                cost += (splines[j](t) - data[j][i]) ** 4

        # Solve
        prob = {'f': cost, 'x': vertcat(*w), 'g': vertcat(*[])}
        solver = nlpsol('spline_fitter', 'ipopt', prob, self._solver_opts)
        sol = solver.call({'lbx': lbw, 'ubx': ubw, 'lbg': [], 'ubg': []})

        # Extract fitted splines
        all_coeffs = sol['x'].full()
        result = []
        for i in range(n_dims):
            c = all_coeffs[len(basis) * i: len(basis) * (i + 1)]
            result.append(BSpline(basis, c))
        return result
