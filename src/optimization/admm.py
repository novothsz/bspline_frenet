"""ADMM state management for distributed formation control.

Replaces the ParamValX, ParamValZ, DecisionVarX, DecisionVarZ classes
with a single ADMMState using pre-computed index arrays.
"""

import numpy as np
from ..bspline import BSpline, make_basis
from ..bspline.operations import extrapolate, crop_spline


class ADMMState:
    """Stores all ADMM primal, dual, and consensus variables for one vehicle."""

    def __init__(self, n_coeffs, n_dimensions, n_neighbours, n_obstacles):
        """
        Args:
            n_coeffs: Number of coefficients per spline (len(basis)).
            n_dimensions: Number of state dimensions (3: p, q, phi).
            n_neighbours: Number of neighbouring vehicles.
            n_obstacles: Number of obstacles.
        """
        self.n_coeffs = n_coeffs
        self.n_dims = n_dimensions
        self.n_neighbours = n_neighbours
        self.n_obstacles = n_obstacles

        dim = n_coeffs * n_dimensions

        # X-update decision variables
        self.y = np.zeros(dim)                           # own trajectory
        self.a = np.zeros(n_coeffs * 2 * n_obstacles)   # hyperplane normals
        self.b = np.zeros(n_coeffs * n_obstacles)        # hyperplane offsets
        self.d_tau = np.zeros(n_coeffs * n_obstacles)    # hyperplane slack

        # Z-update decision variables
        self.z_i = np.zeros(dim)                         # own consensus
        self.z_ij = np.zeros(dim * n_neighbours)         # copies for neighbours

        # Dual variables
        self.lambda_i = np.zeros(dim)
        self.lambda_ij = np.zeros(dim * n_neighbours)

        # Messages from neighbours
        self.y_neighbours = np.zeros(dim * n_neighbours)  # y_j
        self.z_ji = np.zeros(dim * n_neighbours)           # z_j->i
        self.lambda_ji = np.zeros(dim * n_neighbours)      # lambda_j->i

    def lambda_update(self, rho):
        """Standard ADMM dual variable update."""
        self.lambda_i = self.lambda_i + rho * (self.y - self.z_i)
        # lambda_ij uses y_neighbours (y_j from neighbours)
        self.lambda_ij = self.lambda_ij + rho * (self.y_neighbours - self.z_ij)

    def pack_x_decision(self):
        """Pack x-update decision variables into a flat array."""
        return np.concatenate([self.y, self.a, self.b, self.d_tau])

    def unpack_x_decision(self, sol_flat):
        """Unpack x-update solution into individual arrays."""
        nc = self.n_coeffs
        nd = self.n_dims
        no = self.n_obstacles
        idx = 0
        self.y = sol_flat[idx:idx + nc * nd]; idx += nc * nd
        self.a = sol_flat[idx:idx + nc * 2 * no]; idx += nc * 2 * no
        self.b = sol_flat[idx:idx + nc * no]; idx += nc * no
        self.d_tau = sol_flat[idx:idx + nc * no]; idx += nc * no

    def pack_z_decision(self):
        """Pack z-update decision variables."""
        return np.concatenate([self.z_i, self.z_ij])

    def unpack_z_decision(self, sol_flat):
        """Unpack z-update solution."""
        dim = self.n_coeffs * self.n_dims
        self.z_i = sol_flat[:dim]
        self.z_ij = sol_flat[dim:]

    def pack_x_params(self, x0, xf, obstacle_params, obstacle_center_params,
                       waypoint_positions, waypoint_times):
        """Pack all x-update parameters into a flat array.

        The order must match how the NLP was constructed in XProblem.
        """
        parts = [np.array(x0), np.array(xf)]
        # y spline initial values are in w0, not params

        # Waypoint parameters
        for i in range(len(waypoint_times)):
            idx_start = (self.n_dims) * i
            idx_end = idx_start + self.n_dims
            parts.append(np.array(waypoint_positions[idx_start:idx_end]))
            parts.append(np.array([waypoint_times[i]]))

        # Obstacle corner spline coefficients
        parts.append(np.array(obstacle_params))

        # Obstacle center spline coefficients
        parts.append(np.array(obstacle_center_params))

        # ADMM consensus parameters: z_i, lambda_i, z_ji, lambda_ji
        parts.append(self.z_i)
        parts.append(self.lambda_i)
        parts.append(self.z_ji)
        parts.append(self.lambda_ji)

        return np.concatenate(parts)

    def pack_z_params(self):
        """Pack z-update parameters."""
        parts = [
            self.y,           # y_i
            self.lambda_i,    # lambda_i
        ]
        # For each neighbour: y_j, lambda_ij
        dim = self.n_coeffs * self.n_dims
        for i in range(self.n_neighbours):
            s = dim * i
            e = dim * (i + 1)
            parts.append(self.y_neighbours[s:e])
            parts.append(self.lambda_ij[s:e])
        return np.concatenate(parts)


def shift_spline_coefficients(coeffs_flat, n_splines, t_shift, basis):
    """Shift a flat array of spline coefficients forward by t_shift.

    Uses the extrapolate+crop+transform method from VehicleBasis.shift_spline_v2.
    """
    nc = len(basis)
    result = []
    for i in range(n_splines):
        c = coeffs_flat[nc * i: nc * (i + 1)]
        spline = BSpline(basis, np.array(c))

        # Extrapolate, crop, and transform back to original basis
        default_basis = make_basis(degree=basis.degree,
                                   knot_intervals=len(basis) - basis.degree)
        new_basis, new_coeffs = extrapolate(spline.coeffs, t_shift, spline.basis)
        shifted_spline = BSpline(new_basis, new_coeffs)

        from ..bspline.operations import shift_spline
        new_basis2, shifted_coeffs = shift_spline(new_coeffs, t_shift, new_basis)
        shifted_spline2 = BSpline(new_basis2, shifted_coeffs)
        shifted_spline2 = shifted_spline2.scale(1, -t_shift)

        final_coeffs = default_basis.transform(shifted_spline2.basis).dot(shifted_spline2.coeffs)
        result.extend(final_coeffs.tolist())
    return result
