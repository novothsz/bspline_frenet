"""Frenet frame path definition and coordinate transformations.

Defines a sinusoidal reference path that the formation tracks.
All functions are expressed analytically (no autograd dependency).
"""

import numpy as np
import math
import pickle
import os
from numpy import interp

from ..bspline import BSpline, BSplineBasis, make_basis


class FrenetPath:
    """A reference path in the inertial frame, providing Frenet coordinate transforms.

    The path is parameterized by tau in [tau_0, tau_f], mapped to t in [0, 1].
    Path shape: x = tau, y = sin(tau / (pi/2)), z = 0.1*tau.
    """

    def __init__(self, tau_0=-2 * math.pi - math.pi, tau_f=2 * math.pi + math.pi,
                 pickle_path='coeffs.pickle', knot_intervals=10):
        self.tau_0 = tau_0
        self.tau_f = tau_f
        self.knot_intervals = knot_intervals
        self._pickle_path = pickle_path

        # Spline representations (populated by fit_all)
        self.fx_spline = None
        self.fy_spline = None
        self.fz_spline = None
        self.fx_d_spline = None
        self.fy_d_spline = None
        self.fz_d_spline = None
        self.fx_c_spline = None
        self.fy_c_spline = None
        self.fz_c_spline = None
        self.cos_f_theta_spline = None
        self.sin_f_theta_spline = None

        self.fit_all()

    # --- Path functions (analytical) ---

    def fx(self, tau):
        return tau * 1.0

    def fy(self, tau):
        return np.sin(tau / (math.pi / 2.0))

    def fz(self, tau):
        return 0.1 * tau

    # --- First derivatives (analytical, replaces autograd) ---

    def fx_d(self, tau):
        return 1.0

    def fy_d(self, tau):
        return (1.0 / (math.pi / 2.0)) * np.cos(tau / (math.pi / 2.0))

    def fz_d(self, tau):
        return 0.1

    # --- Second derivatives (analytical) ---

    def fx_dd(self, tau):
        return 0.0

    def fy_dd(self, tau):
        return -(1.0 / (math.pi / 2.0)) ** 2 * np.sin(tau / (math.pi / 2.0))

    def fz_dd(self, tau):
        return 0.0

    # --- Derived quantities ---

    def f_theta(self, tau):
        """Heading angle of the path at parameter tau."""
        return math.atan2(self.fy_d(tau) / self.fx_d(tau), 1)

    def fx_c(self, tau):
        """Curvature of fx."""
        return abs(self.fx_dd(tau)) * (1 + self.fx_d(tau) ** 2) ** -1.5

    def fy_c(self, tau):
        """Curvature of fy."""
        return abs(self.fy_dd(tau)) * (1 + self.fy_d(tau) ** 2) ** -1.5

    def fz_c(self, tau):
        """Curvature of fz."""
        return abs(self.fz_dd(tau)) * (1 + self.fz_d(tau) ** 2) ** -1.5

    # --- Parameter mapping ---

    def t_to_tau(self, t):
        """Map normalized time t in [0,1] to path parameter tau."""
        return interp(t, [0, 1], [self.tau_0, self.tau_f])

    def tau_to_t(self, tau):
        """Map path parameter tau to normalized time t in [0,1]."""
        return interp(tau, [self.tau_0, self.tau_f], [0, 1])

    # --- Coordinate transforms ---

    def inertial_to_frenet(self, x, y, t):
        """Transform (x, y) in inertial frame to (p, q) in Frenet frame at time t."""
        tau = self.t_to_tau(t)
        x_f = self.fx(tau)
        y_f = self.fy(tau)
        theta = self.f_theta(tau)
        dx = x - x_f
        dy = y - y_f
        p = dx * math.cos(theta) + dy * math.sin(theta)
        q = -dx * math.sin(theta) + dy * math.cos(theta)
        return p, q

    def frenet_to_inertial(self, p, q, t):
        """Transform (p, q) in Frenet frame to (x, y) in inertial frame at time t."""
        tau = self.t_to_tau(t)
        x_f = self.fx(tau)
        y_f = self.fy(tau)
        theta = self.f_theta(tau)
        x = p * math.cos(theta) - q * math.sin(theta)
        y = p * math.sin(theta) + q * math.cos(theta)
        return x_f + x, y_f + y

    # --- Spline fitting ---

    def fit_all(self):
        """Fit spline representations of the path and its derivatives.

        Results are cached to pickle for fast subsequent loads.
        """
        if self._try_load_pickle():
            return self

        from .spline_fitter import SplineFitter
        fitter = SplineFitter(knot_intervals=self.knot_intervals)
        tau_samples = np.linspace(self.tau_0, self.tau_f, 100).tolist()

        # Fit all path functions and derivatives
        self.fx_spline = fitter.fit([self._sample(self.fx, tau_samples)])[0]
        self.fx_d_spline = fitter.fit([self._sample(self.fx_d, tau_samples)])[0]
        self.fx_c_spline = fitter.fit([self._sample(self.fx_c, tau_samples)])[0]

        self.fy_spline = fitter.fit([self._sample(self.fy, tau_samples)])[0]
        self.fy_d_spline = fitter.fit([self._sample(self.fy_d, tau_samples)])[0]
        self.fy_c_spline = fitter.fit([self._sample(self.fy_c, tau_samples)])[0]

        self.fz_spline = fitter.fit([self._sample(self.fz, tau_samples)])[0]
        self.fz_d_spline = fitter.fit([self._sample(self.fz_d, tau_samples)])[0]
        self.fz_c_spline = fitter.fit([self._sample(self.fz_c, tau_samples)])[0]

        self.cos_f_theta_spline = fitter.fit(
            [self._sample(lambda t: np.cos(self.f_theta(t)), tau_samples)])[0]
        self.sin_f_theta_spline = fitter.fit(
            [self._sample(lambda t: np.sin(self.f_theta(t)), tau_samples)])[0]

        self._save_pickle()
        return self

    def _sample(self, func, tau_samples):
        """Sample a function at tau_samples, returning a flat list."""
        return [float(func(tau)) for tau in tau_samples]

    def _try_load_pickle(self):
        """Try to load pre-fitted spline coefficients from pickle."""
        if not os.path.exists(self._pickle_path):
            return False
        try:
            with open(self._pickle_path, 'rb') as f:
                coeffs = pickle.load(f)

            basis = make_basis(degree=3, knot_intervals=self.knot_intervals)

            self.fx_spline = BSpline(basis, coeffs['fx_spline'])
            self.fx_d_spline = BSpline(basis, coeffs['fx_d_spline'])
            self.fx_c_spline = BSpline(basis, coeffs['fx_c_spline'])
            self.fy_spline = BSpline(basis, coeffs['fy_spline'])
            self.fy_d_spline = BSpline(basis, coeffs['fy_d_spline'])
            self.fy_c_spline = BSpline(basis, coeffs['fy_c_spline'])
            self.fz_spline = BSpline(basis, coeffs['fz_spline'])
            self.fz_d_spline = BSpline(basis, coeffs['fz_d_spline'])
            self.fz_c_spline = BSpline(basis, coeffs['fz_c_spline'])
            self.cos_f_theta_spline = BSpline(basis, coeffs['cos_f_theta_spline'])
            self.sin_f_theta_spline = BSpline(basis, coeffs['sin_f_theta_spline'])
            return True
        except Exception:
            return False

    def _save_pickle(self):
        """Save fitted spline coefficients to pickle."""
        coeffs = {
            'fx_spline': self.fx_spline.coeffs,
            'fx_d_spline': self.fx_d_spline.coeffs,
            'fx_c_spline': self.fx_c_spline.coeffs,
            'fy_spline': self.fy_spline.coeffs,
            'fy_d_spline': self.fy_d_spline.coeffs,
            'fy_c_spline': self.fy_c_spline.coeffs,
            'fz_spline': self.fz_spline.coeffs,
            'fz_d_spline': self.fz_d_spline.coeffs,
            'fz_c_spline': self.fz_c_spline.coeffs,
            'cos_f_theta_spline': self.cos_f_theta_spline.coeffs,
            'sin_f_theta_spline': self.sin_f_theta_spline.coeffs,
        }
        with open(self._pickle_path, 'wb') as f:
            pickle.dump(coeffs, f)

    # --- Plotting helper ---

    def plot_path(self, ax, t_start=0):
        """Plot the reference path and current Frenet frame position."""
        t = np.linspace(0, 1, 100)
        fx_ = [self.fx(self.t_to_tau(t_)) for t_ in t]
        fy_ = [self.fy(self.t_to_tau(t_)) for t_ in t]
        ax.plot(fx_, fy_, label='Frenet path', linestyle=':', color='gray', zorder=2)

        # Current position and orientation
        x_cur, y_cur = self.frenet_to_inertial(0, 0, t_start)
        theta = math.atan2(
            float(self.fy_d_spline(t_start)),
            float(self.fx_d_spline(t_start))
        )
        arrow_len = 0.1
        # Along-path arrow
        ax.arrow(x_cur, y_cur,
                 arrow_len * math.cos(theta), arrow_len * math.sin(theta),
                 head_width=0.01, head_length=0.02, fc='r', ec='r', zorder=3)
        # Normal arrow
        ax.arrow(x_cur, y_cur,
                 arrow_len * math.cos(theta + math.pi / 2),
                 arrow_len * math.sin(theta + math.pi / 2),
                 head_width=0.01, head_length=0.02, fc='r', ec='r', zorder=3)
        return ax
