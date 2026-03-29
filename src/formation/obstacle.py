"""Obstacle representation with spline-fitted corners in the Frenet frame."""

import numpy as np
from ..bspline import BSpline, BSplineBasis, make_basis
from ..bspline.operations import crop_spline


class Obstacle:
    """A rectangular obstacle with corner trajectories in the moving Frenet frame."""

    def __init__(self, obstacle_id, corners, frenet_path, spline_fitter,
                 safety_margin=0.03, fit_knot_intervals=20):
        """
        Args:
            obstacle_id: Unique ID for this obstacle.
            corners: List of [x, y] corners in inertial frame.
            frenet_path: FrenetPath instance (shared).
            spline_fitter: SplineFitter instance (shared).
            safety_margin: Margin to expand corners for danger zone detection.
            fit_knot_intervals: Knot intervals for fitting corner splines.
        """
        self.ID = obstacle_id
        self.corners = corners
        self.center = np.mean(corners, axis=0).tolist()
        self.max_dist_from_center = max(
            np.linalg.norm(np.array(c) - np.array(self.center))
            for c in corners
        )
        self.gate_pair_id = None

        # Fit corners into Frenet frame splines
        self.corners_spline = self._fit_corners(
            corners, frenet_path, spline_fitter, fit_knot_intervals)

        # Scaled (expanded) corners for danger zone detection
        self.scaled_corners = self._compute_scaled_corners(safety_margin)
        self.scaled_corners_spline = self._fit_corners(
            self.scaled_corners, frenet_path, spline_fitter, fit_knot_intervals)

        # Center spline
        self.center_spline = self._fit_center(
            frenet_path, spline_fitter, fit_knot_intervals)

    def _fit_corners(self, corners, frenet_path, fitter, knot_intervals):
        """Fit corner trajectories in the Frenet frame."""
        old_ki = fitter.knot_intervals
        fitter.knot_intervals = knot_intervals
        t_samples = np.linspace(0, 1, 100)
        result = []
        for corner in corners:
            pq = np.array([
                frenet_path.inertial_to_frenet(corner[0], corner[1], t)
                for t in t_samples
            ])
            fitted = fitter.fit([pq[:, 0].tolist(), pq[:, 1].tolist()],
                                y_min=-20, y_max=20)
            result.append(fitted)
        fitter.knot_intervals = old_ki
        return result

    def _fit_center(self, frenet_path, fitter, knot_intervals):
        """Fit center trajectory in the Frenet frame."""
        old_ki = fitter.knot_intervals
        fitter.knot_intervals = knot_intervals
        t_samples = np.linspace(0, 1, 100)
        pq = np.array([
            frenet_path.inertial_to_frenet(self.center[0], self.center[1], t)
            for t in t_samples
        ])
        fitted = fitter.fit([pq[:, 0].tolist(), pq[:, 1].tolist()],
                            y_min=-20, y_max=20)
        fitter.knot_intervals = old_ki
        return fitted

    def _compute_scaled_corners(self, margin):
        """Expand corners outward by margin for danger zone detection."""
        center = np.array(self.center)
        scaled = []
        for corner in self.corners:
            c = np.array(corner)
            direction = c - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                scaled.append((c + direction / norm * margin).tolist())
            else:
                scaled.append(corner)
        return scaled

    def cropped_corners(self, default_basis, t_start, t_end):
        """Return corner splines cropped to [t_start, t_end] and normalized to [0, 1]."""
        eps = 1e-5
        result = []
        for corner_splines in self.corners_spline:
            cropped_corner = []
            for xy_spline in corner_splines:
                xy = crop_spline(xy_spline, t_start, min(t_end, 1.0))
                xy = xy.scale(1, -t_start)
                xy = xy.scale(1.0 / (t_end - t_start), 0)
                # Fix knot endpoints
                xy.basis.knots[:default_basis.degree] = 0.0
                xy.basis.knots[0] = -eps
                xy.basis.knots[-default_basis.degree:] = 1.0
                xy.basis.knots[-1] = 1.0 + eps
                # Transform to default basis
                new_coeffs = default_basis.transform(xy.basis).dot(xy.coeffs)
                cropped_corner.append(BSpline(default_basis, new_coeffs))
            result.append(cropped_corner)
        return result

    def cropped_center(self, default_basis, t_start, t_end):
        """Return center spline cropped and normalized."""
        eps = 1e-5
        result = []
        for xy_spline in self.center_spline:
            xy = crop_spline(xy_spline, t_start, min(t_end, 1.0))
            xy = xy.scale(1, -t_start)
            xy = xy.scale(1.0 / (t_end - t_start), 0)
            xy.basis.knots[:default_basis.degree] = 0.0
            xy.basis.knots[0] = -eps
            xy.basis.knots[-default_basis.degree:] = 1.0
            xy.basis.knots[-1] = 1.0 + eps
            new_coeffs = default_basis.transform(xy.basis).dot(xy.coeffs)
            result.append(BSpline(default_basis, new_coeffs))
        return result
