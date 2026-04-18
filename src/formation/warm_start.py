"""Fast formation waypoint generation replacing the DFG grid search.

Uses analytical critical-angle computation and binary search instead
of exhaustive grid search over rotation/scaling candidates.
"""

import math
import numpy as np
from typing import List, Tuple, Optional


class FormationWarmStarter:
    """Generates intermediate waypoints for the formation to pass through obstacles."""

    def __init__(self, config):
        self.config = config
        self.cum_rotation = 0.0
        self.cum_scaling = 1.0

    def generate_waypoints(self, vehicle_positions, obstacles, obstacle_corners_fn,
                           scaled_corners_fn, t_start, t_end, n_waypoints):
        """Generate formation waypoints for the time window [t_start, t_end].

        Args:
            vehicle_positions: List of [p, q] positions for each vehicle.
            obstacles: List of Obstacle objects.
            obstacle_corners_fn: Function(t) -> list of corner lists for each obstacle.
            scaled_corners_fn: Function(t) -> list of scaled corner lists.
            t_start, t_end: Time window.
            n_waypoints: Number of waypoints to generate.

        Returns:
            (waypoint_positions, waypoint_times, cum_rotation, cum_scaling)
            waypoint_positions: flat list of [p, q, phi, p, q, phi, ...] for each waypoint
            waypoint_times: list of normalized times in [0, 1]
        """
        t_step = 0.01
        positions = [list(p[:2]) for p in vehicle_positions]
        original_positions = [list(p[:2]) for p in vehicle_positions]

        all_wp_positions = []
        all_wp_times = []

        t_current = t_start
        t_limit = min(t_end, 1.0)
        while t_current <= t_limit + 1e-7:
            t_eval = min(max(t_current, 0.0), 1.0)
            # Check for collision at current time
            colliding = self._find_colliding_obstacles(
                positions, scaled_corners_fn(t_eval))

            if colliding:
                # Find danger end time
                t_danger_start = t_eval
                t_danger_end = self._find_danger_end(
                    positions, scaled_corners_fn, colliding, t_danger_start, t_limit)

                t_mid = (t_danger_start + t_danger_end) / 2
                t_check = np.linspace(
                    max(0, t_danger_start),
                    min(1, t_danger_end), 5)

                # Find optimal formation
                new_positions, rotation, scaling, action = self._find_formation(
                    positions, obstacle_corners_fn, scaled_corners_fn,
                    colliding, t_check)

                if action in ('back_transformation', 'yes'):
                    self.cum_rotation += rotation
                    self.cum_scaling *= scaling
                    positions = new_positions

                    t_local = np.interp(t_mid, [t_start, t_end], [0, 1])
                    for pos in new_positions:
                        all_wp_positions.extend([pos[0], pos[1], self.cum_rotation])
                    all_wp_times.append(t_local)

                t_current = t_danger_end + t_step
            else:
                t_current += t_step

        # If no waypoints generated, use original positions at the end
        if not all_wp_times:
            t_local = 1.0
            for pos in original_positions:
                all_wp_positions.extend([pos[0], pos[1], self.cum_rotation])
            all_wp_times.append(t_local)

        # Pad/truncate to n_waypoints
        all_wp_positions, all_wp_times = self._pad_waypoints(
            all_wp_positions, all_wp_times, n_waypoints,
            len(vehicle_positions), self.config.n_dimensions)

        return all_wp_positions, all_wp_times

    def _find_colliding_obstacles(self, positions, obstacle_corners_list):
        """Check which obstacles are in the danger zone."""
        colliding = []
        for i, corners in enumerate(obstacle_corners_list):
            for pos in positions:
                if self._point_in_polygon(pos, corners):
                    colliding.append(i)
                    break
        return colliding

    def _find_danger_end(self, positions, scaled_corners_fn, colliding, t_start, t_max):
        """Binary search for the time when the danger zone ends."""
        t_end = t_start
        dt = 0.001
        while t_end < min(t_max, 1.0):
            t_end += dt
            corners = scaled_corners_fn(t_end)
            still_colliding = False
            for idx in colliding:
                for pos in positions:
                    if self._point_in_polygon(pos, corners[idx]):
                        still_colliding = True
                        break
                if still_colliding:
                    break
            if not still_colliding:
                return t_end
        return min(t_end, 1.0)

    def _find_formation(self, positions, corners_fn, scaled_corners_fn,
                        colliding, t_check):
        """Find optimal rotation/scaling to avoid obstacles.

        Uses analytical critical angles instead of grid search.
        """
        # Step 0: Try back-rotation first
        back_rot = -self.config.back_rotation_factor * self.cum_rotation
        if abs(back_rot) < 5 / 360 * 2 * math.pi and back_rot != 0:
            back_rot = -self.cum_rotation

        back_scale = 1.0
        deviance = abs(1 - 1 / self.cum_scaling) * self.config.back_scaling_factor
        if self.cum_scaling >= 1:
            back_scale = 1 - deviance
        else:
            back_scale = 1 + deviance
        if 0.9 <= back_scale <= 1.1 and back_scale != 1.0:
            back_scale = 1.0 / self.cum_scaling

        back_pos = self._rotate(self._scale(positions, back_scale), back_rot)
        if not self._any_collision(back_pos, corners_fn, t_check):
            if back_rot == 0 and back_scale == 1:
                return positions, 0, 1, 'no_action'
            return back_pos, back_rot, back_scale, 'back_transformation'

        # Step 1: Compute critical angles
        scaling_candidates = [1.0, 0.9, 0.8, 0.6, 1.1, 1.2, 1.4, 1.8, 2.0, 2.5]
        best_cost = float('inf')
        best_result = None

        for scale in scaling_candidates:
            scaled_pos = self._scale(positions, scale)
            angles = self._compute_critical_angles(
                scaled_pos, corners_fn, t_check)

            # Also add regular angle steps as fallback
            step = 5 / 360 * 2 * math.pi
            regular = [step * i for i in range(-18, 19) if i != 0]
            all_angles = sorted(set(list(angles) + regular), key=abs)

            for angle in all_angles:
                rotated = self._rotate(scaled_pos, angle)
                if not self._any_collision(rotated, corners_fn, t_check):
                    cost = self._formation_cost(angle, scale)
                    if cost < best_cost:
                        best_cost = cost
                        best_result = (rotated, angle, scale, 'yes')

            if best_result and best_cost < 0.5:
                break  # Good enough, don't try more scales

        if best_result:
            return best_result

        # Step 2: Fallback coarse grid search when analytical candidates fail.
        # This keeps the fast path while improving robustness in hard geometries.
        fallback_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
        fallback_angles = [
            math.radians(deg)
            for deg in range(-180, 181, 5)
            if deg != 0
        ]

        for scale in fallback_scales:
            scaled_pos = self._scale(positions, scale)
            for angle in fallback_angles:
                rotated = self._rotate(scaled_pos, angle)
                if not self._any_collision(rotated, corners_fn, t_check):
                    cost = self._formation_cost(angle, scale)
                    if cost < best_cost:
                        best_cost = cost
                        best_result = (rotated, angle, scale, 'yes')

        if best_result:
            return best_result

        # Fallback: no solution found
        return positions, 0, 1, 'no_solution_found'

    def _compute_critical_angles(self, positions, corners_fn, t_samples):
        """Compute rotation angles that place vehicles exactly on obstacle edges."""
        angles = set()
        for t in t_samples:
            try:
                all_corners = corners_fn(t)
            except:
                continue
            for corners in all_corners:
                if not isinstance(corners, list) or len(corners) < 3:
                    continue
                n_edges = len(corners)
                for pos in positions:
                    r = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
                    if r < 1e-10:
                        continue
                    theta_v = math.atan2(pos[1], pos[0])
                    for e in range(n_edges):
                        p1 = corners[e]
                        p2 = corners[(e + 1) % n_edges]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        rhs = p1[0] * dy - p1[1] * dx
                        A = r * (dy * math.cos(theta_v) - dx * math.sin(theta_v))
                        B = r * (-dy * math.sin(theta_v) - dx * math.cos(theta_v))
                        denom = math.sqrt(A ** 2 + B ** 2)
                        if denom < 1e-10:
                            continue
                        ratio = rhs / denom
                        if abs(ratio) <= 1.0:
                            base = math.atan2(B, A)
                            delta = math.acos(max(-1, min(1, ratio)))
                            angles.add(base + delta)
                            angles.add(base - delta)
        return sorted(angles, key=abs)

    def _any_collision(self, positions, corners_fn, t_samples):
        """Check if any position collides with any obstacle at any time."""
        for t in t_samples:
            try:
                all_corners = corners_fn(t)
            except:
                continue
            for corners in all_corners:
                for pos in positions:
                    if self._point_in_polygon(pos, corners):
                        return True
        return False

    @staticmethod
    def _point_in_polygon(point, corners):
        """Point-in-convex-polygon test robust to CW/CCW corner order."""
        n = len(corners)
        if n < 3:
            return False

        tol = 1e-12
        has_pos = False
        has_neg = False

        for i in range(n):
            x1, y1 = corners[i][0], corners[i][1]
            x2, y2 = corners[(i + 1) % n][0], corners[(i + 1) % n][1]
            cross = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

            if cross > tol:
                has_pos = True
            elif cross < -tol:
                has_neg = True

            if has_pos and has_neg:
                return False

        return True

    @staticmethod
    def _rotate(positions, angle):
        """Rotate all positions around the origin."""
        c, s = math.cos(angle), math.sin(angle)
        return [[p[0] * c - p[1] * s, p[0] * s + p[1] * c] for p in positions]

    @staticmethod
    def _scale(positions, factor):
        """Scale all positions from the origin."""
        return [[p[0] * factor, p[1] * factor] for p in positions]

    @staticmethod
    def _formation_cost(rotation, scaling):
        """Cost function for formation change (prefer minimal change)."""
        alpha_rotation = 0.1
        alpha_scaling_up = 1000
        alpha_scaling_down = 100
        cost = alpha_rotation * abs(rotation)
        if scaling > 1:
            cost += alpha_scaling_up * abs(1 - scaling)
        elif scaling < 1:
            cost += alpha_scaling_down * abs(1 - scaling)
        return cost

    def _pad_waypoints(self, positions, times, n_target, n_vehicles, n_dims):
        """Pad or truncate waypoints to exactly n_target."""
        wp_size = n_vehicles * n_dims
        n_current = len(times)

        if n_current > n_target:
            positions = positions[:n_target * wp_size]
            times = times[:n_target]
        elif n_current < n_target:
            last_pos = positions[-wp_size:]
            last_time = times[-1]
            for _ in range(n_target - n_current):
                positions.extend(last_pos)
                times.append(last_time)

        return positions, times
