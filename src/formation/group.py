"""Group: multi-vehicle formation coordinator with ADMM solve loop."""

import math
import random
import functools
import numpy as np
import os

from ..config import Config
from ..frenet import FrenetPath, SplineFitter
from ..bspline import make_basis
from .vehicle import Vehicle
from .obstacle import Obstacle
from .warm_start import FormationWarmStarter


class Group:
    """Coordinates multiple vehicles in formation."""

    def __init__(self, n_vehicles, config, start_position, goal_position, seed=0):
        self.config = config
        self.frenet_path = FrenetPath()
        self.spline_fitter = SplineFitter()

        self.vehicles = [Vehicle(i, config) for i in range(n_vehicles)]
        self.warm_starter = FormationWarmStarter(config)

        self.start_position = start_position
        self.goal_position = goal_position
        self.seed = seed
        random.seed(seed)

        self._setup_positions()
        self._obstacle_corners_cache = {}

    def _setup_positions(self):
        """Place vehicles on an ellipse around start/goal positions."""
        r = self.config.radius
        start_pos = self._ellipse_positions(
            self.start_position, len(self.vehicles), a=r * 9, b=r * 5)
        goal_pos = self._ellipse_positions(
            self.goal_position, len(self.vehicles), a=r * 9, b=r * 5)

        for i, v in enumerate(self.vehicles):
            v.set_position(start_pos[i], 'initial')
            v.set_position(goal_pos[i], 'final')

    def _ellipse_positions(self, center, n, a, b, rotation=math.pi / 2):
        """Generate n positions on an ellipse."""
        alpha = math.pi / 4
        positions = []
        for i in range(n):
            p = center[0] + a * math.cos(alpha)
            q = center[1] + b * math.sin(alpha)
            positions.append([p, q, center[2] if len(center) > 2 else 0])
            alpha += 2 * math.pi / n

        if rotation != 0:
            for i, pos in enumerate(positions):
                c, s = math.cos(rotation), math.sin(rotation)
                x, y = pos[0], pos[1]
                positions[i] = [x * c - y * s, x * s + y * c, pos[2]]
                if center[:2] != [0.0, 0.0]:
                    positions[i][0] += center[0]
                    positions[i][1] += center[1]
        return positions

    def generate_obstacles(self):
        """Generate random obstacles along the path."""
        obstacles = []
        t_positions = [0.35, 0.65]

        for i, t in enumerate(t_positions):
            cx = random.uniform(-0.1, 0.1)
            cy = random.uniform(-0.1, 0.1)
            a = random.uniform(0.1, 0.7)
            b = random.uniform(0.1, 0.7)
            alpha = random.uniform(-math.pi / 2, math.pi / 2)

            corners = self._generate_rect_corners([cx, cy], a, b, alpha, t)
            obstacle = Obstacle(
                obstacle_id=i,
                corners=corners,
                frenet_path=self.frenet_path,
                spline_fitter=self.spline_fitter,
                fit_knot_intervals=self.config.obstacle_fit_knot_intervals
            )
            obstacles.append(obstacle)

        return obstacles

    def _generate_rect_corners(self, center, a, b, alpha, t):
        """Generate 4 corners of a rotated rectangle in inertial frame."""
        c, s = math.cos(alpha), math.sin(alpha)
        local_corners = [[-a, -b], [a, -b], [a, b], [-a, b]]
        corners = []
        for lc in local_corners:
            x = lc[0] * c - lc[1] * s + center[0]
            y = lc[0] * s + lc[1] * c + center[1]
            # Transform from Frenet at time t to inertial
            ix, iy = self.frenet_path.frenet_to_inertial(x, y, t)
            corners.append([ix, iy])
        return corners

    def add_obstacles(self, obstacles):
        """Assign obstacles to all vehicles."""
        for v in self.vehicles:
            v.obstacles = obstacles

    def setup_neighbours(self):
        """Set up neighbour relationships (fully connected for small groups)."""
        for v in self.vehicles:
            v.neighbours = [other for other in self.vehicles if other.id != v.id]

    def prepare(self):
        """Build NLP solvers for all vehicles."""
        for v in self.vehicles:
            v.prepare()

    def generate_waypoints(self):
        """Run the warm-starter to generate waypoints for all vehicles."""
        if not self.vehicles[0].obstacles:
            # No obstacles, set default waypoints
            for v in self.vehicles:
                v.waypoint_positions = [v.current_position[0], v.current_position[1],
                                        self.warm_starter.cum_rotation] * self.config.n_waypoints
                v.waypoint_times = [1.0] * self.config.n_waypoints
            return

        vehicle_positions = [v.current_position for v in self.vehicles]

        @functools.lru_cache(maxsize=None)
        def get_obstacle_corners(t):
            corners = []
            for obs in self.vehicles[0].obstacles:
                c = [[float(xy(t)) for xy in corner] for corner in obs.corners_spline]
                corners.append(c)
            return corners

        @functools.lru_cache(maxsize=None)
        def get_scaled_corners(t):
            corners = []
            for obs in self.vehicles[0].obstacles:
                c = [[float(xy(t)) for xy in corner] for corner in obs.scaled_corners_spline]
                corners.append(c)
            return corners

        t_start = self.vehicles[0].t_start
        t_end = self.vehicles[0].t_end

        wp_pos, wp_times = self.warm_starter.generate_waypoints(
            vehicle_positions, self.vehicles[0].obstacles,
            get_obstacle_corners, get_scaled_corners,
            t_start, t_end, self.config.n_waypoints
        )

        # Distribute waypoints to vehicles
        n_dims = self.config.n_dimensions
        n_veh = len(self.vehicles)
        for i, v in enumerate(self.vehicles):
            v_wp = []
            for wp_idx in range(len(wp_times)):
                base = wp_idx * n_veh * n_dims + i * n_dims
                v_wp.extend(wp_pos[base:base + n_dims])
            v.waypoint_positions = v_wp
            v.waypoint_times = list(wp_times)

        # Clear cache
        get_obstacle_corners.cache_clear()
        get_scaled_corners.cache_clear()

    def solve_step(self):
        """Execute one complete ADMM iteration."""
        obs_basis = make_basis(self.config.obstacle_degree, self.config.obstacle_knot_intervals)

        # 1. X-update for all vehicles
        for v in self.vehicles:
            t_eval = np.linspace(v.t_start, min(v.t_start + self.config.t_window_size, 1.0),
                                 self.config.t_resolution_length)

            # Pack obstacle parameters
            obs_params = []
            obs_center_params = []
            for obs in v.obstacles:
                t_end = min(t_eval[-1], 1.0)
                cropped = obs.cropped_corners(obs_basis, t_eval[0], t_end)
                for corner in cropped:
                    for xy in corner:
                        obs_params.extend(xy.coeffs.reshape(-1).tolist())

                cropped_center = obs.cropped_center(obs_basis, t_eval[0], t_end)
                for xy in cropped_center:
                    obs_center_params.extend(xy.coeffs.reshape(-1).tolist())

            v.x_update(obs_params, obs_center_params)

        # 2. Exchange x (broadcast y_i to neighbours)
        for v in self.vehicles:
            y_neighbours = []
            for n in v.neighbours:
                y_neighbours.extend(n.admm.y)
            v.admm.y_neighbours = np.array(y_neighbours)

        # 3. Z-update for all vehicles
        for v in self.vehicles:
            v.z_update()

        # 4. Lambda update
        for v in self.vehicles:
            v.admm.lambda_update(self.config.rho)

        # 5. Exchange z (broadcast z_ij, lambda_ij to neighbours)
        for v in self.vehicles:
            z_ji = []
            lambda_ji = []
            nc = v.admm.n_coeffs * v.admm.n_dims
            for n in v.neighbours:
                # Find which index I am in neighbour's neighbour list
                my_idx = next(j for j, nn in enumerate(n.neighbours) if nn.id == v.id)
                z_ji.extend(n.admm.z_ij[nc * my_idx: nc * (my_idx + 1)])
                lambda_ji.extend(n.admm.lambda_ij[nc * my_idx: nc * (my_idx + 1)])
            v.admm.z_ji = np.array(z_ji)
            v.admm.lambda_ji = np.array(lambda_ji)

    def simulation_step(self):
        """Advance all vehicles by one time step."""
        for v in self.vehicles:
            v.shift_splines()
            v.advance_time()
