import math
import numpy as np


def _compute_critical_angles(group, vehicle_positions, t_values):
    """Compute analytical edge-touch angles for vehicle points and obstacle edges."""
    if type(t_values) != list and type(t_values) != type(np.array([])):
        t_iterable = [t_values]
    else:
        t_iterable = t_values

    angles = set()
    for t_ in t_iterable:
        obstacle_corners_all = group.get_obstacle_corners(t_)
        for corners in obstacle_corners_all:
            if len(corners) < 3:
                continue

            n_edges = len(corners)
            for pos in vehicle_positions:
                r = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
                if r < group.TOL:
                    continue

                theta_v = math.atan2(pos[1], pos[0])
                for edge_idx in range(n_edges):
                    p1 = corners[edge_idx]
                    p2 = corners[(edge_idx + 1) % n_edges]

                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]

                    rhs = p1[0] * dy - p1[1] * dx
                    A = r * (dy * math.cos(theta_v) - dx * math.sin(theta_v))
                    B = r * (-dy * math.sin(theta_v) - dx * math.cos(theta_v))
                    denom = math.sqrt(A ** 2 + B ** 2)

                    if denom < group.TOL:
                        continue

                    ratio = rhs / denom
                    if abs(ratio) <= 1:
                        base = math.atan2(B, A)
                        delta = math.acos(max(-1, min(1, ratio)))
                        angles.add(base + delta)
                        angles.add(base - delta)

    return sorted(angles, key=abs)


def find_analytic_candidate(group, vehicle_positions, t_values):
    """Find best non-colliding formation transform using critical-angle candidates."""
    scaling_factors = [1, 0.9, 0.8, 0.6, 1.1, 1.2, 1.4, 1.8, 2.0, 2.5, 3.0, 4.0]
    degree_step = 5
    radian_step = degree_step / 360 * 2 * math.pi

    best_cost = math.inf
    best_positions = None
    best_rotation = 0
    best_scaling = 1

    for scaling_factor in scaling_factors:
        vehicle_positions_scaled = group.scale_formation(vehicle_positions, scaling_factor)
        critical_angles = _compute_critical_angles(group, vehicle_positions_scaled, t_values)
        regular_angles = [radian_step * i for i in range(-36, 37) if i != 0]
        candidate_angles = sorted(set(critical_angles + regular_angles), key=abs)

        for rotation_angle in candidate_angles:
            vehicle_positions_scaled_rotated = group.rotate_formation(vehicle_positions_scaled, rotation_angle)

            collision_saved_tmp = []
            for t_ in t_values:
                all_collisions, collision = group.check_collision_with_obstacles(
                    vehicle_positions_scaled_rotated, group.get_obstacle_corners(t_)
                )
                collision_saved_tmp += [collision]
                if collision:
                    break

            if all(collision is False for collision in collision_saved_tmp):
                cost = group.formation_change_cost_calculator(
                    vehicle_positions,
                    vehicle_positions_scaled_rotated,
                    rotation_angle,
                    scaling_factor,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_positions = vehicle_positions_scaled_rotated
                    best_rotation = rotation_angle
                    best_scaling = scaling_factor

        if best_positions is not None and best_cost < 0.5:
            break

    if best_positions is None:
        return None
    return best_positions, best_rotation, best_scaling
