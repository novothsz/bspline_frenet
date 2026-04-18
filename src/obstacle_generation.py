import math
import random

from .obstacle import Obstacle


def generate_obstacles(group, seed: int = 42):
    # There are the following types of obstacles:
    # - obstacles on the path: these obstacles are created using the ellipse generator algorithm.
    # It's centerpoint, alpha, a, b is a random number in the Frenet frame (all of which in a defined bound).
    # Then, the corners are transformed from Frenet to Inertial.
    # - gates: two corners of each obstacle, that form a gate are generated with the
    # ellipse generator algorithm. However alpha is always zero and b has a minimum value.
    # (both of these constarints ensure, that there is a tunner, kinda parallel with the Frenet path so that
    # the DFG algorithm will be able to find a solution.)
    # The corners are transformed from Frenet to Inertial and extended to the environment limits.
    # - wall on one side: same as the gate, but drops one of the obstacle, that forms a gate.

    # Spacing of the obstacles:
    # randomly, but at least t_spacing between each obstacle.
    # no obstacle is allowed at the end

    # variables
    t_spacing = 0.2
    t_free_begin = 0.2
    t_free_end = 0.2

    n_obst_along = 3
    random.seed(seed)
    centerpoint_x_bound = [-0.1, 0.1]
    centerpoint_y_bound = [-0.1, 0.1]

    a_bound = [0.1, 0.7]
    b_bound = [0.1, 1.5]

    alpha_bound = [-math.pi / 2, math.pi / 2]

    # Generate obstacles along the way
    obstacles = []
    t_bound = []
    t_tmp = [0.4, 0.6, 0.8]
    for i in range(n_obst_along):
        centerpoint = [
            random.uniform(centerpoint_x_bound[0], centerpoint_x_bound[1]),
            random.uniform(centerpoint_y_bound[0], centerpoint_y_bound[1]),
            0,
        ]

        a = random.uniform(a_bound[0], a_bound[1])
        b = random.uniform(b_bound[0], b_bound[1])
        alpha = random.uniform(alpha_bound[0], alpha_bound[1])
        ellipse_corners = group.ellipse_generator(
            centerpoint=centerpoint,
            n_positions=4,
            a=a,
            b=b,
            ellipse_rotation=alpha,
            vehicles_rotation=math.pi / 4,
        )

        obstacle_corners = [corner[:2] for corner in ellipse_corners]
        t = random.uniform(a_bound[0], a_bound[1])
        # We need to place them at random location along the path.
        # This is done by converting their frenet coordinates to the inertial frame at random times.
        obstacle_corners = [group.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i]) for corner in obstacle_corners]
        obstacles += [Obstacle(ID=i, corners=obstacle_corners)]

    # Generate gates
    # Okay. We have generated obstacles along the way.
    # Let's generate gates now! :)
    n_obst_gate = 3
    gate_gap_bound = [3.5 * group.vehicles[0].radious, 10 * group.vehicles[0].radious]
    gate_length_bound = 0.3  # 0.3

    # obstacles = []
    t_tmp = [0.3, 0.5, 0.7]
    for i in range(n_obst_gate):
        gate_points_tmp = random.uniform(gate_gap_bound[0], gate_gap_bound[1])
        # The lower part of the gate
        # corners = [top-right, top_left]
        gate1_inside_corners = [[gate_length_bound / 2, -gate_points_tmp], [-gate_length_bound / 2, -gate_points_tmp]]

        # corners = [bottom-left, bottom-right]
        gate2_inside_corners = [[gate_length_bound / 2, gate_points_tmp], [-gate_length_bound / 2, gate_points_tmp]]

        # transforming the frenet coordinates to inertial frame at random times
        g1 = [list(group.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i])) for corner in gate1_inside_corners]
        g2 = [list(group.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i])) for corner in gate2_inside_corners]

        if i == -1:
            pass
        else:
            # Extending till the edge of the environment
            g1 = g1 + [[g1[-1][0], -6]] + [[g1[0][0], -6]]
            g2 = g2 + [[g2[-1][0], 6]] + [[g2[0][0], 6]]
            obstacles += [Obstacle(ID=3 + i * 2, corners=g1)]
            obstacles += [Obstacle(ID=3 + i * 2 + 1, corners=g2)]
            # Sharing ID-s between gate pairs
            obstacles[-2].gate_pair_ID = obstacles[-1].ID
            obstacles[-1].gate_pair_ID = obstacles[-2].ID

    return obstacles