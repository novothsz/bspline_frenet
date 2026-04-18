import math
import random
import time

import matplotlib.pyplot as plt

from src import (
    GroupRuntimeConfig,
    Obstacle,
    attach_obstacles,
    create_group,
    ensure_output_dirs,
    set_ellipse_initial_final,
    warmup_mpc,
)


def _default_obstacles():
    obstacles = []

    obstacles.append(
        Obstacle(
            ID=0,
            corners=([0.2, -0.25], [0.52, -0.02], [0.15, 0.5], [-0.2, 0.23]),
        )
    )

    dx = 0.3
    dy = 0.3
    obstacles.append(
        Obstacle(
            ID=1,
            corners=([-2.4674 - dx, -1 - dy], [-2.4674 + dx, -1 - dy], [-2.4674 + dx, -6], [-2.4674 - dx, -6]),
        )
    )
    obstacles.append(
        Obstacle(
            ID=2,
            corners=([-2.4674 - dx, -1 + dy], [-2.4674 + dx, -1 + dy], [-2.4674 + dx, 6], [-2.4674 - dx, 6]),
        )
    )
    obstacles.append(
        Obstacle(
            ID=3,
            corners=([2.474 - dx, 1 - dy], [2.474 + dx, 1 - dy], [2.474 + dx, -6], [2.474 - dx, -6]),
        )
    )
    obstacles.append(
        Obstacle(
            ID=4,
            corners=([2.474 - dx, 1 + dy], [2.474 + dx, 1 + dy], [2.474 + dx, 6], [2.474 - dx, 6]),
        )
    )

    return obstacles


def run_optimization(start_position, goal_position, stage=0):
    plt.close("all")
    seed = 64
    random.seed(seed)
    print("Seed was:", seed)

    group = create_group(start_position=start_position, goal_position=goal_position, stage=stage, target_height=0.8)
    attach_obstacles(group, _default_obstacles())

    n_intermediate_admm = 1
    GroupRuntimeConfig(
        n_intermediate_ADMM=n_intermediate_admm,
        t_step=0.1,
        t_window_size=0.2,
        t_end=0.2,
        knot_intervals=5,
        t_resolution_length=6,
        rho=50,
        rho_input=200,
        rho_final_value=5000,
        mpc_version="MPC_param",
        n_of_saved_waypoints=5,
        back_scaling_factor=0.3,
        back_rotation_factor=0.4,
    ).apply(group)

    set_ellipse_initial_final(
        group,
        start_position=start_position,
        goal_position=goal_position,
        a_scale=6,
        b_scale=3,
        ellipse_rotation=math.pi / 2,
    )
    warmup_mpc(group)

    t_iter = time.time()
    n_steps = math.floor(1 / group.vehicles[0].t_step)

    for i in range(n_steps):
        group.set_var({"stage": i})
        for _ in range(n_intermediate_admm):
            group.ACC_MPC_t_param()
            group.solve()
            group.set_simulation(False)

        dt = time.time() - t_iter
        print(str(i) + "th iteration time: " + str(dt) + " seconds")
        t_iter = time.time()
        group.simulation_step()

    return n_steps, group


# Compatibility alias used by older ad-hoc tooling.
run_optimizaiton = run_optimization


def main():
    ensure_output_dirs()

    n_steps, group = run_optimization(
        start_position=[0.0, 0.0, 0.0],
        goal_position=[0.0, 0.0, 0.0],
        stage=0,
    )
    group.plot_moovie_frames(n_steps, iternum=0, seed=0)


if __name__ == "__main__":
    main()
