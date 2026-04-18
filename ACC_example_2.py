import math
import random
import time

import matplotlib.pyplot as plt

from src import (
    GroupRuntimeConfig,
    attach_random_obstacles,
    create_group,
    ensure_output_dirs,
    set_ellipse_initial_final,
    warmup_mpc,
)


def run_optimization(start_position, goal_position, stage=0):
    plt.close("all")
    seed = 64
    random.seed(seed)
    print("Seed was:", seed)

    group = create_group(start_position=start_position, goal_position=goal_position, stage=stage, target_height=0.8)
    attach_random_obstacles(group, seed=42)

    n_intermediate_admm = 1
    GroupRuntimeConfig(
        n_intermediate_ADMM=n_intermediate_admm,
        t_step=0.01,
        t_window_size=0.2,
        t_end=0.2,
        knot_intervals=5,
        t_resolution_length=10,
        rho=50,
        rho_input=200,
        rho_final_value=5000,
        mpc_version="MPC_param",
        n_of_saved_waypoints=5,
        back_scaling_factor=0.3,
        back_rotation_factor=0.4,
        dfg_mode="legacy",  # Change to "analytic" to use analytic DFG candidate search.
        dfm_lookahead_ratio=0.2,
        dfm_lookback_ratio=0.3,
    ).apply(group)

    set_ellipse_initial_final(
        group,
        start_position=start_position,
        goal_position=goal_position,
        a_scale=9,
        b_scale=5,
        ellipse_rotation=math.pi / 2,
    )
    warmup_mpc(group)

    t_iter = time.time()
    iteration_times = []
    n_steps = math.floor(1 / group.vehicles[0].t_step)

    for i in range(n_steps):
        group.set_var({"stage": i})
        for _ in range(n_intermediate_admm):
            group.ACC_MPC_t_param()
            group.solve()
            group.set_simulation(False)

        dt = time.time() - t_iter
        iteration_times.append(dt)
        print(str(dt) + " seconds")
        t_iter = time.time()
        group.simulation_step()

    return n_steps, group, iteration_times


# Compatibility alias used by older ad-hoc tooling.
run_optimizaiton = run_optimization


def main():
    ensure_output_dirs()

    n_steps, group, _ = run_optimization(
        start_position=[0.0, 0.0, 0.0],
        goal_position=[0.0, 0.0, 0.0],
        stage=0,
    )
    group.write_iteration_times(prefix="single_core_")
    group.plot_moovie_frames(n_steps, iternum=0, seed=0)
    group.plot_frenet_view()


if __name__ == "__main__":
    main()

