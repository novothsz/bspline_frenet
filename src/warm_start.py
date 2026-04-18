import numpy as np


def run_acc_mpc_t_param(group):
    """Run ACC waypoint generation in MPC_param mode."""
    t_sweep_start = group.vehicles[0].t_start
    t_sweep_end = group.vehicles[0].t_end

    if group.stage == 0:
        for vehicle in group.vehicles:
            vehicle.variable_history["x_intermediate_list"] = []
            vehicle.variable_history["a_intermediate_list"] = []
            vehicle.variable_history["t_intermediate_list"] = []
            vehicle.variable_history["t_real_intermediate_list"] = []

    for vehicle in group.vehicles:
        vehicle.x_intermediate_list = []
        vehicle.a_intermediate_list = []
        vehicle.t_intermediate_list = []
        vehicle.t_real_intermediate_list = []
        vehicle.t_real_activation_list = []

    max_len_x = group.vehicles[0].n_of_saved_waypoints * 3
    max_len_a = group.vehicles[0].n_of_saved_waypoints * len(group.vehicles[0].obstacles) * 2
    max_len_t = group.vehicles[0].n_of_saved_waypoints

    cum_rotation_old = group.cum_rotation
    cum_scaling_old = group.cum_scaling
    group.sweep_ACC(t_sweep_start=t_sweep_start, t_sweep_end=t_sweep_end)
    group.cum_rotation = cum_rotation_old
    group.cum_scaling = cum_scaling_old

    for vehicle in group.vehicles:
        greater_ = False
        index_ = 0
        change_current = False

        if len(vehicle.variable_history["t_real_intermediate_list"]) > 0:
            for i, t in enumerate(vehicle.variable_history["t_real_activation_list"][-1]):
                greater_new = t[0] <= t_sweep_start + vehicle.t_step
                if greater_ and (not greater_new):
                    index_ = i - 1
                    change_current = True
                    break
                greater_ = greater_new

            if greater_ and greater_new:
                index_ = len(vehicle.variable_history["t_real_activation_list"][-1]) - 1
                change_current = True

        if change_current and index_ >= 0 and len(vehicle.variable_history["t_real_intermediate_list"]) > 0:
            idx = np.arange(3 * index_, 3 * index_ + 3)
            vehicle.current_configuration_position = (
                np.array(vehicle.variable_history["x_intermediate_list"][-1]).reshape(-1)[idx].tolist()[:3]
            )
            group.cum_rotation = np.array(vehicle.variable_history["x_intermediate_list"][-1]).reshape(-1)[idx].tolist()[2]
            group.cum_scaling = cum_scaling_old

        vehicle.variable_history["current_configuration_position"] += [vehicle.current_configuration_position]

    for vehicle in group.vehicles:
        x_intermediate_list = vehicle.x_intermediate_list
        a_intermediate_list = vehicle.a_intermediate_list
        t_intermediate_list = vehicle.t_intermediate_list

        current_len_x = len(x_intermediate_list)
        current_len_a = len(a_intermediate_list)
        current_len_t = len(t_intermediate_list)
        single_len_a = len(group.vehicles[0].obstacles) * 2

        if len(x_intermediate_list) > max_len_x:
            vehicle.x_intermediate_list = x_intermediate_list[: int((current_len_x - max_len_x) / 3)]
            vehicle.a_intermediate_list = a_intermediate_list[: int((current_len_a - max_len_a) / 2)]
            vehicle.t_intermediate_list = vehicle.t_intermediate_list[: (current_len_t - max_len_t)]
            vehicle.t_real_intermediate_list = vehicle.t_real_intermediate_list[: (current_len_t - max_len_t)]

        if len(x_intermediate_list) < max_len_x:
            diff_x = max_len_x - current_len_x
            diff_a = max_len_a - current_len_a
            diff_t = max_len_t - current_len_t
            vehicle.x_intermediate_list = vehicle.x_intermediate_list + vehicle.x_intermediate_list[-3:] * int(diff_x / 3)
            vehicle.a_intermediate_list = vehicle.a_intermediate_list + vehicle.a_intermediate_list[-single_len_a:] * int(
                diff_a / single_len_a
            )
            if max_len_a != len(vehicle.a_intermediate_list):
                print("Baj van fonok!")

            vehicle.t_intermediate_list = vehicle.t_intermediate_list + [vehicle.t_intermediate_list[-1]] * diff_t
            vehicle.t_real_intermediate_list = vehicle.t_real_intermediate_list + [vehicle.t_real_intermediate_list[-1]] * diff_t

        assert max_len_x == len(vehicle.x_intermediate_list)
        assert max_len_t == len(vehicle.t_intermediate_list)

        vehicle.variable_history["x_intermediate_list"][-1] = vehicle.x_intermediate_list
        vehicle.variable_history["a_intermediate_list"][-1] = vehicle.a_intermediate_list
        vehicle.variable_history["t_intermediate_list"][-1] = vehicle.t_intermediate_list
        vehicle.variable_history["t_real_intermediate_list"][-1] = vehicle.t_real_intermediate_list

    return group
