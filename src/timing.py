import csv
import math


def write_iteration_times(group, prefix=""):
    first_line = [
        "vehicle 1",
        "vehicle 2",
        "vehicle 3",
        "vehicle 4",
        "worst",
        "best",
        "worst - best",
        "sum",
        "worst * 4",
    ]
    mode = "w"
    n_steps = math.floor(1 / group.vehicles[0].t_step)
    horizon_num_original = int(n_steps)

    with open(group.cwd + "/log/" + prefix + "x_update_times.csv", mode=mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(first_line)
        for i in range(horizon_num_original):
            horizon_num = int(i * group.vehicles[0].n_intermediate_ADMM + group.vehicles[0].n_intermediate_ADMM - 1)
            line = []
            worst = 0
            best = 1e6
            sum_ = 0
            for vehicle in group.vehicles:
                sol_time = vehicle.variable_history["x_update_time"][horizon_num]
                line += [sol_time]
                worst = worst * (worst > sol_time) + sol_time * (sol_time > worst)
                best = best * (best < sol_time) + sol_time * (sol_time < best)
                sum_ += sol_time
            line += [worst, best, worst - best, sum_, worst * 4]
            writer.writerow(line)

    with open(group.cwd + "/log/" + prefix + "z_update_times.csv", mode=mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(first_line)
        for i in range(horizon_num_original):
            horizon_num = int(i * group.vehicles[0].n_intermediate_ADMM + group.vehicles[0].n_intermediate_ADMM - 1)
            line = []
            worst = 0
            best = 1e6
            sum_ = 0
            for vehicle in group.vehicles:
                sol_time = vehicle.variable_history["z_update_time"][horizon_num]
                line += [sol_time]
                worst = worst * (worst > sol_time) + sol_time * (sol_time > worst)
                best = best * (best < sol_time) + sol_time * (sol_time < best)
                sum_ += sol_time
            line += [worst, best, worst - best, sum_, worst * 4]
            writer.writerow(line)

    first_line = ["worst x", "worst z", "(worst x + worst z)", "(worst x + worst z) * 4"]
    with open(group.cwd + "/log/" + prefix + "combined_update_times.csv", mode=mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(first_line)
        for i in range(horizon_num_original):
            horizon_num = int(i * group.vehicles[0].n_intermediate_ADMM + group.vehicles[0].n_intermediate_ADMM - 1)
            line = []
            worst_x = 0
            worst_z = 0
            for vehicle in group.vehicles:
                sol_time = vehicle.variable_history["x_update_time"][horizon_num]
                sol_time_z = vehicle.variable_history["z_update_time"][horizon_num]
                worst_x = worst_x * (worst_x > sol_time) + sol_time * (sol_time > worst_x)
                worst_z = worst_z * (worst_z > sol_time_z) + sol_time_z * (sol_time_z > worst_z)
            line += [worst_x, worst_z, worst_x + worst_z, (worst_x + worst_z) * 4]
            writer.writerow(line)

    return group
