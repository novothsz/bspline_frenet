import csv
import math


UPDATE_HEADER = [
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


def _horizon_num(group, index):
    return int(index * group.vehicles[0].n_intermediate_ADMM + group.vehicles[0].n_intermediate_ADMM - 1)


def _row_for_metric(group, horizon_num, metric_key):
    values = [vehicle.variable_history[metric_key][horizon_num] for vehicle in group.vehicles]
    worst = max(values)
    best = min(values)
    total = sum(values)
    return values + [worst, best, worst - best, total, worst * 4]


def _write_metric_table(group, prefix, metric_key, filename, n_rows):
    with open(group.cwd + "/log/" + prefix + filename, mode="w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(UPDATE_HEADER)
        for i in range(n_rows):
            horizon_num = _horizon_num(group, i)
            writer.writerow(_row_for_metric(group, horizon_num, metric_key))


def write_iteration_times(group, prefix=""):
    n_steps = math.floor(1 / group.vehicles[0].t_step)
    n_rows = int(n_steps)

    _write_metric_table(group, prefix, "x_update_time", "x_update_times.csv", n_rows)
    _write_metric_table(group, prefix, "z_update_time", "z_update_times.csv", n_rows)

    combined_header = ["worst x", "worst z", "(worst x + worst z)", "(worst x + worst z) * 4"]
    with open(group.cwd + "/log/" + prefix + "combined_update_times.csv", mode="w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(combined_header)
        for i in range(n_rows):
            horizon_num = _horizon_num(group, i)
            worst_x = max(vehicle.variable_history["x_update_time"][horizon_num] for vehicle in group.vehicles)
            worst_z = max(vehicle.variable_history["z_update_time"][horizon_num] for vehicle in group.vehicles)
            writer.writerow([worst_x, worst_z, worst_x + worst_z, (worst_x + worst_z) * 4])

    return group
