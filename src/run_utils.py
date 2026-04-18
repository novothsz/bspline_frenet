import math
from pathlib import Path


OUTPUT_DIRS = ("log", "urdf", "csv", "figures", "video", "yaml")


def ensure_output_dirs(base_path="."):
    base = Path(base_path)
    for name in OUTPUT_DIRS:
        (base / name).mkdir(parents=True, exist_ok=True)


def create_group(start_position, goal_position, stage, target_height=0.8, n_vehicles=4):
    from .group import Group

    group = Group(
        n_vehicles=n_vehicles,
        start_position=start_position,
        goal_position=goal_position,
        stage=stage,
    )
    group.set_group_position(position=group.start_position, targetHeight=target_height, position_type="initial")
    group.set_group_position(position=group.goal_position, targetHeight=target_height, position_type="final")
    return group


def attach_random_obstacles(group, seed=42):
    obstacles = group.generate_obstacles(seed)
    group.add_obstacles(obstacles)
    group.organise_neighbours()
    return group


def attach_obstacles(group, obstacles):
    group.add_obstacles(obstacles)
    group.organise_neighbours()
    return group


def set_ellipse_initial_final(group, start_position, goal_position, a_scale, b_scale, ellipse_rotation=math.pi / 2):
    n_positions = len(group.vehicles)

    initial_positions = group.ellipse_generator(
        centerpoint=start_position,
        n_positions=n_positions,
        a=group.vehicles[0].radious * a_scale,
        b=group.vehicles[0].radious * b_scale,
        ellipse_rotation=ellipse_rotation,
    )
    for i in range(n_positions):
        group.vehicles[i].set_position(position=initial_positions[i], position_type="initial")

    final_positions = group.ellipse_generator(
        centerpoint=goal_position,
        n_positions=n_positions,
        a=group.vehicles[0].radious * a_scale,
        b=group.vehicles[0].radious * b_scale,
        ellipse_rotation=ellipse_rotation,
    )
    for i in range(n_positions):
        group.vehicles[i].set_position(position=final_positions[i], position_type="final")

    return group


def warmup_mpc(group):
    group.ACC_MPC_t_param()
    group.prepare()
    group.ACC_MPC_t_param()
    return group
