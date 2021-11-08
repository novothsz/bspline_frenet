"""
Main function

Ways to improve:
    - hyperparam optimization
"""

import time
from src import *
import matplotlib.pyplot as plt
import numpy as np

import random
import sys
import math

import os
os.system("mkdir log")
os.system("mkdir urdf")
os.system("mkdir csv")
os.system("mkdir figures")
os.system("mkdir video")
os.system("mkdir yaml")


from concurrent.futures import ProcessPoolExecutor, as_completed


def writing_parameters_to_file(iteration_times):
    """
    Writing parameteres to file
    """
    cwd = os.getcwd()
    log_file = open(cwd + "/log/" + "log.txt", "w")

    # Writing time required for iteration into file
    log_file.writelines('\n')
    log_file.writelines('Iteration times: \n')
    for i in range(len(iteration_times)):
        log_file.writelines(str(iteration_times[i]) + '\n')

    log_file.writelines('Final time: \n')
    log_file.writelines(str(np.sum(iteration_times)))


targetHeight = 0.8
obstacles = []

plt.close('all')
random.seed(64)
seed = 64
print("Seed was:", 64)


"Obstacles"
obstacles = []
# Obstacle 1
corners = ([0.0, -0.3], [0.5,-0.3], [0.5, 0.3], [0.0, 0.3])
delta = 0.25
corners = ([0.0+delta, -delta], [1+delta,0.6-delta], [1-delta, 0.6+delta], [0.0-delta, delta])
# obstacles += [Obstacle(ID = 0, corners = corners)]
# Obstacle 1 NEW
corners = ([0.0, -0.3], [0.5,-0.3], [0.5, 0.3], [0.0, 0.3])
delta = 0.15
corners = ([0.0+delta, -delta], [1+delta,0.6-delta], [1-delta, 0.6+delta], [0.0-delta, delta])
corners = ([0.15, -0.15], [0.5, 0.053], [0.25, 0.38], [-0.145, 0.145])
corners = ([0.2, -0.25], [0.52, -0.02  ], [0.15, 0.5], [-0.2, 0.23])
corners = ([-0.323, -0.2], [-0.18, -0.4  ], [0.5, 0.0], [0.323, 0.2]) # ACC
corners = ([-0.323, -0.2], [0, -0.6  ], [0.658, -0.174], [0.323, 0.2]) # ACC
delta_x = 0.03
delta_y = -0.1
corners = ([-0.323 + delta_x, -0.2 + delta_y], [0 + delta_x, -0.6  + delta_y ], [0.658 + delta_x, -0.174 + delta_y], [0.323 + delta_x, 0.2 + delta_y]) # ACC


delta_x = 0.03 * -1
delta_y = -0.1 * -1
delta_x = 0.03 * -0
delta_y = -0.1 * -0
corners = ([-0.132, -0.2], [0.182, -0.6  ], [0.495, -0.4], [0.185, 0.0]) # ACC
corners = ([-0.132 + delta_x, -0.2 + delta_y], [0.182 + delta_x, -0.6 + delta_y], [0.495 + delta_x, -0.4 + delta_y], [0.185 + delta_x, 0.0 + delta_y]) # ACC
obstacles += [Obstacle(ID = 0, corners = corners)]

# Obstacle 2
dx = 0.3
dy = 0.4
tmp_obs = ([-2.4674-dx, -1-dy],
         [-2.4674+dx, -1-dy],
         [-2.4674+dx, -6],
         [-2.4674-dx, -6])
obstacles += [Obstacle(ID = 1, corners = tmp_obs)]

# Obstacle 3
tmp_obs = ([-2.4674-dx, -1+dy],
         [-2.4674+dx, -1+dy],
         [-2.4674+dx, 6],
         [-2.4674-dx, 6])
obstacles += [Obstacle(ID = 2,  corners = tmp_obs)]

# Obstacle 4
tmp_obs = ([2.474-dx, 1-dy],
          [2.474+dx, 1-dy],
          [2.474+dx, -6],
          [2.474-dx, -6])
obstacles += [Obstacle(ID = 3, corners = tmp_obs)]
# Obstacle 5
# tmp_obs = ([2.474-dx, 1+dy],
#           [2.474+dx, 1+dy],
#           [2.474+dx, 6],
#           [2.474-dx, 6])
# obstacles += [Obstacle(ID = 4, corners = tmp_obs)]


dx = 0.3
dy = 0.4
tmp_obs = ([2.474-dx, 1-dy],
          [2.474+dx, 1-dy],
          [2.474+dx, 1+dy],
          [2.474-dx, 1+dy])

# obstacles += [Obstacle(ID = 5, corners = tmp_obs)]

# """    
group_stages = []
corners_list = []
min_iterations = 2
max_iterations = 2


# Stage 0
stage = 0
start_position = [0.0, 0.0, 0.0]
goal_position = [0.0, 0.0, 0.0]
        
# Create group
group = Group(n_vehicles=4, start_position = start_position, goal_position = goal_position, stage = stage)
group.set_group_position(
    position=group.start_position,targetHeight = targetHeight, position_type='initial')
group.set_group_position(
    position=group.goal_position, targetHeight = targetHeight, position_type='final')
group.add_obstacles(obstacles)
group.organise_neighbours()



n_intermediate_ADMM = 1
"n_steps = math.floor(1 / group.vehicles[0].t_step)"
# n_steps = 10
group.set_var({'n_intermediate_ADMM': n_intermediate_ADMM})
group.set_var({'t_step': 0.04})
group.set_var({'t_window_size': 0.2})
group.set_var({'t_end': 0 + 0.2})
group.set_var({'knot_intervals': 5})
group.set_var({'t_resolution_length': 10})
group.set_var({'rho': 50})
group.set_var({'rho_input': 200})
group.set_var({'rho_final_value': 5000})
group.back_scaling_factor = 0.3
group.back_rotation_factor = 0.4
group.DFM_lookahead = group.vehicles[0].t_window_size * 0.2
group.DFM_lookback = group.vehicles[0].t_window_size * 0.3

# group.set_var({'MPC_version': True})
group.set_var({'MPC_version': 'MPC_param'}) 
# group.set_var({'n_of_saved_waypoints': int(group.vehicles[0].t_window_size / group.vehicles[0].t_step) + 1}) 
group.set_var({'n_of_saved_waypoints': 5}) 
# print(np.linspace(group.vehicles[0].t_step, 1, group.vehicles[0].n_of_saved_waypoints).tolist())
# print(group.vehicles[0].n_of_saved_waypoints)






"Changing default rotation for initial position"
positions = group.ellipse_generator(centerpoint = start_position, n_positions = len(group.vehicles), a = group.vehicles[0].radious * 9, b = group.vehicles[0].radious * 5,
                                           ellipse_rotation = math.pi / 2)
for i in range(len(group.vehicles)):
        group.vehicles[i].set_position(position = positions[i], position_type = 'initial')
        
        
"Changing default rotation for final position"
positions = group.ellipse_generator(centerpoint = goal_position, n_positions = len(group.vehicles), a = group.vehicles[0].radious * 9, b = group.vehicles[0].radious * 5,
                                           ellipse_rotation = math.pi / 2)
for i in range(len(group.vehicles)):
        group.vehicles[i].set_position(position = positions[i], position_type = 'final')

new_version = True

if new_version == False:
    group.intermediate_position_generator_PENI_MPC()
else:
    group.ACC_MPC_t_param()   
group.prepare()
if new_version == False:
    group.intermediate_position_generator_PENI_MPC()
else:
    # group.ACC_MPC()
    group.ACC_MPC_t_param()
   
# Till here we did single processing...

# target_function = group.vehicles[0].solver.call
# target_function_z = group.vehicles[0].solver_z.call

target_function = group.vehicles[0].distributed_x_update
target_function_z = group.vehicles[0].distributed_z_update

def target_function(list_):
    args, idx = list_
    start_time = time.time()
    res = group.vehicles[idx].solver.call(args)
    final_time = time.time()
    update_time = final_time - start_time
    return {idx: [res, update_time]}

def target_function_z(list_):
    args, idx = list_
    start_time = time.time()
    res = group.vehicles[idx].solver_z.call(args)
    final_time = time.time()
    z_update_time = final_time - start_time
    return {idx: [res, z_update_time]}





if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        n_steps = math.floor(1 / group.vehicles[0].t_step)
        iteration_times = []
        
        for i in range(0, n_steps):
            t_iter = time.time()
            
            
            # vehicles = [vehicle for vehicle in group.vehicles]
            group.set_var({'stage': i})
            
            "x update"
            for i in range(4):
                group.vehicles[i] = group.vehicles[i].x_update_prior()
            args = [vehicle.arg for vehicle in group.vehicles]
                
            futures = [pool.submit(target_function, [arg, j]) for j, arg in enumerate(args)]
            res = [f.result() for f in as_completed(futures)]
            
            # we need to combine the list into a dictionary
            res_dicitonary = {}
            for res_ in res:
                res_dicitonary.update(res_)
            
            for i in range(4):
                group.vehicles[i].solution = res_dicitonary[i][0]
                group.vehicles[i].variable_history["x_update_time"] += [res_dicitonary[i][1]]
                
            for i in range(4):
                group.vehicles[i] = group.vehicles[i].x_update_posterior()
                
            "data exchange x"
            group = group.data_exchange_x()
            
            "z update"
            for i in range(4):
                group.vehicles[i] = group.vehicles[i].z_update_prior()
            args = [vehicle.arg_z for vehicle in group.vehicles]
                
                
            futures = [pool.submit(target_function_z, [arg, j]) for j, arg in enumerate(args)]
            res = [f.result() for f in as_completed(futures)]
            
            # we need to combine the list into a dictionary
            res_dicitonary = {}
            for res_ in res:
                res_dicitonary.update(res_)
            
            for i in range(4):
                group.vehicles[i].solution_z = res_dicitonary[i][0]
                group.vehicles[i].variable_history["z_update_time"] += [res_dicitonary[i][1]]
                
            for i in range(4):
                group.vehicles[i] = group.vehicles[i].z_update_posterior()
                
            "lambda update, data exchange z"
            group = group.lambda_update_data_exchange_z()
            
            # group.solve()
            
            print(str(time.time() - t_iter) + " seconds")
            iteration_times += [time.time() - t_iter]
            t_iter = time.time()
            
            
            group.ACC_MPC_t_param()
            group.simulation_step()
            
    group.write_iteration_times(prefix = 'multi_core_')




























