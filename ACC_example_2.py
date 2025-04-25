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
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
os.system("mkdir log")
os.system("mkdir urdf")
os.system("mkdir csv")
os.system("mkdir figures")
os.system("mkdir video")
os.system("mkdir yaml")





def run_optimizaiton(corners_list, start_position, goal_position, min_iterations, max_iterations, stage):

    targetHeight = 0.8
    obstacles = []
    
    plt.close('all'); random.seed(64); seed = 64; print("Seed was:", 64)

    # Create group
    group = Group(n_vehicles=4, start_position = start_position, goal_position = goal_position, stage = stage)
    group.set_group_position(
        position=group.start_position,targetHeight = targetHeight, position_type='initial')
    group.set_group_position(
        position=group.goal_position, targetHeight = targetHeight, position_type='final')
    
    # obstacles = group.generate_obstacles()
    # group.add_obstacles(obstacles)
    
    obstacles = group.generate_obstacles(0)
    group.add_obstacles(obstacles)
    group.organise_neighbours()
    
    
    
    n_intermediate_ADMM = 1
    "n_steps = math.floor(1 / group.vehicles[0].t_step)"
    # n_steps = 10
    group.set_var({'n_intermediate_ADMM': n_intermediate_ADMM})
    group.set_var({'t_step': 0.01})
    group.set_var({'t_window_size': 0.12})
    group.set_var({'t_end': 0 + 0.12})
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
    positions = group.ellipse_generator(centerpoint = start_position, n_positions = len(group.vehicles), a = group.vehicles[0].radious * 9 * 1, b = group.vehicles[0].radious * 5 * 1,
                                               ellipse_rotation = math.pi / 2 * 1)
    for i in range(len(group.vehicles)):
            group.vehicles[i].set_position(position = positions[i], position_type = 'initial')
            
            
    "Changing default rotation for final position"
    positions = group.ellipse_generator(centerpoint = goal_position, n_positions = len(group.vehicles), a = group.vehicles[0].radious * 9 * 1, b = group.vehicles[0].radious * 5 * 1,
                                               ellipse_rotation = math.pi / 2 * 1)
    for i in range(len(group.vehicles)):
            group.vehicles[i].set_position(position = positions[i], position_type = 'final')
    
    new_version = True
    
    group.ACC_MPC_t_param()  
    group.prepare()
    group.ACC_MPC_t_param()
       
       
    
    import time
    t_iter = time.time()
    iteration_times = []
    
    # group.intermediate_position_generator()
    n_steps = math.floor(1 / group.vehicles[0].t_step)
    for i in range(0, n_steps):
    # for i in range(19):
        
        group.set_var({'stage': i})
        for j in range(n_intermediate_ADMM):
            
            group.ACC_MPC_t_param()
            group.solve()
            # group.frenet_plotter(iternum = j, seed = seed)
            group.set_simulation(False)
        
        # group.save_trajectory_to_csv(t_desired = 3, t_hover = 0)
        
        # Time-related things
        iteration_times += [time.time() - t_iter]
        # print(str(i) + "th iteration time: " + str(time.time() - t_iter) + " seconds")
        print(str(time.time() - t_iter) + " seconds")
        # print(" ")
        t_iter = time.time()

        group.simulation_step()
            

    # group.plot_moovie_frames(iternum=i, seed=seed)
    
    # writing_parameters_to_file(iteration_times)
    
    return n_steps, group, iteration_times
    
# """    
group_stages = []
corners_list = []
min_iterations = 2
max_iterations = 2


# Stage 0
stage = 0
start_position = [0.0, 0.0, 0.0]
goal_position = [0.0, 0.0, 0.0]
n_steps, group, iteration_times = run_optimizaiton(corners_list, start_position, goal_position, min_iterations, max_iterations, stage)

group.write_iteration_times(prefix = 'single_core_')
group.plot_moovie_frames(n_steps, iternum=0, seed=0)
group.plot_frenet_view()


vehicle_stats = []
for vehicle in group.vehicles:
    vehicle_stats += [vehicle.variable_history["feasibility_dict"]]
    
    
veh = 3
len_ = len(group.vehicles[veh].variable_history['y'])
for veh in range(4):
    print("")
    print("vehicle " + str(veh) + "---------------------")
    a_fes = group.vehicles[veh].variable_history["feasibility_dict"]
    success = [group.vehicles[veh].variable_history["feasibility_dict"][i]["IPOPT_SUCCESS"] for i in range(len_)]
    
    
    status = [group.vehicles[veh].variable_history["feasibility_dict"][i]["IPOPT_RETURN_STATUS"] for i in range(len_)]
    first_time = [group.vehicles[veh].variable_history["first_time_success"][i] for i in range(len_)]
    
    for i, (stat, first) in enumerate(zip(status, first_time)):
        if first == False and stat == "Solve_Succeeded":
            print(str(i) +" - helped")
        elif first == False and stat != "Solve_Succeeded":
            print(str(i) +" - Did not help")

a_fes = group.vehicles[2].variable_history["feasibility_dict"][5]