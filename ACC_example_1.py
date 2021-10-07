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





def run_optimizaiton(corners_list, start_position, goal_position, min_iterations, max_iterations, stage):


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
    n_steps = 1
    group.set_var({'n_intermediate_ADMM': n_intermediate_ADMM})
    group.set_var({'t_step': 0})
    group.set_var({'t_window_size': 1})
    group.set_var({'t_end': 1})
    # group.set_var({'knot_intervals': 35})
    group.set_var({'knot_intervals': 25 * 2})
    # group.set_var({'t_resolution_length': 120})
    group.set_var({'t_resolution_length': 50})
    group.set_var({'rho': 500})
    group.set_var({'rho_input': 100})
    group.back_scaling_factor = 0.4
    group.back_rotation_factor = 1 # 0.4
    group.DFM_division = 15
    group.DFM_lookback = 1 / group.DFM_division / 2
    group.DFM_lookahead = 1 / group.DFM_division / 2
    
    
    group.set_var({'MPC_version': False})
    
    
    
    
    # group.DFM_division = 4
    # group.DFM_lookback = 0.1
    # group.DFM_lookahead = 0.1
    # group.back_scaling_factor = 1
    # group.back_rotation_factor = 1
    
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
    
    print("mosoly")
    # group.intermediate_position_generator_PENI_full()
    # group.intermediate_position_generator_SZILARD_sweep()
    
    # group.vehicles[0].obstacles[0].plot_corners_spline()
    # assert 0
    
    group.sweep_ACC()
    
    # return 0, group
    
    # group.intermediate_position_generator_PENI_full()
    # group.intermediate_position_generator_SINGLE_RUN()
    # assert 0
    group.prepare()
    
    import time
    t_iter = time.time()
    iteration_times = []
    
    
    
    for i in range(0, n_steps):
        
        group.set_var({'stage': i})
        for j in range(n_intermediate_ADMM):
            group.solve()
            # group.frenet_plotter(iternum = j, seed = seed)
            group.set_simulation(False)
        
        # group.save_trajectory_to_csv(t_desired = 3, t_hover = 0)
        
        # Time-related things
        iteration_times += [time.time() - t_iter]
        print(str(i) + "th iteration time: " + str(time.time() - t_iter) + " seconds")
        t_iter = time.time()
        
        # group.intermediate_position_generator()
        group.frenet_plotter(iternum = i, seed = seed)
        # group.simulation_step()
            

    # group.plot_moovie_frames(iternum=i, seed=seed)
    
    # writing_parameters_to_file(iteration_times)
    
    return n_steps, group
    
# """    
group_stages = []
corners_list = []
min_iterations = 2
max_iterations = 2


# Stage 0
stage = 0
start_position = [0.0, 0.0, 0.0]
goal_position = [0.0, 0.0, 0.0]
n_steps, group = run_optimizaiton(corners_list, start_position, goal_position, min_iterations, max_iterations, stage)
# group.plot_moovie_frames(n_steps, iternum=0, seed=0)
group.plot_moovie_frames_old(iternum=0, seed=0)

# """
# group.vehicles[0].calculate_formation_error()
# group.calculate_formation_error()
group.save_trajectory_to_csv(n_steps)
"This is not good like this! We need to save the final plots for the various n_intermediate_ADMM values and run the code multiple times"
"Only then can we assemble and compare the results."

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
group.plot_environment(ax, 0)
group.vehicles[0].plot_moovie_frames(ax, 0)
plt.show()


group.vehicles[0].obstacles[0].plot_corners_spline()
