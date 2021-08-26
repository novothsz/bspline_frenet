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
    delta = 0.15
    corners = ([0.0+delta, -delta], [1+delta,0.6-delta], [1-delta, 0.6+delta], [0.0-delta, delta])
    obstacles += [Obstacle(ID = 0, corners = corners)]
    
    # Obstacle 2
    dx = 0.25
    dy = 0.3
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
    
            
    # Create group
    group = Group(n_vehicles=4, start_position = start_position, goal_position = goal_position, stage = stage)
    group.set_group_position(
        position=group.start_position,targetHeight = targetHeight, position_type='initial')
    group.set_group_position(
        position=group.goal_position, targetHeight = targetHeight, position_type='final')
    group.add_obstacles(obstacles)
    group.organise_neighbours()
    
    group.prepare()
    
    import time
    t_iter = time.time()
    iteration_times = []
    
    # Optimization
    for i in range(3):
        group.solve()
        
        # Time-related things
        iteration_times += [time.time() - t_iter]
        print(str(i) + "th iteration time: " + str(time.time() - t_iter) + " seconds")
        t_iter = time.time()
        
        group.frenet_plotter(iternum = i, seed = seed)
        
    # Simulation steps
    # for i in range(3, 15):
    #     group.simulation_step()
        
    #     group.solve()
        
    #     # Time-related things
    #     iteration_times += [time.time() - t_iter]
    #     print(str(i) + "th iteration time: " + str(time.time() - t_iter) + " seconds")
    #     t_iter = time.time()
        
    #     group.frenet_plotter(iternum = i, seed = seed)
        

    # group.plot_moovie_frames(iternum=i, seed=seed)
    
    # writing_parameters_to_file(iteration_times)
    
    return group
    
    
group_stages = []
corners_list = []
min_iterations = 2
max_iterations = 2


# Stage 0
stage = 0
start_position = [0.0, 0.0]
goal_position = [0.0, 0.0]
group = run_optimizaiton(corners_list, start_position, goal_position, min_iterations, max_iterations, stage)
