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



def run_optimizaiton(seed):

    start_position = [0.0, 0.0, 0.0]
    goal_position = [0.0, 0.0, 0.0]

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

    """
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
    """
    
            
    # Create group
    group = Group(n_vehicles=4, start_position = start_position, goal_position = goal_position, stage = 0)
    group.set_group_position(
        position=group.start_position,targetHeight = targetHeight, position_type='initial')
    group.set_group_position(
        position=group.goal_position, targetHeight = targetHeight, position_type='final')
    
    # obstacles = group.generate_obstacles()
    # group.add_obstacles(obstacles)
    
    obstacles = group.generate_obstacles(seed)
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
    positions = group.ellipse_generator(centerpoint = start_position, n_positions = len(group.vehicles), a = group.vehicles[0].radious * 9, b = group.vehicles[0].radious * 5,
                                               ellipse_rotation = math.pi / 2)
    for i in range(len(group.vehicles)):
            group.vehicles[i].set_position(position = positions[i], position_type = 'initial')
            
            
    "Changing default rotation for final position"
    positions = group.ellipse_generator(centerpoint = goal_position, n_positions = len(group.vehicles), a = group.vehicles[0].radious * 9 * 0.8, b = group.vehicles[0].radious * 5,
                                               ellipse_rotation = math.pi / 2)
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
    group.plot_moovie_frames(n_steps, iternum=0, seed=0)
    group.plot_frenet_view()
    
    return_dict = {}
    return_dict["n_steps"] = n_steps
    # return_dict["group"] = group
    return_dict["iteration_times"] = iteration_times
    # return n_steps, group, iteration_times
    return return_dict
    
# """    

# return_dict = run_optimizaiton(seed)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(run_optimizaiton, seed) for seed in range(1)]
        res = [f.result() for f in as_completed(futures)]
    



# group.write_iteration_times(prefix = 'single_core_')



"""



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
    
    
    
    
    
    
    
# group.plot_moovie_frames_old(iternum=0, seed=0)


# group.vehicles[0].calculate_formation_error()
# group.calculate_formation_error()
# group.save_trajectory_to_csv(n_steps)

# This is not good like this! We need to save the final plots for the various n_intermediate_ADMM values and run the code multiple times
# Only then can we assemble and compare the results.


kappa_real = group.vehicles[0].variable_history['t_real_intermediate_list']
kappa_pos = group.vehicles[0].variable_history['x_intermediate_list']
kappa_act = group.vehicles[0].variable_history['t_real_activation_list']
kappa_local = group.vehicles[0].variable_history['t_intermediate_list']
kappa_current = group.vehicles[0].variable_history['current_configuration_position']

kappa_real = self.vehicles[0].variable_history['t_real_intermediate_list']
kappa_pos = self.vehicles[0].variable_history['x_intermediate_list']
kappa_act = self.vehicles[0].variable_history['t_real_activation_list']
kappa_local = self.vehicles[0].variable_history['t_intermediate_list']
kappa_current = self.vehicles[0].variable_history['current_configuration_position']


current_configuration_position
"""