"""
Main function

Ways to improve:
    - hyperparam optimization
"""

from src import *

import time
import math
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
    
    # Create group
    group = Group(n_vehicles=4, start_position = start_position, goal_position = goal_position, stage = stage)
    group.set_group_position(position=group.start_position,targetHeight = targetHeight, position_type='initial')
    group.set_group_position(position=group.goal_position, targetHeight = targetHeight, position_type='final')
   
    obstacles = group.generate_obstacles()
    group.add_obstacles(obstacles)
    group.organise_neighbours()
    
    # Setting some variable for all vehicles
    n_intermediate_ADMM = 1
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
    group.set_var({'MPC_version': 'MPC_param'}) 
    group.set_var({'n_of_saved_waypoints': 5}) 
    
    # Some preparation stuff TODO
    group.ACC_MPC_t_param()  
    group.prepare()
    group.ACC_MPC_t_param()
       
    t_iter = time.time()
    iteration_times = []
    
    
    n_steps = math.floor(1 / group.vehicles[0].t_step)
    for i in range(0, n_steps):
        group.set_var({'stage': i})
        for j in range(n_intermediate_ADMM):
            group.ACC_MPC_t_param()
            group.solve()
            group.set_simulation(False)
        
        iteration_times += [time.time() - t_iter]
        print(str(time.time() - t_iter) + " seconds")
        t_iter = time.time()

        group.simulation_step()
    
    return n_steps, group, iteration_times
    


if __name__ == "__main__":
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