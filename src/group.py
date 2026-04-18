from .vehicle import Vehicle
import numpy as np
import math
import matplotlib.pyplot as plt
from .environment import Environment
from numpy import interp
import time
import random
import copy

try:
    import yaml as _yaml
    yaml = _yaml if hasattr(_yaml, "dump") else None
except Exception:
    yaml = None

from .obstacle import Obstacle
from .admm import data_exchange_x as run_data_exchange_x
from .admm import lambda_update_data_exchange_z as run_lambda_update_data_exchange_z
from .admm import solve as run_solve
from .group_lifecycle import initialize_values as run_initialize_values
from .group_lifecycle import prepare as run_prepare
from .group_lifecycle import set_simulation as run_set_simulation
from .group_lifecycle import set_var as run_set_var
from .group_lifecycle import simulation_step as run_simulation_step
from .timing import write_iteration_times as run_write_iteration_times
from .warm_start import run_acc_mpc_t_param
from .visualization import plot_frenet_view as render_plot_frenet_view
from .visualization import plot_moovie_frames as render_plot_moovie_frames

class Group(Environment):
    def __init__(self, n_vehicles : int, start_position = [-0.8, 0, 0], goal_position = [0.8, 0, 0], stage = 0):
        self.vehicles = []
        for i in range(n_vehicles):
            vehicle = Vehicle()
            vehicle.ID = i
            vehicle.stage = stage
            self.vehicles += [vehicle]
        self.start_position = start_position
        self.goal_position = goal_position
        self.stage = stage
        super().__init__()

        # Create figures for plotting
        self.figures = {}
        fig, ax = plt.subplots()
        self.figures["figures"] = [fig, ax]
        fig, ax = plt.subplots()
        self.figures["videos"] = [fig, ax]
        
        self.rotation_angle = 0
        self.scaling_factor = 1
        self.og_final_positions = []
        self.back_scaling_factor = 0.2 * 0
        self.back_rotation_factor = 0.2 * 0
        
        self.DFM_division = 12
        self.DFM_lookback = 1 / self.DFM_division / 2
        self.DFM_lookahead = 1 / self.DFM_division / 2
        self.MPC_version = []
        
        
        self.cum_rotation = 0
        self.cum_scaling = 1
        
        
        self.ACC_MPC_t_queue = []
        self.ACC_MPC_pos_queue = []
        
        
        
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    import functools
    @functools.lru_cache(maxsize=None)
    def get_obstacle_corners(self, t):
        """ This function returns the corners for all the obstacles in a list.
        (To loop through: 
        """
        corners = []
        for obstacle in self.vehicles[0].obstacles:
            corners_tmp = [ [corner[0](t).tolist()[0][0], corner[1](t).tolist()[0][0]] for corner in obstacle.corners_spline]
            corners += [corners_tmp]
        return corners
    
    @functools.lru_cache(maxsize=None)
    def get_scaled_obstacle_corners(self, t):
        """ This function returns the corners for all the obstacles in a list.
        (To loop through: 
        """
        corners = []
        for obstacle in self.vehicles[0].obstacles:
            corners_tmp = [ [corner[0](t).tolist()[0][0], corner[1](t).tolist()[0][0]] for corner in obstacle.scaled_corners_spline]
            corners += [corners_tmp]
        return corners
    
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    def ACC_MPC_t_param(self):
        return run_acc_mpc_t_param(self)


    # Okay, this will be the version, where we look, whether the obstacle has entered the danger zone.
    # How large is the danger zone?
    # Well, the size of the maximum formation size.
    def sweep_ACC(self, t_sweep_start = 0, t_sweep_end = 1):
        """This is the function, described as Dynamic Formaiton Generator(DFG), that was described in the ACC paper.
        It works as follows:
            sweeps the time interval t \in [t_sweep_start, t_sweep_in]
            searches for collision
            if collision found, h... let's rather implemment it in a new function....
        """
        
        t_iter = time.time()
        
        # The timestep of the DFG algorithm.
        t_step = 0.01
        
        # The initial configuration with which DFG calculates.
        # This value is sometimes being changed in "ACC_MPC_t_param", that is in an MPC implementation.
        vehicle_positions = []
        for vehicle in self.vehicles:
            # vehicle_positions += [vehicle.x0[:2]]
            vehicle_positions += [vehicle.current_configuration_position[:2]]
        vehicle_positions_original = np.array(vehicle_positions).tolist()
                
        
        # The DFG iteration!s
        t_end = t_sweep_start # TODO: jajj, ne hívjuk már t_end-nek...
        while t_end <= t_sweep_end + self.TOL:
            # Step 1: Check if inside danger zone at time t
            # all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions, self.get_obstacle_corners(t_end))
            all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions, self.get_scaled_obstacle_corners(t_end))
            
            # Let's check, with which obstacle we have collision.
            obstacle_idx = []
            for i, obstacle in enumerate(self.vehicles[0].obstacles):
                idx = np.arange(len(self.vehicles)*i,len(self.vehicles)*i+len(self.vehicles)) 
                idx = np.arange(4*i,4*i+4) # bacuse calculated with danger zone, where values are obst0->[corner 0, c1, c2, c3]; o1->[c0, c1, c2, c3]; ...
                # is_collision = any(np.array(all_collisions)[idx].tolist())
                is_collision = all_collisions[i] # no more corners indexing :)
                if is_collision:
                    # It's okay, that we check if the obstacle enters into the danger zone, but 
                    # when looking for the obstacle ID, we should find the one, with which we will collide!!
                    # Meaning: use the collision function and not the danger zone function.
                    # (If none, then we can keep the formation :) )
                    # all_collisions2, collision2 = self.check_collision_with_obstacles(vehicle_positions, [self.get_obstacle_corners(t_end)[i]])
                    # all_collisions2, collision2 = self.check_collision_with_obstacles(vehicle_positions, np.array(self.get_obstacle_corners(t_end))[ [i] ].tolist()) 
                    
                    
                    # --> we need to use this, because danger zone is kind of arbitrary
                    # we are using the non-scaled positions for a reason...
                    # And the reason being
                    all_collisions2, collision2 = self.check_collision_with_obstacles(vehicle_positions, np.array(self.get_obstacle_corners(t_end))[ [i] ].tolist()) 
                    # nah... maybe it is better to use the same, because let's just use the same. Because they might give different results.
                    # all_collisions2, collision2 = self.check_danger_zone_with_obstacles(vehicle_positions, np.array(self.get_scaled_obstacle_corners(t_end))[ [i] ].tolist()) 
                    is_collision2 = collision2
                    if is_collision2:
                        # If we collide with an obstacle, we save its index
                        obstacle_idx += [i]
                        
            # if (t_end >= 1 - self.TOL and t_end <= 1 + self.TOL) == True:
                # obstacle_idx = [0]
                # TODO: what was the above code doing??
                # print("Setting original configuration")
            # if t_end >= t_sweep_end - t_step:
            #     kappa = True
            #     all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions, self.get_scaled_obstacle_corners(t_end))
            # Step 2: If collision has been found, find t_danger_end time.
            if obstacle_idx != []:
                
                t_danger_start = t_end
                t_danger_end = t_danger_start
                t_danger_step = t_step / 10 # if collision is detected, we step this 'smoothly' until no danger is detected
                while t_danger_end <= 1:
                    t_danger_end += t_danger_step
                    # Check collision ---!!!but only with obstacles, which are dangerous!!!--- (we expect, that this doesn't change. Meaning:
                    # obstacles are spaced out well.)
                    # all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions, np.array(self.get_obstacle_corners(t_danger_end))[obstacle_idx].tolist())
                    all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions, np.array(self.get_scaled_obstacle_corners(t_danger_end))[obstacle_idx].tolist())
                    if collision == False:
                        # this is the t_danger_end we were looking for
                        
                        # This line isn't even doing anything...
                        self.check_danger_zone_with_obstacles(vehicle_positions, np.array(self.get_scaled_obstacle_corners(t_danger_end))[obstacle_idx].tolist())
                        break
                    
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111)
                    # for pos in np.array(self.get_scaled_obstacle_corners(t_danger_end))[obstacle_idx].tolist()[0]:
                    #     plt.plot(pos[0], pos[1], 'ro')
                        
                    # s_danger = 0.6988905493709299    
                    # circle = plt.Circle((0, 0), s_danger, color='r', alpha=0.5, zorder = 0)
                    # ax.add_patch(circle)
                    # ax.set_aspect('equal', adjustable='box')
                    # plt.show()
                    
                # Step 3: Find the right formation configuration for t \in [t_danger_start, t_danger_end]
                # Form t_zizz
                t = (t_danger_start + t_danger_end) / 2
                lookback = (t_danger_end - t_danger_start) / 2 * 1.0
                lookahead = (t_danger_end - t_danger_start) / 2 * 1.0
                t_zizz = np.linspace( (t-lookback >= 0) * (t-lookback) + (t-lookback > 0) * 0,
                                          (t+lookahead <= 1) * (t+lookahead) + (t+lookahead > 1) * 1,
                                          10)
                
                # Get the least cost formation & ACTION_TAKEN
                cum_rotation = self.cum_rotation
                cum_scaling = self.cum_scaling
                
                # Find optimal formation configuration
                vehicle_positions, cum_rotation, cum_scaling, ACTION_TAKEN = self.intermediate_position_generator_SZILARD(vehicle_positions, cum_rotation, cum_scaling, t_zizz)
            
                
                self.cum_rotation = cum_rotation
                self.cum_scaling = cum_scaling
                
                
                # plt.figure()
                # for pos in self.get_obstacle_corners(t_end)[1]:
                #     plt.plot(pos[0], pos[1], 'ro')
                
                # for veh in vehicle_positions:
                #     plt.plot(veh[0], veh[1], 'ko')
                # plt.show()
                
                
                # Step 4: Handle actions taken (back_transformation, yes, no_action, no_solution_found)
                # Create a t_intermediate & x_intermediate from this
                if ACTION_TAKEN == "back_transformation" or ACTION_TAKEN == "yes":
                    
                    if self.MPC_version == 'MPC_param':
                        # We need to account for the fact, that t_global_horizon != t_local_horizon
                        t_local = interp(t,[t_sweep_start,t_sweep_end],[0,1])
                        for i, vehicle in enumerate(self.vehicles):
                            vehicle.x_intermediate_list += [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]
                            # vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]]
                            vehicle.t_intermediate_list += [t_local]
                            vehicle.t_real_intermediate_list += [t]
                            vehicle.t_real_activation_list += [[t_danger_start, t_danger_end]]
                            # vehicle.variable_history['t_intermediate_list'] += [t_local]
                            
                            # Okay... So if ACTION_TAKE == "yes", that means, that an obstacle has triggered DFG
                            # and an intermediate way-point was created.
                            # We would want to associate a hyperplane direction to this time and obstacle
                            
                            if ACTION_TAKEN == "back_transformation":
                                pass
                            
                            if ACTION_TAKEN == "yes":
                                # assume, that only a single obstacle is causing trouble... (for simplicity)
                                obstacle_idx[0]
                                obst_ID = self.vehicles[0].obstacles[obstacle_idx[0]].ID
                                
                                
                                # - outside or inside the formation?
                                obstacle_corners = np.array(self.get_obstacle_corners(t))[obstacle_idx[0]].tolist()
                                point = [vehicle_positions[i][0], vehicle_positions[i][1]]
                                inside = self.check_point_inside(point, obstacle_corners)
                                
                                a_hyp_direction = point[1] - obstacle_corners[0][1]
                                
                                if a_hyp_direction > 0:
                                    a = [0, 1]
                                else:
                                    a = [0, -1]
                                a = [0, 0]
                                
                                
                                """
                                # Okay, so the normal vector of the hyperplane points from the obstacle to 
                                # the vehicle.
                                # If the obstacle is inside the formation, that means, that the normal vector
                                # should point "away" from the from the Frenet center
                                # Okay, and how do we decide, in which direction? "upward" or "downward"?
                                # Well, if x_intermediate[0] > 0, then upward, otherwise downward.
                                if inside:
                                    if point[1] > 0:
                                        a = [0, 1]
                                    elif point[1] < 0:
                                        a = [0,-1]
                                    else:
                                        raise Exception("Azt hogy? :D")
                                # If the obstacle is outside the formation, then a_n should point "inwards", in the 
                                # direction of the Frenet center
                                else:
                                    if point[1] > 0:
                                        a = [0, -1]
                                        # a = [1, 0]
                                    elif point[1] < 0:
                                        a = [0, 1]
                                        # a = [-1,0]
                                    else:
                                        raise Exception("Azt hogy? :D")
                                """
                                        
                                # If this is a gate, then it has a pair. Let's get its ID as well
                                gate_pair_exists = False
                                gate_pair_ID = []
                                for obst in self.vehicles[0].obstacles:
                                    if obst.ID == obst_ID:
                                        if obst.gate_pair_ID != []:
                                            gate_pair_exists = True
                                            gate_pair_ID = obst.gate_pair_ID
                                        else:
                                            gate_pair_exists = False
                                            gate_pair_ID = obst.gate_pair_ID
                                            
                                
                                        
                                # Okay. So for this specific obstacle, we add a, and [0, 0] for the others.
                                for obst in self.vehicles[0].obstacles:
                                    if obst.ID == obst_ID:
                                        vehicle.a_intermediate_list += a
                                        vehicle.a_intermediate_ID_list += [ int(obst_ID * (obst.ID == obst_ID)) + int(obst_ID * (obst.gate_pair_ID == obst_ID))]
                                    elif (gate_pair_exists and gate_pair_ID == obst.ID):
                                        vehicle.a_intermediate_list += [-a[0], -a[1]]
                                        vehicle.a_intermediate_ID_list += [ int(obst_ID * (obst.ID == obst_ID)) + int(obst_ID * (obst.gate_pair_ID == obst_ID))]
                                    else:
                                        vehicle.a_intermediate_list += [0, 0]
                                        vehicle.a_intermediate_ID_list += ["[0, 0]"]
                                        
                                        
                                        
                                
                                    
                                    
                                
                        
                    elif self.MPC_version == False or self.MPC_version == True:
                        for i, vehicle in enumerate(self.vehicles):
                            vehicle.x_intermediate_list += [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]
                            # vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]]
                            vehicle.t_intermediate_list += [t]
                            # vehicle.variable_history['t_intermediate_list'] += [t]
                        
                        
                elif ACTION_TAKEN == 'no_action':
                    pass
                elif ACTION_TAKEN == 'no_solution_found':
                    print("No solution has ben found. We need to halve the time. This should be implemented later :)")
                    assert 0
                
                        
                
                t_end = t_danger_end + t_step
                
        
            elif obstacle_idx == []:
                t_end += t_step
                
                
        # print("ACC_sweep finished")
        # print('t_intermediate_list')
        # print(self.vehicles[0].t_intermediate_list)
        # print('x_intermediate_list')
        # print(self.vehicles[0].x_intermediate_list)
        
        tmp_len = len(self.vehicles[0].t_intermediate_list)

                
                
        if tmp_len == 0:
            # print("We haven't generated anything, therefore what we started off with, is okay. Use that.")
            # then we can set the original configuraiton back
            for i, vehicle in enumerate(self.vehicles):
                vehicle.x_intermediate_list += [vehicle_positions_original[i][0], vehicle_positions_original[i][1], self.cum_rotation]
                vehicle.a_intermediate_list += [0, 0] * len(vehicle.obstacles)
                # vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions_original[i][0], vehicle_positions_original[i][1], 0]]
                t_local = interp(t_sweep_end,[t_sweep_start,t_sweep_end],[0,1])
                vehicle.t_intermediate_list += [t_local]
                vehicle.t_real_intermediate_list += [t_sweep_end]
                # vehicle.variable_history['t_intermediate_list'] += [1]
                
        
        # Saving stuff to the history
        for i, vehicle in enumerate(self.vehicles):
            vehicle.variable_history['x_intermediate_list'] += [vehicle.x_intermediate_list]
            vehicle.variable_history['a_intermediate_list'] += [vehicle.a_intermediate_list]
            vehicle.variable_history['t_intermediate_list'] += [vehicle.t_intermediate_list]
            vehicle.variable_history['t_real_intermediate_list'] += [vehicle.t_real_intermediate_list]
            vehicle.variable_history['t_real_activation_list'] += [vehicle.t_real_activation_list]
            
        # print('ACC_sweep full time: ' + str(time.time() - t_iter))
        
        return self
        

    
        
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    
    
    """
    Okay... so...
    Assume, we are doing MPC.
    Then, in the first run, we will have to create t_intermediate steps. Let it be self.n_of_saved_waypoints = int(self.t_window_size / self.t_step) assert 
    # We already have this, actually...
        
        self.n_of_saved_waypoints = int(self.t_window_size / self.t_step) + 1; epsilon = 10e-10; assert self.t_window_size % self.t_step > -epsilon and \
                                                                                                        self.t_window_size % self.t_step < epsilon# math.floor(self.t_window_size / self.t_step)
                                                                                                        
    s = np.sqrt(definite_integral(self.fp.fx_spline.derivative()**2 + self.fp.fy_spline.derivative()**2,0, 1))   
    !!!!! VIGYÁZZ, ITT MÉG HOZZA KELL ADNI AZT, HOGY HALAD ALATTUNK AZ ÚT, TEHÁT AZ, AKI EGY IRÁNYBA 
    AKAR MENNI A FRENET SEBESSÉGÉVEL, ANNAK MÉG GYORSABBAN KELL MENNIE!!!!!
    
    
    T = 10
    v_frenet = s / T
    
    s_max = self.vehicles[0].radious * 5 * 2
    t_DFG = s_max / v_frenet
    
    
    
    
    """
    
    def intermediate_position_generator_SZILARD_action(self, t_end):
        
        "Some part are copied over from intermediate_position_generator_PENI_MPC"
        # namely: setting x0
        
        """ This function is used, when generating new waypoints. 
        """
        # Step 1: Set x0 as starting position
        if self.vehicles[0].t_intermediate_list == []:
            vehicle_positions = []
            for vehicle in self.vehicles:
                vehicle_positions += [vehicle.x0[:2]]
        else:
            vehicle_positions = []
            for vehicle in self.vehicles:
                final_waypoints = vehicle.x_intermediate_list[-3:-1]
                vehicle_positions += [final_waypoints]
                
            
        # Step 2: get the current final time, rotation and scaling
        # t_end = self.vehicles[0].t_end + self.vehicles[0].t_step #... well, maybe  - self.vehicles[0].t_step? Depends on when we call this method
        
        t_end = t_end # :))
        
        cum_rotation = self.cum_rotation
        cum_scaling = self.cum_scaling
                
        # Step 3: zizzentsük be 
        # TODO: ?Mennyi legyen a zizz?
        t = t_end
        # lookback = self.vehicles[0].t_window_size * 0.2
        # lookahead = self.vehicles[0].t_window_size * 0.2
        lookback = self.DFM_lookback
        lookahead = self.DFM_lookahead
        t_zizz = np.linspace( (t-lookback >= 0) * (t-lookback) + (t-lookback > 0) * 0,
                                  (t+lookahead <= 1) * (t+lookahead) + (t+lookahead > 1) * 1,
                                  10)
        # Step 4: Get the least cost formation & ACTION_TAKEN
        vehicle_positions, cum_rotation, cum_scaling, ACTION_TAKEN = self.intermediate_position_generator_SZILARD(vehicle_positions, cum_rotation, cum_scaling, t_zizz)
        
        # Step 5: Save stuff
        self.cum_rotation = cum_rotation
        self.cum_scaling = cum_scaling
        
        # Step 6: update x_intermediate_list & t_intermediate_list
        # Well... we have the following ACTION_TAKEN options:
        # - no_action -> do nothing
        # - back_transformation -> save it
        # - no_solution_found -> do nothing
        # - yes -> save it
        
        if ACTION_TAKEN == "back_transformation" or ACTION_TAKEN == "yes":
        
            
            # for vehicle_positions, cum_rotation, t_ in zip(vehicle_positions_saved, cum_rotation_saved, t_waypoints):
            for i, vehicle in enumerate(self.vehicles):
                vehicle.x_intermediate_list += [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]
                vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]]
                vehicle.t_intermediate_list += [t]
                
            # "MPC style"
            # for i, vehicle in enumerate(self.vehicles):
            #     # delete first elemnt, attach new element to the end
            #     vehicle.x_intermediate_list = vehicle.x_intermediate_list[3:] + [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation] 
            #     vehicle.variable_history['x_intermediate_list'] += [vehicle.x_intermediate_list]
            #     vehicle.xf = vehicle.x_intermediate_list[-3:] + vehicle.xf[3:]
                
            # vehicle_positions_new = []
            # for i, vehicle in enumerate(self.vehicles):
            #     vehicle_positions_new += [[vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]]
            # self.set_var({'new_positions': {'stage' : self.stage, 'vehicle_positions_new' : vehicle_positions_new}})
            
        if ACTION_TAKEN == 'no_action' or ACTION_TAKEN == 'no_solution_found':
            
            pass
        
        print('t_end, ACTION_TAKEN: ' + str(t_end) + ', ' + str(ACTION_TAKEN))
        
        return self
        
    def intermediate_position_generator_SZILARD(self, vehicle_positions, cum_rotation, cum_scaling, t):
        """ This function is called iteratively every ?t_step? -> What should this value be?. It check for collision. 
        If no collision:
            - rotate/scale back -> save it to t&x_intermediate_list
            - do nothing
        If collision:
            - save it to t&x_intermediate_list
        """        
        
        "The best way to start is to copy everything from intermediate_position_generator_PENI"
        # we will have an indicator on what action we have taken
        ACTION_TAKEN = []
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
        """The goal of this function is to receive a set of vehicle positions and calculate a rotated-scaled frame, that does not collide with 
        obstaacles at the given time-point.
        If t is a list of time values, then each of these time values will be checked for collision"""
        if type(t) != list and type(t) != type(np.array([])):
            if t >= 1 - self.TOL and t <= 1 + self.TOL:
                self.back_rotation_factor = 1
                self.back_scaling_factor = 1
        elif type(t) == list or type(t) == type(np.array([])):
            if any([(t_ >= 1 - self.TOL and t_ <= 1 + self.TOL) for t_ in t]):
                self.back_rotation_factor = 1
                self.back_scaling_factor = 1
                
        if self.stage == 23:
            kappa = True
            # print(t)
        # Step 0: first always try to turn&scale it back... :)
        #Backturning
        rotation_angle = -1 * self.back_rotation_factor * cum_rotation
        # --minimum rotation value--
        if -5/360 * 2 * math.pi <= rotation_angle < 0.0 or 0.0 < rotation_angle <= -5/360 * 2 * math.pi:
            rotation_angle = -1 * cum_rotation
        
        
        # Backscaling
        deviance = abs(1 - 1 / cum_scaling)
        deviance *= self.back_scaling_factor
        
        # if the formation is larger, than the reference size, then we want to shrink it
        if cum_scaling >= 1:
            scaling_factor = 1 - deviance
        # if the formation is smaller, than the reference size, then we want to expand it
        else:
            scaling_factor = 1 + deviance
            
        # --minimum scaling value--
        # if the back-scaling factor changes the size less then 10%, then we scale back completely
        if 0.9 <= scaling_factor < 1.0 or 1.0 < scaling_factor <= 1.1:
            deviance = abs(1 - 1 / cum_scaling)
            if cum_scaling >= 1:
                scaling_factor = 1 - deviance
            else:
                scaling_factor = 1 + deviance
        
        vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
        vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
        collision_saved = []
        for t_ in t:
            all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, self.get_scaled_obstacle_corners(t_))
            # all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions_scaled_rotated, self.get_scaled_obstacle_corners(t_))
           
            collision_saved += [False]
            if collision == True:
                collision_saved[-1] = True
                break
        
        if all(collision is False for collision in collision_saved):
            # If we are in this if, then no collision happens when we try to rotate/scale back the formation.
            # print('-, -')
            if rotation_angle == 0 and scaling_factor == 1:
                ACTION_TAKEN = "no_action"
            else:
                ACTION_TAKEN = "back_transformation"
            return vehicle_positions_scaled_rotated, cum_rotation + rotation_angle, cum_scaling * scaling_factor, ACTION_TAKEN
        
        # Otherwise, if we cannot rotate&scale back, find something else:
        # Step 1: generate possible rotation angles & scaling factors
        degree_step = 5
        radian_step = degree_step/360 * 2 * math.pi
        rotation_angles = [[0 + radian_step * i, 0 - radian_step * i] for i in range(1, int( (math.pi/2) / radian_step))]
        rotation_angles = np.array(rotation_angles).reshape(-1).tolist()
        
        scaling_step = 1.5
        scaling_factors_shrink = [  1 / (scaling_step ** i)  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
        scaling_factors_expand = [  1 * scaling_step ** i  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
        scaling_factors = scaling_factors_shrink + scaling_factors_expand
        scaling_factors = [1, 0.9, 0.8, 0.6, 1.1, 1.2, 1.4, 1.8, 2.0, 2.5, 3.0, 4.0]
        
        # Step 2: We iterate through all possible rotation & scaling possibilities
        # Then, in b) we check, if that specific rotation & sacling results in collision between [t0, tf] or not.
        costs = []
        vehicle_positions_new_saved = []
        rotation_angle_new_saved = []
        scaling_factor_new_saved = []
        collision_saved = []
        for scaling_factor in scaling_factors:
            vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
            for rotation_angle in rotation_angles:
                
                vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
                # Step 2b) check for each t_ in t if collision happens. If yes, do not check further, 
                # the given scaling factor & rotation angle is not good.
                
                # First let's save these :)
                vehicle_positions_new_saved += [vehicle_positions_scaled_rotated]
                rotation_angle_new_saved += [rotation_angle]
                scaling_factor_new_saved += [scaling_factor]
                collision_saved += [False]
                
                collision_saved_tmp = []
                for t_ in t:
                    all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, self.get_obstacle_corners(t_))
                    collision_saved_tmp += [collision]
                    if collision == True:
                        collision_saved[-1] = True
                        costs += [math.inf]
                        break
                # if for any t_ we did not break and put inf cost into costs, then calculate a proper cost   
                if all(collision is False for collision in collision_saved_tmp):
                    # Step 2c) if no collision happens, calculate a cost for the given rotation & scaling combo
                    cost = self.formation_change_cost_calculator(vehicle_positions, vehicle_positions_scaled_rotated, rotation_angle, scaling_factor)
                    # print(scaling_factor, rotation_angle, cost)
                    costs += [cost]
                    
        max_distance = self.max_distance_during_turning(vehicle_positions, vehicle_positions_new_saved)
        # if we cannot find good solution, do nothing
        if all(collision_saved):
            vehicle_positions_new = vehicle_positions
            cum_rotation = 0
            cum_scaling = 1
            print("We have a problem boss! Every formation candidate collides :/")
            ACTION_TAKEN = "no_solution_found"
        else: 
            # Now we have the costs and everything in order.
            # Let's find the least cost value.
            cost_min = min(costs)
            cost_min_idx = costs.index(cost_min)
            # And the ideal position, rotation angle & scaling factor is:
            vehicle_positions_new = vehicle_positions_new_saved[cost_min_idx]
            rotation_angle_new = rotation_angle_new_saved[cost_min_idx]
            scaling_factor_new = scaling_factor_new_saved[cost_min_idx]
            
            cum_rotation += rotation_angle_new
            cum_scaling *= scaling_factor_new
            
            ACTION_TAKEN = "yes"
            
            # print(rotation_angle_new, scaling_factor_new)
        
        
        return vehicle_positions_new, cum_rotation, cum_scaling, ACTION_TAKEN

        
       
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ########################################################################### 
        
    def max_distance_during_turning(self, vehicle_positions_original , vehicle_positions_new_saved):
        max_distance = -math.inf
        # for each formation candidate
        for vehicle_positions in vehicle_positions_new_saved:
            # for each vehicle
            for pos_new, pos_original in zip(vehicle_positions, vehicle_positions_original):
                distance = np.sqrt((pos_new[0] - pos_original[0])**2 + (pos_new[1] - pos_original[1])**2)
                if distance > max_distance:
                    max_distance = distance
                
        return max_distance
                        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    def intermediate_position_generator_PENI_MPC(self):  
        """ This function is used, when generating new waypoints. 
        """
        # Step 1: Set x0 as starting position
        if self.vehicles[0].t_intermediate_list == []:
            vehicle_positions = []
            for vehicle in self.vehicles:
                vehicle_positions += [vehicle.x0[:2]]
        else:
            vehicle_positions = []
            for vehicle in self.vehicles:
                final_waypoints = vehicle.x_intermediate_list[-3:-1]
                vehicle_positions += [final_waypoints]
            
        # Step X: if the list "t_intermediate_list" hasn't reached its full length, then we are at the beginning, and need to fill it up.
        # This means we have to call intermediate_position_generator_PENI_full for the t_window_size
        
        if self.vehicles[0].t_intermediate_list == []:
            
            epsilon = 10e-10; assert ( self.vehicles[0].t_window_size / (self.vehicles[0].n_of_saved_waypoints - 1)) % self.vehicles[0].t_step > -epsilon and \
                                     ( self.vehicles[0].t_window_size / (self.vehicles[0].n_of_saved_waypoints - 1)) % self.vehicles[0].t_step < epsilon
            DFM_values = [np.linspace(0, self.vehicles[0].t_window_size, self.vehicles[0].n_of_saved_waypoints),
                          self.vehicles[0].t_window_size * 0.1, # lookback
                          self.vehicles[0].t_window_size * 0.1] # lookahead
            self.intermediate_position_generator_PENI_full(DFM_values = DFM_values)
            # In this case, we are done :)
            # No, we are not done. The vehicles have their local time... ;)
            for vehicle in self.vehicles:
                vehicle.t_intermediate_list = np.linspace(self.vehicles[0].t_step, 1, self.vehicles[0].n_of_saved_waypoints).tolist()
                # print(vehicle.x_intermediate_list)
                
                vehicle.variable_history['x_intermediate_list'] += [vehicle.x_intermediate_list] 
                
                # print(vehicle.variable_history['x_intermediate_list'][-1])
                # assert 0
            
            return self
            
        # Step 2: get the current final time, rotation and scaling
        t_end = self.vehicles[0].t_end + self.vehicles[0].t_step #... well, maybe  - self.vehicles[0].t_step? Depends on when we call this method
        
        cum_rotation = self.cum_rotation
        cum_scaling = self.cum_scaling
        
        # Step 3: zizzentsük be
        t = t_end
        lookback = self.vehicles[0].t_window_size * 0.2
        lookahead = self.vehicles[0].t_window_size * 0.2
        lookback = self.DFM_lookback
        lookahead = self.DFM_lookahead
        t_zizz = np.linspace( (t-lookback >= 0) * (t-lookback) + (t-lookback > 0) * 0,
                                  (t+lookahead <= 1) * (t+lookahead) + (t+lookahead > 1) * 1,
                                  10)
        # Step 4: Get the least cost formation
        vehicle_positions, cum_rotation, cum_scaling = self.intermediate_position_generator_PENI(vehicle_positions, cum_rotation, cum_scaling, t_zizz)
        
        # Step 5: Save stuff
        self.cum_rotation = cum_rotation
        self.cum_scaling = cum_scaling
        
        # Step 6: update x_intermediate_list & t_intermediate_list
        
        for i, vehicle in enumerate(self.vehicles):
            # delete first elemnt, attach new element to the end
            vehicle.x_intermediate_list = vehicle.x_intermediate_list[3:] + [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation] 
            vehicle.variable_history['x_intermediate_list'] += [vehicle.x_intermediate_list]
            vehicle.xf = vehicle.x_intermediate_list[-3:] + vehicle.xf[3:]
            
        vehicle_positions_new = []
        for i, vehicle in enumerate(self.vehicles):
            vehicle_positions_new += [[vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]]
        self.set_var({'new_positions': {'stage' : self.stage, 'vehicle_positions_new' : vehicle_positions_new}})
            
            # We are not allowed to update the t_intermediate_list, because the vehicle has its "local" time
            # vehicle.t_intermediate_list = vehicle.t_intermediate_list[1:] + [t]
                
        return self
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for position in vehicle_positions:
    #     ax.plot(position[0], position[1], 'ro')
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for vehicle in group.vehicles:
    #     # for x_intermediate_list in vehicle.variable_history['x_intermediate_list'][-1]:
    #     x_intermediate_list = vehicle.variable_history['x_intermediate_list'][4]
    #     print(x_intermediate_list)
    #     for i in [4]:
    #         idx = np.arange(int(vehicle.state_len/2)*i,int(vehicle.state_len/2)*i+int(vehicle.state_len/2))
    #         print(idx)
    #         position = np.array(x_intermediate_list)[idx].tolist()
    #         ax.plot(position[0], position[1], 'ro')
            
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for vehicle in group.vehicles:
    #     # for x_intermediate_list in vehicle.variable_history['x_intermediate_list'][-1]:
    #     x_intermediate_list = vehicle.variable_history['x_intermediate_list'][9]
    #     print(x_intermediate_list)
    #     for i in [4]:
    #         idx = np.arange(int(vehicle.state_len/2)*i,int(vehicle.state_len/2)*i+int(vehicle.state_len/2))
    #         print(idx)
    #         position = np.array(x_intermediate_list)[idx].tolist()
    #         ax.plot(position[0], position[1], 'ro')
            
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()
    
        
        
    def intermediate_position_generator_PENI_full(self, DFM_values = []):
        import time
        t_iter = time.time()
        # Set x0 as starting position
        vehicle_positions = []
        for vehicle in self.vehicles:
            vehicle_positions += [vehicle.x0[:2]]
            
        vehicle_positions_saved = []
        cum_rotation_saved = []
        cum_scaling_saved = []
        cum_rotation = self.cum_rotation
        cum_scaling = self.cum_scaling
        # we start from 0, the first item in vehicle_positions_saved will be the original
        # vehicle_positions, because at timepoint 0 there will be no collision, hence no
        # rotation or scaling will occure
        if DFM_values == []:
            t_waypoints = np.linspace(0, 1, self.DFM_division)
            lookback = self.DFM_lookback
            lookahead = self.DFM_lookahead
        else:
            t_waypoints = DFM_values[0]
            lookback = DFM_values[1]
            lookahead = DFM_values[2]
            
        for t in t_waypoints: # if you change this line, make sure you also change the appropriate for loop at the end of the function if necessary
            
            t_zizz = np.linspace( (t-lookback >= 0) * (t-lookback) + (t-lookback > 0) * 0,
                                  (t+lookahead <= 1) * (t+lookahead) + (t+lookahead > 1) * 1,
                                  10)
            
            vehicle_positions, cum_rotation, cum_scaling = self.intermediate_position_generator_PENI(vehicle_positions, cum_rotation, cum_scaling, t_zizz)
            
            vehicle_positions_saved += [vehicle_positions]
            cum_rotation_saved += [cum_rotation]
            cum_scaling_saved += [cum_scaling]
            
        print('peni full runtime: ' + str(time.time() - t_iter))
        for vehicle in self.vehicles:
            vehicle.x_intermediate_list = []
            vehicle.t_intermediate_list = []
        for vehicle_positions, cum_rotation, t_ in zip(vehicle_positions_saved, cum_rotation_saved, t_waypoints):
            for i, vehicle in enumerate(self.vehicles):
                vehicle.x_intermediate_list += [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]
                vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]]
                vehicle.t_intermediate_list += [t_]
        
        return self
        # return cum_rotation_saved, cum_scaling_saved, vehicle_positions_saved
    
    
    def intermediate_position_generator_PENI(self, vehicle_positions, cum_rotation, cum_scaling, t):
        """The goal of this function is to receive a set of vehicle positions and calculate a rotated-scaled frame, that does not collide with 
        obstaacles at the given time-point.
        If t is a list of time values, then each of these time values will be checked for collision"""
        
        if t[-1] >= self.vehicles[0].t_end + self.vehicles[0].t_window_size:
            self.back_rotation_factor = 1
            self.back_scaling_factor = 1
            print(self.vehicles[0].t_end)
            print(self.vehicles[0].t_end + self.vehicles[0].t_window_size)
        # Step 0: first always try to turn&scale it back... :)
        #Backturning
        rotation_angle = -1 * self.back_rotation_factor * cum_rotation
        # --minimum rotation value--
        if -5/360 * 2 * math.pi <= rotation_angle < 0.0 or 0.0 < rotation_angle <= -5/360 * 2 * math.pi:
            rotation_angle = -1 * cum_rotation
        
        
        # Backscaling
        deviance = abs(1 - 1 / cum_scaling)
        deviance *= self.back_scaling_factor
        
        # if the formation is larger, than the reference size, then we want to shrink it
        if cum_scaling >= 1:
            scaling_factor = 1 - deviance
        # if the formation is smaller, than the reference size, then we want to expand it
        else:
            scaling_factor = 1 + deviance
            
        # --minimum scaling value--
        # if the back-scaling factor changes the size less then 10%, then we scale back completely
        if 0.9 <= scaling_factor < 1.0 or 1.0 < scaling_factor <= 1.1:
            deviance = abs(1 - 1 / cum_scaling)
            if cum_scaling >= 1:
                scaling_factor = 1 - deviance
            else:
                scaling_factor = 1 + deviance
        
        vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
        vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
        collision_saved = []
        for t_ in t:
            all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, self.get_obstacle_corners(t_))
           
            collision_saved += [False]
            if collision == True:
                collision_saved[-1] = True
                break
        
        if all(collision is False for collision in collision_saved):
            # If we are in this if, then no collision happens when we try to rotate/scale back the formation.
            # print('-, -')
            return vehicle_positions_scaled_rotated, cum_rotation + rotation_angle, cum_scaling * scaling_factor
        
        # Otherwise, if we cannot rotate&scale back, find something else:
        # Step 1: generate possible rotation angles & scaling factors
        degree_step = 5
        radian_step = degree_step/360 * 2 * math.pi
        rotation_angles = [[0 + radian_step * i, 0 - radian_step * i] for i in range(1, int( (math.pi/2) / radian_step))]
        rotation_angles = np.array(rotation_angles).reshape(-1).tolist()
        
        scaling_step = 1.5
        scaling_factors_shrink = [  1 / (scaling_step ** i)  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
        scaling_factors_expand = [  1 * scaling_step ** i  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
        scaling_factors = scaling_factors_shrink + scaling_factors_expand
        
        
        # Step 2: We iterate through all possible rotation & scaling possibilities
        # Then, in b) we check, if that specific rotation & sacling results in collision between [t0, tf] or not.
        costs = []
        vehicle_positions_new_saved = []
        rotation_angle_new_saved = []
        scaling_factor_new_saved = []
        collision_saved = []
        for scaling_factor in scaling_factors:
            vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
            for rotation_angle in rotation_angles:
                
                vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
                # Step 2b) check for each t_ in t if collision happens. If yes, do not check further, 
                # the given scaling factor & rotation angle is not good.
                
                # First let's save these :)
                vehicle_positions_new_saved += [vehicle_positions_scaled_rotated]
                rotation_angle_new_saved += [rotation_angle]
                scaling_factor_new_saved += [scaling_factor]
                collision_saved += [False]
                
                collision_saved_tmp = []
                for t_ in t:
                    all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, self.get_obstacle_corners(t_))
                    collision_saved_tmp += [collision]
                    if collision == True:
                        collision_saved[-1] = True
                        costs += [math.inf]
                        break
                # if for any t_ we did not break and put inf cost into costs, then calculate a proper cost   
                if all(collision is False for collision in collision_saved_tmp):
                    # Step 2c) if no collision happens, calculate a cost for the given rotation & scaling combo
                    cost = self.formation_change_cost_calculator(vehicle_positions, vehicle_positions_scaled_rotated, rotation_angle, scaling_factor)
                    # print(scaling_factor, rotation_angle, cost)
                    costs += [cost]
        
        # if we cannot find good solution, do nothing
        if all(collision_saved):
            vehicle_positions_new = vehicle_positions
            cum_rotation = 0
            cum_scaling = 1
            print("We have a problem boss! Every formation candidate collides :/")
        else: 
            # Now we have the costs and everything in order.
            # Let's find the least cost value.
            cost_min = min(costs)
            cost_min_idx = costs.index(cost_min)
            # And the ideal position, rotation angle & scaling factor is:
            vehicle_positions_new = vehicle_positions_new_saved[cost_min_idx]
            rotation_angle_new = rotation_angle_new_saved[cost_min_idx]
            scaling_factor_new = scaling_factor_new_saved[cost_min_idx]
            
            cum_rotation += rotation_angle_new
            cum_scaling *= scaling_factor_new
            
            # print(rotation_angle_new, scaling_factor_new)
        
        
        return vehicle_positions_new, cum_rotation, cum_scaling
        
        
        

        
    
    def formation_change_cost_calculator(self, vehicle_positions_original, vehicle_positions_new, rotation_angle = 0, scaling_factor = 1):
        cost = 0
        alpha_distance = 0.1 * 1 * 0
        alpha_rotation = 0.1
        alpha_scaling_up = 1000
        alpha_scaling_down = 100
        for original, new in zip(vehicle_positions_original, vehicle_positions_new):
            cost += alpha_distance * ((original[0] - new[0])**2 + (original[1] - new[1])**2)
            
            
        cost += alpha_rotation * abs(rotation_angle)
        # cost += alpha_scaling * abs(1-scaling_factor)
        if scaling_factor > 1:
            cost += alpha_scaling_up * abs(1-scaling_factor)
        if scaling_factor < 1:
            cost += alpha_scaling_down * abs(1-scaling_factor)
                
            # cost += alpha_scaling_up * scaling_factor * abs(1-scaling_factor)
            
            
        # # Dereasing the cost, if we are forming back to the original formation
        # for original, new in zip(self.og_final_positions, vehicle_positions_new):
        #     cost += alpha_distance * (original[0] - new[0])**2 + (original[1] - new[1])**2
            
            
        # sign = (1 - scaling_factor) * (1 - self.scaling_factor)
        # if sign < 0:
        #     cost -= np.mean([alpha_scaling_down, alpha_scaling_up]) * abs(scaling_factor - self.scaling_factor)
            
        # sign = (rotation_angle * self.rotation_angle)#  / abs(rotation_angle * self.rotation_angle)
        # if sign < 0:
        #     cost -= alpha_rotation * abs(rotation_angle - self.rotation_angle)
            
        return cost 
    
    def rotate_formation(self, vehicle_positions, angle):
        vehicle_positions_new = []
        for position in vehicle_positions:
            vehicle_positions_new += [self.rotate_vector(position, angle)]
            
        return vehicle_positions_new
        
    def scale_formation(self, vehicle_positions, scaling_factor):
        vehicle_positions_new = []
        for position in vehicle_positions:
            vehicle_positions_new += [self.scale_vector(position, scaling_factor)]
            
        return vehicle_positions_new
        
    def rotate_vector(self, vector, angle):
        a1, a2 = vector[0], vector[1]
        
        return (a1 * math.cos(angle) - a2 * math.sin(angle), \
                        a1 * math.sin(angle) + a2 * math.cos(angle))
            
    def shift_vector(self, vector1, vector2):
        a1, a2 = vector1[0], vector1[1]
        b1, b2 = vector2[0], vector2[1]
        
        return [a1 + b1, a2 + b2]
        
        
    def scale_vector(self, vector, scaling_factor):
        scaled_vector = [vector[0] * scaling_factor, vector[1] * scaling_factor]
        
        return scaled_vector
    
    def check_point_inside(self, point, obstacle_corners):
        
        # 1. Obtain obstacle center
        obst_center = self.get_obstacle_center(obstacle_corners)
        # 1.5 Moove the point to the center (and also moove the obstacle corners with it)
        corners = [ [corner[0] - point[0], corner[1] - point[1]] for corner in obstacle_corners  ]
        # 2. Transform obstacle & circle
        corners = [ [corner[0] - obst_center[0], corner[1] - obst_center[1]] for corner in corners  ]
        d_zone_center = [-obst_center[0], -obst_center[1]]
        # 3. calculate angle of obstacle
        obst_angle = self.get_obstacle_angle(corners)
        # 4. rotate obstacle & circle
        new_corners = self.rotate_formation(corners, -obst_angle)
        new_d_zone_center = self.rotate_formation([d_zone_center], -obst_angle)
        # 5. check intersection
        s_danger = 0.0
        inside = self.intersects([list(new_d_zone_center[0]), s_danger], new_corners)
        
            
        return inside
            
        
    
    def check_danger_zone_with_obstacles(self, vehicle_positions, obstacle_corners):
        any_inside = []
        
        # Getting the largest distance from the origo
        s_danger = -math.inf
        for position in vehicle_positions:
            vehicle_distance_from_origo = np.sqrt((0-position[0])**2 + (0-position[1])**2)
            if vehicle_distance_from_origo > s_danger:
                s_danger = vehicle_distance_from_origo
        # print(s_danger)
        s_danger = 0.6988905493709299
        # Check if any of the obstacles are in the danger zone
        any_inside = []
        corner_inside = []
        
        "Old, foolish code:"
        # for corners in obstacle_corners:
        #     for corner in corners:
        #         corner_distance_from_origo = np.sqrt((0-corner[0])**2 + (0-corner[1])**2)
        #         if corner_distance_from_origo <= s_danger: # is inside the danger zone?
        #             corner_inside += [True]
        #         else:
        #             corner_inside += [False]
            
            
        # from sympy import Point, Polygon, Circle
        # d_zone = Circle(Point(0, 0), s_danger)
        # for corners in obstacle_corners:
        #     poly_corners = [(corner) for corner in corners]
        #     poly_obstacle = Polygon(*poly_corners)
        #     # poly_obstacle = Polygon(poly_corners[0], poly_corners[1], poly_corners[2], poly_corners[3])
        #     isIntersection = d_zone.intersection(poly_obstacle)
            
        "New, highly professional code"
        for corners in obstacle_corners:
            # 1. Obtain obstacle center
            obst_center = self.get_obstacle_center(corners)
            # 2. Transform obstacle & circle
            corners = [ [corner[0] - obst_center[0], corner[1] - obst_center[1]] for corner in corners  ]
            d_zone_center = [-obst_center[0], -obst_center[1]]
            # 3. calculate angle of obstacle
            obst_angle = self.get_obstacle_angle(corners)
            # 4. rotate obstacle & circle
            new_corners = self.rotate_formation(corners, -obst_angle)
            new_d_zone_center = self.rotate_formation([d_zone_center], -obst_angle)
            # 5. check intersection
            is_intersection = self.intersects([list(new_d_zone_center[0]), s_danger], new_corners)
            corner_inside += [is_intersection]
            
        # return corner_inside, is_intersection
        return corner_inside, any(corner_inside)
        
    def get_obstacle_center(self, corners):
        mean_x = np.array([])
        mean_y = np.array([])
        
        for corner in corners:
            mean_x = np.append(mean_x, corner[0])
            mean_y = np.append(mean_y, corner[1])
            
        return [np.mean(mean_x), np.mean(mean_y)]
    
    def get_obstacle_angle(self, corners):
        corners = corners + [corners[0]]
        
        edge_vectors = [ [p2[0] - p1[0], p2[1] - p1[1]]  for p1, p2 in zip(corners[:-1], corners[1:])]
        
        angles = [math.atan2(vector[1], vector[0]) for vector in edge_vectors]
        
        angles = np.array([])
        for vector in edge_vectors:
            vector = np.array(vector)
            x_axis = np.array([1, 0])
            alpha = math.acos( np.dot(vector, x_axis) / (np.linalg.norm(vector) * 1))
            angles = np.append(angles, alpha)
        angles_degree = angles / math.pi * 180
        min_rot_angle = max(angles)
        for angle in angles:
            min_rot_angle = min_rot_angle * (angle < 0 or angle > min_rot_angle) + angle * (not(angle < 0 or angle > min_rot_angle))
        
        return min_rot_angle
    
    
    # fig, ax = plt.subplots()
    # for pos in corners:
    #     ax.plot(pos[0], pos[1], 'bo')
    # for pos in new_corners:
    #     ax.plot(pos[0], pos[1], 'go')
        
    #     circle = plt.Circle(new_d_zone_center[0], s_danger, color='k', alpha=0.5, zorder = 10)
    #     ax.add_patch(circle)
        
    # for pos in vehicle_positions:
    #     ax.plot(pos[0], pos[1], 'ro')
        
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()
        
    
    # def rotate_formation(self, formation):
    #     return "Done already :)"
    
    def intersects(self, circle, rect):
        # https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
    
        circle_x = circle[0][0]
        circle_y = circle[0][1]
        circle_radious = circle[1]
        rect_x, rect_y = self.get_obstacle_center(rect)
        rect_width = abs(rect[0][0] * 2)
        rect_height = abs(rect[0][1] * 2)
        circleDistance_x = abs(circle_x - rect_x);
        circleDistance_y = abs(circle_y - rect_y);
    
        if (circleDistance_x > (rect_width/2 + circle_radious)):
            return False
        if (circleDistance_y > (rect_height/2 + circle_radious)):
            return False
    
        if (circleDistance_x <= (rect_width/2)):
            return True
        if (circleDistance_y <= (rect_height/2)):
            return True
    
        cornerDistance_sq = (circleDistance_x - rect_width/2)**2 + \
                             (circleDistance_y - rect_height/2)**2
    
        return cornerDistance_sq <= circle_radious**2
    
    # def check_danger_zone_with_obstacle(self, vehicle_pos, obstacle_corners):
    #     import matplotlib.path as mpltPath
    #     path = mpltPath.Path(obstacle_corners)
    #     try:
    #         inside = path.contains_points(np.array([vehicle_pos]))[0] #, radius = 0.001)
    #     except:
    #         kappa = True
            
    #     return inside
        
    def check_collision_with_obstacles(self, vehicle_positions, obstacle_corners):
        any_inside = []
        for corners in obstacle_corners:
            for position in vehicle_positions:
                try:
                    any_inside += [self.check_collision_with_obstacle(position, corners)]
                except:
                    kappa = True
                        
        return any_inside, any(any_inside)
                    
    
    def check_collision_with_obstacle(self, vehicle_pos, obstacle_corners):
        import matplotlib.path as mpltPath
        path = mpltPath.Path(obstacle_corners)
        try:
            inside = path.contains_points(np.array([vehicle_pos]))[0] #, radius = 0.001)
        except:
            kappa = True
            
        return inside
    
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
        

    def set_group_position(self, position : list, targetHeight : float = 0.8,  position_type : str = 'initial'):


        # targetHeight = 1.3
        # def float_representer(dumper, value):
        #     text = '{0:.4f}'.format(value)
        #     return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
        # yaml.add_representer(float, float_representer)
        if position_type == 'initial':
            position = self.start_position
            positions = self.position_generator(centerpoint = position, n_positions = len(self.vehicles), r = self.vehicles[0].radious * 6)
            positions = self.ellipse_generator(centerpoint = position, n_positions = len(self.vehicles), a = self.vehicles[0].radious * 6, b = self.vehicles[0].radious * 3,
                                               ellipse_rotation = math.pi / 2 + math.pi / 4)
            
            
            
    # def ellipse_generator(self, centerpoint : list, n_positions : int, a : float, b : float, 
    #                       ellipse_rotation : float = 0, vehicles_rotation : float = 0,
    #                       ellipse_scale_x : float = 1, ellipse_scale_y : float = 1):
        
        
            if self.stage == 0:
                yaml_dict = {'crazyflies' : []}
                for i, vehicle in enumerate(self.vehicles):
                    # initialPosition = [positions[i][j].tolist() for j in range(len(positions[i]))] + [targetHeight]
                    initialPosition = [float(pos) for pos in positions[i]]
                    initialPosition = initialPosition[:2] + [float(targetHeight)]
                    yaml_dict['crazyflies'] += [{'id' : i, 'channel' : 100,
                                                 'initialPosition' : initialPosition,
                                                 'type' : 'default'
                                                 }]
                if yaml is not None:
                    with open(self.cwd + "/yaml/initialPosition.yaml", "w") as file_descriptor:
                        yaml.dump(yaml_dict, file_descriptor)

            # "But actually we want to read in :)"

            # # Read yaml
            # with open("initialPosition.yaml", "r") as file:
            #     initialPosition = yaml.load(file, Loader=yaml.FullLoader)

            # # Assign these as starting positions
            # positions = []
            # for i, vehicle in enumerate(self.vehicles):
            #     positions += [initialPosition['crazyflies'][i]['initialPosition'][:2]]
        elif position_type == 'final':
            position = self.goal_position
            positions = self.position_generator(centerpoint = position, n_positions = len(self.vehicles), r = self.vehicles[0].radious * 6)
            positions = self.ellipse_generator(centerpoint = position, n_positions = len(self.vehicles), a = self.vehicles[0].radious * 6, b = self.vehicles[0].radious * 3,
                                               ellipse_rotation = math.pi / 2 + math.pi / 4)
            self.og_final_positions = positions
            if self.stage == 0:
                yaml_dict = {'crazyflies' : []}
                for i, vehicle in enumerate(self.vehicles):
                    # initialPosition = [positions[i][j].tolist() for j in range(len(positions[i]))] + [targetHeight]
                    initialPosition = [float(pos) for pos in positions[i]]
                    initialPosition = initialPosition[:2] + [float(targetHeight)]
                    yaml_dict['crazyflies'] += [{'id' : i, 'channel' : 100,
                                                 'finalPosition' : initialPosition,
                                                 'type' : 'default'
                                                 }]
                if yaml is not None:
                    with open(self.cwd + "/yaml/finalPosition.yaml", "w") as file_descriptor:
                        yaml.dump(yaml_dict, file_descriptor)
        else:
            NotImplementedError()


        for i in range(len(self.vehicles)):
            self.vehicles[i].set_position(position = positions[i], position_type = position_type)


    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    def position_generator(self, centerpoint : list, n_positions : int, r : float):
        positions = []
        alpha = np.pi / 4.0 + np.pi / 8.0 # initial angle
        alpha += centerpoint[2]
        for i in range(n_positions):
            positions += [ [centerpoint[0] + r * np.sin(alpha), centerpoint[1] + r * np.cos(alpha), centerpoint[2]] ] # [x, vx, y, vy, z, vz]
            alpha = alpha - np.pi * 2.0 / n_positions

        return positions
    
    # def ellipse_generator(self,
    # import functools
    # @functools.lru_cache(maxsize=None)
    def ellipse_generator(self, centerpoint : list, n_positions : int, a : float, b : float, 
                          ellipse_rotation : float = 0, vehicles_rotation : float = math.pi / 4,
                          ellipse_scale_x : float = 1, ellipse_scale_y : float = 1):
        # centerpoint = [0, 0, 0]
        # n_positions = 30
        # a = self.vehicles[0].radious * 6
        # b = self.vehicles[0].radious * 2
        
        # Rotating vehicles on the ellipse
        alpha = 0 + vehicles_rotation
        # Scaling
        a *= ellipse_scale_x
        b *= ellipse_scale_y
        positions = []
        for i in range(n_positions):
            positions += [ [centerpoint[0] + a * np.cos(alpha), centerpoint[1] + b * np.sin(alpha), centerpoint[2]] ] # [p, q, phi]
            # Because of having cos_phi and sin_phi instead of a single phi value, our position array will be 4 long
            # positions += [ [centerpoint[0] + a * np.cos(alpha), centerpoint[1] + b * np.sin(alpha), np.cos(centerpoint[2]), np.sin(centerpoint[2])] ] # [p, q, cos_phi, sin_phi]
            alpha += np.pi * 2.0 / n_positions
        
        # Rotating the ellipse itself with the vehicles already in place
        if ellipse_rotation != 0 and centerpoint[:2] == [0.0, 0.0]:
            for i, pos in enumerate(positions):
                positions[i][:2] = self.rotate_vector(pos[:2], ellipse_rotation)
                
        if ellipse_rotation != 0 and centerpoint[:2] != [0.0, 0.0]:
            # raise NotImplementedError("Please set the centerpoint to [0, 0]")
            # Actually, not an error... We can still rotate around (0, 0) and then shift the 
            # corners with the centerpoint vector.
            
            for i, pos in enumerate(positions):
                positions[i][:2] = self.rotate_vector(pos[:2], ellipse_rotation)
                positions[i][:2] = self.shift_vector(positions[i][:2], centerpoint[:2])
                
            
            
            
        
        return positions
        # plt.figure()
        # for pos in positions:
        #     plt.plot(pos[0], pos[1], '.')
        # plt.show()
        # kappa = True
        # return self
    
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    

    def add_obstacles(self, obstacles : list):
        for i in range(len(self.vehicles)):
            self.vehicles[i].obstacles = obstacles
        return self


    def organise_neighbours(self):
        """Sets up the l_neighbours for the vehicles in the group. Every agent
        n number of neighbours are assigned. The value n is hard-coded in
        the is_this_my_neighbour() function.

        Parameters
        ----------

        Returns
        -------
        self
        """
        for i in range(len(self.vehicles)):
            neighbours = []
            for j in range(len(self.vehicles)):
                my_id = self.vehicles[i].ID
                neighbour_id = self.vehicles[j].ID
                condition = self.is_this_my_neighbour(my_id = my_id, neighbour_id = neighbour_id)
                if condition == True:
                    neighbours += [self.vehicles[j]]
            self.vehicles[i].neighbours = neighbours
        return self

    def is_this_my_neighbour(self, my_id : int, neighbour_id : int):
        """Helper function, which tells wether neighbour_id is a
           neighbour of my_id or not
           (currently 2-distance neighbourhood is hard-coded -> max_on)

        Parameters
        ----------
        my_id : int
            ID of current vehicle
        neighbour_id : int
            ID of the other vehicle, whose neighbourhood is questioned

        Returns
        -------
        res : bool
            True if neighbour_id is a neighbour, False otherwise
        """
        # First we have to create a list of numbers
        old_list = np.linspace(0, len(self.vehicles) - 1, len(self.vehicles))
        # We cut the list where I am at. Put the first cut in to the front, the rest to the back
        # Example: [0, 1, 2!, 3, 4, 5, 6] -> [2!, 3, 4, 5, 6, 0, 1]
        new_list = np.append( old_list[my_id:] , old_list[0:my_id] )
        old_list = new_list
        max_on = 4 # max_observed_neighbours. How many neighbours we have on our right and on our left. 2-> 2+2=4 neighbours
        if(max_on + 1 < len(new_list) and 0 < len(new_list)):
            if any([neighbour_id == i for i in old_list[1:max_on + 1]]) or any( [neighbour_id == i for i in old_list[-max_on:]] ):
                res = True
            else:
                res = False
        elif max_on + 1 >= len(new_list):
            # No need to check anything further, because in the outer
            # loop we are only looking at potential neighbours anyway
            if my_id != neighbour_id:
                res = True
            else:
                res = False
        else:
            res = False
        return res

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def initialize_values(self):
        return run_initialize_values(self)

    def prepare(self):
        return run_prepare(self)
    
    def simulation_step(self):
        return run_simulation_step(self)
            
    def set_var(self, var):
        return run_set_var(self, var)
    def set_simulation(self, simulation = False):
        return run_set_simulation(self, simulation=simulation)
        
    def data_exchange_x(self):
        return run_data_exchange_x(self)
    
    def lambda_update_data_exchange_z(self):
        return run_lambda_update_data_exchange_z(self)
        
        

    def solve(self):
        return run_solve(self)


    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################

    def write_iteration_times(self, prefix = ''):
        return run_write_iteration_times(self, prefix=prefix)



    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################

    def generate_obstacles(self, seed : int = 42):
        # There are the following types of obstacles:
            # - obstacles on the path: these obstacles are created using the ellipse generator algorithm.
            # It's centerpoint, alpha, a, b is a random number in the Frenet frame (all of which in a defined bound).
            # Then, the corners are transformed from Frenet to Inertial.
            # - gates: two corners of each obstacle, that form a gate are generated with the 
            # ellipse generator algorithm. However alpha is always zero and b has a minimum value.
            # (both of these constarints ensure, that there is a tunner, kinda parallel with the Frenet path so that 
            # the DFG algorithm will be able to find a solution.)
            # The corners are transformed from Frenet to Inertial and extended to the environment limits.
            # - wall on one side: same as the gate, but drops one of the obstacle, that forms a gate.
        
        # Spacing of the obstacles:
            # randomly, but at least t_spacing between each obstacle.
            # no obstacle is allowed at the end
            
        # Ellipse generator input:
            # def ellipse_generator(self, centerpoint : list, n_positions : int, a : float, b : float, 
            #                       ellipse_rotation : float = 0, vehicles_rotation : float = math.pi / 4,
            #                       ellipse_scale_x : float = 1, ellipse_scale_y : float = 1):
             
                
        # variables
        t_spacing = 0.2
        t_free_begin = 0.2
        t_free_end = 0.2
        
        
        n_obst_along = 3
        random.seed(seed)
        centerpoint_x_bound = [-0.1, 0.1]
        centerpoint_y_bound = [-0.1, 0.1]
        
        a_bound = [0.1, 0.7]
        b_bound = [0.1, 1.5]
        
        alpha_bound = [-math.pi/2, math.pi/2]
        
        
        # Generate obstacles along the way
        obstacles = []
        t_bound = []
        t_tmp = [0.4, 0.6, 0.8]
        for i in range(n_obst_along):
            centerpoint = [random.uniform(centerpoint_x_bound[0], centerpoint_x_bound[1]), \
                           random.uniform(centerpoint_y_bound[0], centerpoint_y_bound[1]), 0 ]
                
            a = random.uniform(a_bound[0], a_bound[1])
            b = random.uniform(b_bound[0], b_bound[1])
            alpha = random.uniform(alpha_bound[0], alpha_bound[1])
            ellipse_corners = self.ellipse_generator( centerpoint = centerpoint, n_positions = 4, a = a, b = b, 
                                                      ellipse_rotation = alpha, vehicles_rotation = math.pi / 4)
            
            obstacle_corners = [corner[:2] for corner in ellipse_corners]
            t = random.uniform(a_bound[0], a_bound[1])
            # we need to place them at random location along the path
            # this is done by converting their frenet coordinates to the inertial frame at random times
            obstacle_corners = [self.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i]) for corner in obstacle_corners]
            obstacles += [Obstacle(ID = i, corners = obstacle_corners)]
        
        # Generate gates
        # Okay. We have generated obstacles along the way.
        # Let's generate gates now! :)
        n_obst_gate = 3
        gate_gap_bound = [3.5 * self.vehicles[0].radious, 10 * self.vehicles[0].radious]
        gate_length_bound = 0.3 # 0.3
        
        
        # obstacles = []
        t_tmp = [0.3, 0.5, 0.7]
        for i in range(n_obst_gate):
            gate_points_tmp = random.uniform(gate_gap_bound[0], gate_gap_bound[1])
            # The lower part of the gate
            # corners = [top-right, top_left]
            gate1_inside_corners = [ [gate_length_bound / 2, -gate_points_tmp], \
                                     [-gate_length_bound / 2, -gate_points_tmp] ]
            
            # corners = [bottom-left, bottom-right]
            gate2_inside_corners = [ [gate_length_bound / 2, gate_points_tmp], \
                                     [-gate_length_bound / 2, gate_points_tmp] ]
                
            # transforming the frenet coordinates to inertial frame at random times
            g1 = [list(self.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i])) for corner in gate1_inside_corners]
            g2 = [list(self.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i])) for corner in gate2_inside_corners]
            
            if i == -1:
                pass
            else:
                # Extending till the edge of the environment
                g1 = g1 + [[g1[-1][0], -6]] + [[g1[0][0], -6]]
                g2 = g2 + [[g2[-1][0],  6]] + [[g2[0][0],  6]]
                obstacles += [Obstacle(ID = 3+i*2, corners = g1)]
                obstacles += [Obstacle(ID = 3+i*2 + 1, corners = g2)]
                # Sharing ID-s between gate pairs
                obstacles[-2].gate_pair_ID = obstacles[-1].ID
                obstacles[-1].gate_pair_ID = obstacles[-2].ID
        
        # plt.figure()
        # for pos in ellipse_corners:
        #     plt.plot(pos[0], pos[1], 'ro')
        # plt.show()
            
        
        # [print(obst.corners) for obst in obstacles]
        return obstacles

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    "For plotting stuff"





    def calculate_formation_error(self):
        
        angle_errors_intermediate_all = []
        for intermediate_ADMM_idx in range(self.n_intermediate_ADMM):
            angle_errors = []
            for vehicle in self.vehicles:
                t, angle_errors_tmp = vehicle.calculate_formation_error(intermediate_ADMM_idx = intermediate_ADMM_idx)
                angle_errors += [angle_errors_tmp]
            
                
            # return angle_errors
            angle_errors_mean = []
            for i in range(len(angle_errors_tmp)):
                angle_errors_sum_tmp = np.array(angle_errors[0][0]) * 0
                for j in range(len(self.vehicles)):
                    # angle_errors_sum_tmp += np.linalg.norm(angle_errors[j][i])
                    angle_errors_sum_tmp += np.array(angle_errors[j][i])
                angle_errors_mean += [angle_errors_sum_tmp / len(self.vehicles)]
            angle_errors_intermediate_all += [angle_errors_mean]
        
        
        from matplotlib.pyplot import cm
        color=cm.rainbow(np.linspace(0,1,self.n_intermediate_ADMM))
        
        plt.figure()
        for angle_errors_mean, c in zip(angle_errors_intermediate_all, color):
            for time, angle in zip(t, angle_errors_mean):
                plt.plot(time, angle, c = c)
        plt.show()

    def plot_frenet_view(self):
        return render_plot_frenet_view(self)
    
    def plot_moovie_frames(self, n_frames, iternum : int = 0, seed = ''):
        return render_plot_moovie_frames(self, n_frames, iternum=iternum, seed=seed)
    
    
    

