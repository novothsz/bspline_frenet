from .vehicle import Vehicle
import numpy as np
import math
import matplotlib.pyplot as plt
from .frenet_path import FrenetPath
from .environment import Environment
import yaml
from numpy import interp

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
        
        
        self.ACC_MPC_pos_queue = []
        self.ACC_MPC_t_queue = []
        
        
        
    
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
        "To this end, we introduce, ... :))"
        # nothing, dude, just call the sweep_ACC :)
        # We call the function BEFORE the simulation step, therefore to get the correct values for the next iteration, lets add a t_step to the values :)
        
        t_sweep_start = self.vehicles[0].t_start + self.vehicles[0].t_step
        t_sweep_end = self.vehicles[0].t_end + self.vehicles[0].t_step
        
        
        
        if self.stage == 7:
            kappa = True
        
        # Okay, there is actually a little difference compared to the simple sweep:
            # 1. We need to clear the intermediate values before every sweep
            # 2. We need to duplicate the last values so that the length is consistent. 
            # 3. (or truncate)
            
        # Step 1: clear the intermediate lists
        for i, vehicle in enumerate(self.vehicles):
            vehicle.x_intermediate_list = []
            vehicle.t_intermediate_list = []
            vehicle.t_real_intermediate_list = []
            vehicle.t_real_activation_list = []
        
        max_len_x = self.vehicles[0].n_of_saved_waypoints * 3
        max_len_t = self.vehicles[0].n_of_saved_waypoints
        self.sweep_ACC(t_sweep_start = t_sweep_start, t_sweep_end = t_sweep_end)
        
        # Before we do this duplication stuff, we need to find the next "current_configuration_position" value
        # What do we do? So if in the next step we will be in between some intermediate-calculated values, then we shall
        # use the formation configuration valid at that timestep as a starting-point for the next DFG iteration.
        # How do we do that?
        # So if history of t_real_intermediate_list doesn't exist, we don't do anything
        # If it does, then, we search for the highest index in the list, that is lower, than t_sweep_start
        # use that index to extract the correct x_intermediate_list
        
        for v, vehicle in enumerate(self.vehicles):
            greater_ = False
            index_ = 0
            if len(vehicle.variable_history['t_real_intermediate_list']) > 0: # asszem ez csak annyi, hogy nem az első iteráció...
                for i, t in enumerate(vehicle.variable_history['t_real_activation_list'][-1]):
                    greater_new = (t[0] <= t_sweep_start + vehicle.t_step)
                    if (greater_ == True) and (greater_new == False):
                        index_ = i-1
                        break
                    greater_ = greater_new
            if index_ >= 0 and len(vehicle.variable_history['t_real_intermediate_list']) > 0:
                idx = np.arange(int(vehicle.state_len/2)*index_,int(vehicle.state_len/2)*index_+int(vehicle.state_len/2))
                new_ = np.array(vehicle.variable_history['x_intermediate_list'][-1]).reshape(-1)[idx].tolist()[:2]
                vehicle.current_configuration_position = new_
                
        for i, vehicle in enumerate(self.vehicles):
            x_intermediate_list = vehicle.x_intermediate_list
            # vehicle.variable_history['x_intermediate_list']
            t_intermediate_list = vehicle.t_intermediate_list
            # t_real_intermediate_list = vehicle.t_intermediate_list
            
            current_len_x = len(x_intermediate_list)
            current_len_t = len(t_intermediate_list)
            if len(x_intermediate_list) > max_len_x:
                vehicle.x_intermediate_list = x_intermediate_list[:int((current_len_x-max_len_x)/3)]
                vehicle.t_intermediate_list = vehicle.t_intermediate_list[:(current_len_t-max_len_t)]
                vehicle.t_real_intermediate_list = vehicle.t_real_intermediate_list[:(current_len_t-max_len_t)]
            if len(x_intermediate_list) < max_len_x:
                diff_x = max_len_x - current_len_x
                diff_t = max_len_t - current_len_t
                vehicle.x_intermediate_list = vehicle.x_intermediate_list + vehicle.x_intermediate_list[-3:] * int(diff_x/3)
                vehicle.t_intermediate_list = vehicle.t_intermediate_list + [vehicle.t_intermediate_list[-1]] * diff_t
                vehicle.t_real_intermediate_list = vehicle.t_real_intermediate_list + [vehicle.t_real_intermediate_list[-1]] * diff_t
                
            vehicle.variable_history['x_intermediate_list'][-1] = vehicle.x_intermediate_list
            vehicle.variable_history['t_intermediate_list'][-1] = vehicle.t_intermediate_list
            vehicle.variable_history['t_real_intermediate_list'][-1] = vehicle.t_real_intermediate_list
            if len(vehicle.t_intermediate_list) < max_len_t or len(vehicle.x_intermediate_list) < max_len_x:
                kappa = True
        return self
        
    
    
    
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
        import time
        t_iter = time.time()
        
        t_step = 0.01
        
        vehicle_positions = []
        for vehicle in self.vehicles:
            # vehicle_positions += [vehicle.x0[:2]]
            vehicle_positions += [vehicle.current_configuration_position[:2]]
            
        vehicle_positions_original = np.array(vehicle_positions).tolist()
                
                
        # s_danger = 0.5
        
        t_end = t_sweep_start # TODO: jajj, ne hívjuk már t_end-nek...
        while t_end <= t_sweep_end:
            
            # Check for collision
            
            # If no collision: step or rotate back
            # If collision: step forward, until no collision.
            #   save t_safe
            # find the appropriate formation during this dangerous time.
            
            # necessary functions:
                # - function checking collision at time t (done)
                # - function, which upon detection of collision finds the safe time
                # - function, which finds the formation configuration during the danger time
                # time splitting if no solution has been found.
                
                # save the intermediate step
                # increase the time to t_danger_end + t_step :)
            
            "Actually, let's check, if it is inside of the danger zone or not."
            
            
            
            # Step 1: check collision at time t
            # all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions, self.get_obstacle_corners(t_end))
            all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions, self.get_obstacle_corners(t_end))
            obstacle_idx = []
            for i, obstacle in enumerate(self.vehicles[0].obstacles):
                idx = np.arange(len(self.vehicles)*i,len(self.vehicles)*i+len(self.vehicles))
                is_collision = any(np.array(all_collisions)[idx].tolist())
                if is_collision:
                    obstacle_idx += [i]
            # Step 2: if collision has been found, find t_danger_end time
            if obstacle_idx != []:
                t_danger_start = t_end
                t_danger_end = t_danger_start
                t_danger_step = t_step / 10 # if collision is detected, we step this 'smoothly' until no danger is detected
                while t_danger_end <= 1:
                    t_danger_end += t_danger_step
                    # Check collision but only with obstacles, which are dangerous (we expect, that this doesn't change. Meaning:
                    # obstacles are spaced out well.)
                    # all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions, np.array(self.get_obstacle_corners(t_danger_end))[obstacle_idx].tolist())
                    all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions, np.array(self.get_scaled_obstacle_corners(t_danger_end))[obstacle_idx].tolist())
                    if collision == False:
                        # this is the t_danger_end we were looking for
                        break
                # Okay... we have [t_danger_start, t_danger_end]
                # Let's get the right formation configuration
                
                
                t = (t_danger_start + t_danger_end) / 2
                lookback = (t_danger_end - t_danger_start) / 2 * 1.0
                lookahead = (t_danger_end - t_danger_start) / 2 * 1.0
                t_zizz = np.linspace( (t-lookback >= 0) * (t-lookback) + (t-lookback > 0) * 0,
                                          (t+lookahead <= 1) * (t+lookahead) + (t+lookahead > 1) * 1,
                                          10)
                
                # Step: Get the least cost formation & ACTION_TAKEN
                cum_rotation = self.cum_rotation
                cum_scaling = self.cum_scaling
                
                vehicle_positions, cum_rotation, cum_scaling, ACTION_TAKEN = self.intermediate_position_generator_SZILARD(vehicle_positions, cum_rotation, cum_scaling, t_zizz)
            
                self.cum_rotation = cum_rotation
                self.cum_scaling = cum_scaling
                
                    
                # Create a t_intermediate & x_intermediate from this
                if ACTION_TAKEN == "back_transformation" or ACTION_TAKEN == "yes":
                    
                    if self.MPC_version == 'MPC_param':
                        # we need to account for the fact, that t_global_horizon != t_local_horizon
                        # AAAND for the fact, that the optimization problem is pre-built. So the intermediate lists have to have a certain length always!!!
                        # We need to convert t_global_horizon to t_local_horizon
                        t_local = interp(t,[t_sweep_start,t_sweep_end],[0,1])
                        # t_local = t - t_sweep_start
                        # t_local = (t <= t_sweep_end) * t_local + (t > t_sweep_end) * t_sweep_end
                        for i, vehicle in enumerate(self.vehicles):
                            vehicle.x_intermediate_list += [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]
                            # vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]]
                            vehicle.t_intermediate_list += [t_local]
                            vehicle.t_real_intermediate_list += [t]
                            vehicle.t_real_activation_list += [[t_danger_start, t_danger_end]]
                            # vehicle.variable_history['t_intermediate_list'] += [t_local]
                        
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
        if t_end >= t_sweep_end or self.vehicles[0].t_intermediate_list == []:
            if tmp_len > 0:
                if self.vehicles[0].t_intermediate_list[-1] + 0.1 < t_sweep_end:
                    for i, vehicle in enumerate(self.vehicles):
                        vehicle.x_intermediate_list += [vehicle_positions_original[i][0], vehicle_positions_original[i][1], 0]
                        # vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions_original[i][0], vehicle_positions_original[i][1], 0]]
                        t_local = interp(t_sweep_end,[t_sweep_start,t_sweep_end],[0,1])
                        vehicle.t_intermediate_list += [t_local]
                        vehicle.t_real_intermediate_list += [t_sweep_end]
                        # vehicle.variable_history['t_intermediate_list'] += [1]
            else:
                for i, vehicle in enumerate(self.vehicles):
                    vehicle.x_intermediate_list += [vehicle_positions_original[i][0], vehicle_positions_original[i][1], 0]
                    # vehicle.variable_history['x_intermediate_list'] += [[vehicle_positions_original[i][0], vehicle_positions_original[i][1], 0]]
                    t_local = interp(t_sweep_end,[t_sweep_start,t_sweep_end],[0,1])
                    vehicle.t_intermediate_list += [t_local]
                    vehicle.t_real_intermediate_list += [t_sweep_end]
                    # vehicle.variable_history['t_intermediate_list'] += [1]
                
        
        # Saving stuff to the history
        for i, vehicle in enumerate(self.vehicles):
            vehicle.variable_history['x_intermediate_list'] += [vehicle.x_intermediate_list]
            vehicle.variable_history['t_intermediate_list'] += [vehicle.t_intermediate_list]
            vehicle.variable_history['t_real_intermediate_list'] += [vehicle.t_real_intermediate_list]
            vehicle.variable_history['t_real_activation_list'] += [vehicle.t_real_activation_list]
            
        print('ACC_sweep full time: ' + str(time.time() - t_iter))
        
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
    
    def intermediate_position_generator_SZILARD_sweep(self):
        from .spline_extra import definite_integral
        s = np.sqrt(definite_integral(self.fp.fx_spline.derivative()**2 + self.fp.fy_spline.derivative()**2,0, 1))    
        T = 10
        v_frenet = s / T
        
        s_max = self.vehicles[0].radious * 5 * 2
        s_max = 1.39
        v_max = v_frenet * 5
        t_DFG = s_max / (v_max - abs(v_frenet))
        
    
        
        t_step = t_DFG.reshape(-1).tolist()[0] # self.vehicles[0].t_step
        
        self.lookback = 0
        self.lookahead = t_step
        print('lookback&lookahead has benn changed')
        
        
        # t_step = 0.02
        t_end = 0
        while t_end <= 1:
            self.intermediate_position_generator_SZILARD_action(t_end)
            t_end += t_step
        
        return self
    
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
            # all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, self.get_scaled_obstacle_corners(t_))
            all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions_scaled_rotated, self.get_scaled_obstacle_corners(t_))
           
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
        
        
    def scale_vector(self, vector, scaling_factor):
        scaled_vector = [vector[0] * scaling_factor, vector[1] * scaling_factor]
        
        return scaled_vector
    
    def check_danger_zone_with_obstacles(self, vehicle_positions, obstacle_corners):
        any_inside = []
        
        # Getting the largest distance from the origo
        s_danger = -math.inf
        for position in vehicle_positions:
            vehicle_distance_from_origo = np.sqrt((0-position[0])**2 + (0-position[1])**2)
            if vehicle_distance_from_origo > s_danger:
                s_danger = vehicle_distance_from_origo
        # print(s_danger)
        # s_danger = 0.6988905493709299
        # Check if any of the obstacles are in the danger zone
        any_inside = []
        corner_inside = []
        for corners in obstacle_corners:
            # corner_inside = []
            for corner in corners:
                corner_distance_from_origo = np.sqrt((0-corner[0])**2 + (0-corner[1])**2)
                if corner_distance_from_origo <= s_danger: # is inside the danger zone?
                    corner_inside += [True]
                else:
                    corner_inside += [False]
            # if any(corner_inside):
            #     any_inside += [True]
            # else:
            #     any_inside += [False]
                
                        
        return corner_inside, any(corner_inside)
        
    
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
            positions += [ [centerpoint[0] + a * np.cos(alpha), centerpoint[1] + b * np.sin(alpha), centerpoint[2]] ] # [x, vx, y, vy, z, vz]
            alpha += np.pi * 2.0 / n_positions
        
        # Rotating the ellipse itself with the vehicles already in place
        if ellipse_rotation != 0 and centerpoint[:2] == [0.0, 0.0]:
            for i, pos in enumerate(positions):
                positions[i][:2] = self.rotate_vector(pos[:2], ellipse_rotation)
                
        if ellipse_rotation != 0 and centerpoint[:2] != [0.0, 0.0]:
            raise NotImplementedError()
            
        
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
        """ This function performs a single optimization step, whereas the
        decision variables and parameters that will be used later in the ADMM iteration
        are initialized.
        It lets the vehicles to generate a trajectory from the starting position
        to the goal position, without regarding each other, but avoiding the obstacles.
        The duplicate variables are set to equal their original counterparts,
        the a, b and d_tau values are initialized with the values found in this
        optimization stepd and all lambda values are set to 1.
        The data_exchange functions are reused here to exchange the data between
        the agents.
        
        Besides the below step the following steps have to be taken in the 
        vehicle group:
            - define initialize_x()
            - define initialize_values()
            - update DvX.w0, DvZ.w0, PvX, PvZ before creating the solver in setup_x_update() and setup_y_update()
        """

        "Step 1: trajectory optimization"
        # Initial optimization step (only finding the optimal trajectory,
        # without considering formation)
        for i in range(len(self.vehicles)):
            # Let's also do the initialization stuff here
            
            self.vehicles[i].initialize_x()

        "Step 2: exchanging solution"
        # Exchanging information
        message_container = []
        # Collecting messages
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_x_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_x_receive(message_container)

        "Step 3: initializing decision variables & parameters"
        # Initializing decision variables and parameters.
        for i in range(len(self.vehicles)):
            self.vehicles[i].initialize_values()

        "Step 4: plotting"
        # self.plot_initial_values()

    def prepare(self):
        """ Requests all vehicles to perform the preparation processes for
        creating the necessary variables and solver that are needed for the
        ADMM iteration.
        """

        # Extra step: we initialize decision variables and parameters for faster convergence
        self.initialize_values()

        for i in range(len(self.vehicles)):
            self.vehicles[i].prepare0()
            self.vehicles[i].prepare1()
            self.vehicles[i].prepare2()

        return self
    
    def simulation_step(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].simulation_step()
            
    def set_var(self, var):
        for i in range(len(self.vehicles)):
            if 'n_intermediate_ADMM' in var:
                self.vehicles[i].n_intermediate_ADMM = var['n_intermediate_ADMM']
                self.n_intermediate_ADMM = var['n_intermediate_ADMM']
            if 'stage' in var:
                self.vehicles[i].stage = var['stage']
                self.stage = var['stage']
                
            if 'new_positions' in var:
                self.vehicles[i].vehicle_positions_new['stage'] += [var['new_positions']['stage']]
                if var['new_positions']['vehicle_positions_new'] != []:
                    self.vehicles[i].vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new'][i]]
                else:
                    self.vehicles[i].vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new']]
            if 'new_times' in var:
                self.vehicles[i].vehicle_positions_new['stage'] += [var['new_times']['stage']]
                if var['new_times']['vehicle_times_new'] != []:
                    self.vehicles[i].vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new'][i]]
                else:
                    self.vehicles[i].vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new']]
                    
            if 't_step' in var:
                self.vehicles[i].t_step = var['t_step']
            if 't_window_size' in var:
                self.vehicles[i].t_window_size = var['t_window_size']
            if 't_end' in var:
                self.vehicles[i].t_end = var['t_end']
            if 'knot_intervals' in var:
                self.vehicles[i].knot_intervals = var['knot_intervals']
            if 't_resolution_length' in var:
                self.vehicles[i].t_resolution_length = var['t_resolution_length']
            if 'rho' in var:
                self.vehicles[i].rho = var['rho']
            if 'rho_input' in var:
                self.vehicles[i].rho_input = var['rho_input']
            if 'rho_final_value' in var:
                self.vehicles[i].rho_final_value = var['rho_final_value']
                
            if 'MPC_version' in var:
                self.vehicles[i].MPC_version = var['MPC_version']
                self.MPC_version = var['MPC_version']
            if 'n_of_saved_waypoints' in var:
                self.vehicles[i].n_of_saved_waypoints = var['n_of_saved_waypoints']
                
                
                
                
                
    def set_simulation(self, simulation = False):
        for i in range(len(self.vehicles)):
            self.vehicles[i].simulation = simulation
            self.vehicles[i].shift_enabled = simulation
        
        

    def solve(self):
        
        # self.intermediate_position_generator()
        """
        1) x_update(), which optimizes the trajectory of the given vehicle.
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].x_update()


        """
        2) data_exchange_x(), where these values are shared between agents.
        """
        message_container = []
        # Collecting messages
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_x_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_x_receive(message_container)

        """
        3) z_update(), optimizing the the duplicate variables z and z_ij.
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].z_update()

        """
        4) lambda_update(), which updates the lambda values.
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].lambda_update()

        """
        5) data_exchange_z, where z_i, z_ij, lambda_i, lambda_ij are shared.
        """
        message_container = []
        # Collecting messages
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_z_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_z_receive(message_container)


        return self


    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################






    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    "For plotting stuff"

    def plot_setup(self):
        """This function plots the environment and the group starting and final
        position"""

        fig, ax = plt.subplots()
        self.vehicles[0].plot_environment(ax, 0)
        for obstacle in self.vehicles[0].obstacles:
            obstacle.plot_obstacle(ax)
        for vehicle in self.vehicles:
            x, y = self.fp.frenet_to_inertial(vehicle.xf[0], vehicle.xf[1], 0)
            x0_plot = ax.plot(x, y, 'ko', markersize = 3)
            x, y = self.fp.frenet_to_inertial(vehicle.xf[0], vehicle.xf[1], 1)
            xf_plot =  ax.plot(x, y, 'go', markersize = 3)
        x0_plot[0].set_label("Starting positions")
        xf_plot[0].set_label("Final positions")
        # ax.legend([x0_plot[0], xf_plot[0]], ["Starting positions", "Final positions"])
        ax.legend(fontsize = 'x-small')
            
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_aspect('equal', adjustable='box')
        print("Kezemet, mert csalok!") # x0 helyett is xf


    def plot_initial_values(self):
        """This function plots initial trajectories generated only considering
        collision avoidance with obstacles."""
        flatten = lambda t: [item for sublist in t for item in sublist]
        fig, ax = plt.subplots()
        for i in range(len(self.vehicles)):
            # ax = self.vehicles[i].plot_vehicle_trajectories(ax)
            x, y = self.vehicles[i].astar_initials["y_astar"]
            t = np.linspace(0, 1, 100)
            x_t = [x(t_) for t_ in t]
            y_t = [y(t_) for t_ in t]

            x_t = flatten(x_t)
            y_t = flatten(y_t)
            ax.plot(x_t, y_t, 'b.')

        # Axis related stuff
        ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(-1, "?"))
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_aspect('equal', adjustable='box')
        # Saving figure to folder
        fig.savefig('figures/' + 'astar_stage' + '{:0>1d}'.format(self.stage) +'.png', dpi = 200)
        fig.clear()

        return self



    def plotter(self, iternum : int = 0, seed = ''):
        """This function plots the trajectories calculated by each of the agent.
        It also plots the Frenet path.
        """
        # Plotting parameters for a single vehicle
        # self.l_groups[0].vehicles[0].plot_params(folder)
        # plt.close('all')

        # Plotting trajectories
        # https://stackoverflow.com/questions/34442791/pass-plot-to-function-matplotlib-python

        # Plotting frenet path
        # self.vehicles[0].fp.plot_path(ax, 1) # t = interp(i,[0,N-1],[0,1])

        # Plotting trajectory of the vehicles
        fig, ax = self.figures["figures"]
        for i in range(len(self.vehicles)):
            ax = self.vehicles[i].plot_vehicle_trajectories(ax)

        # Axis related stuff
        ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_aspect('equal', adjustable='box')
        # Saving figure to folder
        fig.savefig(self.cwd + '/figures/' +'{:0>1d}'.format(self.stage) + '{:0>2d}'.format(iternum) +'.png', dpi = 200)
        ax.clear()

        return self

    def plot_vehicle_trajectories_gradient(self):
        fig, ax = self.figures["figures"]
        for i in range(len(self.vehicles)):
            ax = self.vehicles[i].plot_vehicle_trajectories_gradient(ax)

        # Axis related stuff
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_title("Trajectory change over the iterations")
        ax.set_aspect('equal', adjustable='box')
        # Saving figure to folder
        fig.savefig(self.cwd + '/figures/' + 'stage_' + '{:0>1d}'.format(self.stage) + 'trajectory_gradients' +'.pdf')
        ax.clear()

    def check_collision(self, plot_distances = False):
        x_t, y_t = [], []
        for i in range(len(self.vehicles)):
            x_t_tmp, y_t_tmp = self.vehicles[i].get_final_trajectories_t()
            x_t += [x_t_tmp]
            y_t += [y_t_tmp]

        def distance(a, b):
            return np.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )

        r = self.vehicles[0].radious
        collision = [0] * len(x_t[0])
        collision_happened = False
        mean_dists = []
        minimum_dists = []
        for i in range(len(x_t[0])):
            dists = []
            for j in range(len(x_t)):
                for k in range(len(x_t)):
                    if j != k:
                        a = [ x_t[j][i], y_t[j][i] ]
                        b = [ x_t[k][i], y_t[k][i] ]
                        dist = distance(a, b)
                        if dist < r:
                            collision[i] = 1
                        dists += [dist]
            mean_dists += [np.mean(dists)]
            minimum_dists += [np.amin(dists)]
        for i, min_dist in enumerate(minimum_dists):
            if min_dist < r*2 * self.vehicle_avoidnce_multiplier:
                collision_happened = True

        if plot_distances == True:
            plt.figure()
            plt.plot(mean_dists)
            plt.plot(minimum_dists)
            plt.plot([(r*2)] * len(x_t[0]), 'r:')
            plt.plot([(r*2) * self.vehicle_avoidnce_multiplier] * len(x_t[0]), 'g:')
            plt.plot(collision, 'y:')
            plt.savefig('figures/' + 'stage' + '_{:0>1d}'.format(self.stage) + 'mean_and_minimum_distances.png', dpi = 100)

        return collision_happened # , mean_dists
    
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

    def calculate_formation_error_old(self):

        # Getting the formation errors
        formation_error_means = []
        formation_error_summed_over_iter_means = []
        tmp1 = []
        tmp2 = []
        for i in range(len(self.vehicles)):
            formation_error, formation_error_summed_over_iter = self.vehicles[i].calculate_formation_error()
            tmp1 += [formation_error]
            tmp2 += [formation_error_summed_over_iter]

        # Formation error means
        formation_error_means = np.mean(np.array(tmp1), axis=0)
        formation_error_summed_over_iter_means = np.mean(np.array(tmp2), axis=0)

        # New plot
        fig, ax = plt.subplots()
        # Plotting
        from matplotlib.pyplot import cm
        color=cm.rainbow(np.linspace(0,1,len(formation_error_means)))
        [ax.plot(np.linspace(0, 1, 100), formation_error_means[i], c = color[i, :]) for i in range(len(formation_error_means))]

        # Axis realated stuff
        ax.set_xlabel("time t")
        ax.set_ylabel("formation error")
        ax.set_title("Formation error of the group for subsequent iterations")
        # Saving the figure
        fig.savefig(self.cwd + '/figures/' + 'stage' + '_{:0>1d}'.format(self.stage) + 'formation_error_mean.png', dpi = 100)

        # New plot
        fig, ax = plt.subplots()
        # Plotting
        ax.plot(formation_error_summed_over_iter_means)

        # Axis realated stuff
        ax.set_xlabel("iteration k")
        ax.set_ylabel("formation error")
        ax.set_title("Formation error of the group for subsequent iterations")
        # Saving the figure
        fig.savefig(self.cwd + '/figures/' + 'stage' + '_{:0>1d}'.format(self.stage) + 'formation_error_summed_over_iter_mean.png', dpi = 100)



        return self
    
    
    def plot_moovie_frames(self, n_frames, iternum : int = 0, seed = ''):
        fig, ax = self.figures["figures"]
        ax.clear()
        frame_num = 0
        horizon_num = 0
        for t in np.linspace(0, 1, n_frames):

            # Then we plot the vehicles
            for i in range(len(self.vehicles)):
                t_start, ax = self.vehicles[i].plot_moovie_frames_mooving_horizon(ax, horizon_num)
            horizon_num += 1

            # Axis related stuff
            # ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
            # Or setting the ax limits 
            ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
            ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
            ax.set_xlim(self.vehicles[0].fp.fx_spline(t_start)[0][0] - 2, self.vehicles[0].fp.fx_spline(t_start)[0][0] + 2*3)
            ax.set_ylim(self.vehicles[0].fp.fy_spline(t_start)[0][0] - 2.5, self.vehicles[0].fp.fy_spline(t_start)[0][0] + 2.5)
            # ax.set_xlabel("x axis")
            # ax.set_ylabel("y axis")
            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            # Saving figure to folder
            # fig.savefig(self.cwd + '/video/' + '{:0>1d}'.format(self.stage) + '{:0>2d}'.format(frame_num) +'.png', dpi = 200)
            fig.savefig(self.cwd + '/video/' + '{:0>2d}'.format(frame_num) +'.png', dpi = 200)
            ax.clear()
            frame_num += 1
            # print('t_start, fx(t_start)' + str(t_start) + ',' + str(self.vehicles[0].fp.fx_spline(t_start)[0][0]))

        return self
    
    
    
    def plot_moovie_frames_old(self, iternum : int = 0, seed = ''):

        # self.check_collision()
        # self.calculate_formation_error()
        
        "Zoomed-in version"
        fig, ax = self.figures["figures"]

        frame_num = 0
        for t in np.linspace(0, 1, 100):
            # fig.clear()
            # fig, ax = plt.subplots()

            # First we plot the paths
            # for i in range(len(self.vehicles)):
                # ax = self.vehicles[i].plot_path_frames(ax, t)

            # Then we plot the vehicles
            for i in range(len(self.vehicles)):
                ax = self.vehicles[i].plot_moovie_frames(ax, t)

            # Axis related stuff
            # ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
            ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
            ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
            # Or setting the ax limits 
            ax.set_xlim(self.vehicles[0].fp.fx_spline(t)[0][0] - 2*2, self.vehicles[0].fp.fx_spline(t)[0][0] + 2*2)
            ax.set_ylim(self.vehicles[0].fp.fy_spline(t)[0][0] - 1*2, self.vehicles[0].fp.fy_spline(t)[0][0] + 1*2)
            ax.set_xlabel("x axis")
            ax.set_ylabel("y axis")
            ax.set_aspect('equal', adjustable='box')
            
            
            
            plt.axis('off')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            
            
            # Saving figure to folder
            fig.savefig(self.cwd + '/video/' + '{:0>1d}'.format(self.stage) + '{:0>2d}'.format(frame_num) +'.png', dpi = 200)
            ax.clear()
            frame_num += 1
            
        # "Zoomed-out version #entire landscape"
        # fig, ax = self.figures["figures"]

        # frame_num = 0
        # for t in np.linspace(0, 1, 100):
        #     # fig.clear()
        #     # fig, ax = plt.subplots()

        #     # First we plot the paths
        #     # for i in range(len(self.vehicles)):
        #         # ax = self.vehicles[i].plot_path_frames(ax, t)

        #     # Then we plot the vehicles
        #     for i in range(len(self.vehicles)):
        #         ax = self.vehicles[i].plot_moovie_frames(ax, t)

        #     # Axis related stuff
        #     # ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
        #     ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        #     ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        #     # Or setting the ax limits 
        #     # ax.set_xlim(self.vehicles[0].fp.fx_spline(t)[0][0] - 2*2, self.vehicles[0].fp.fx_spline(t)[0][0] + 2*2)
        #     # ax.set_ylim(self.vehicles[0].fp.fy_spline(t)[0][0] - 1*2, self.vehicles[0].fp.fy_spline(t)[0][0] + 1*2)
        #     ax.set_xlabel("x axis")
        #     ax.set_ylabel("y axis")
        #     ax.set_aspect('equal', adjustable='box')
            
            
            
        #     plt.axis('off')
        #     ax.axes.xaxis.set_visible(False)
        #     ax.axes.yaxis.set_visible(False)
            
        #     # Saving figure to folder
        #     fig.savefig(self.cwd + '/video/' + 'entire_{:0>1d}'.format(self.stage) + '{:0>2d}'.format(frame_num) +'.pdf', dpi = 200)
        #     ax.clear()
        #     frame_num += 1
        

        return self

    def save_trajectory_to_csv(self, n_steps, t_desired = 1, t_hover = 0.1):
        # for horizon_num in range(n_steps):
        #         # self.vehicles[0].save_trajectory_to_csv(horizon_num, t_desired = t_desired, t_hover = t_hover)
        #     for i in range(len(self.vehicles)):
        #         self.vehicles[i].save_trajectory_to_csv(horizon_num, t_desired = t_desired, t_hover = t_hover)
                
                
        for i in range(len(self.vehicles)):
            self.vehicles[i].save_trajectory_to_csv_SINGLE(n_steps, t_desired = t_desired, t_hover = t_hover)
        return self

    def frenet_plotter(self, iternum : int = 0, seed = ''):
        """This function plots the trajectories calculated by each of the agent.
        It also plots the Frenet path.
        """
        
        # Plotting trajectory of the vehicles
        fig, ax = self.figures["figures"] # plt.subplots()
        for i in range(len(self.vehicles)):
            ax = self.vehicles[i].plot_vehicle_frenet_trajectories(ax)
        
        try:
            if self.vehicle_positions_new != []:
                for position in self.vehicle_positions_new:
                    x, y = self.fp.frenet_to_inertial(position[0], position[1], self.vehicles[0].t_end + self.vehicles[0].t_step)
                    ax.plot(x, y, 'ro', markersize = 1)
            else:
                for vehicle in self.vehicles:
                    position = vehicle.xf[0:2]
                    x, y = self.fp.frenet_to_inertial(position[0], position[1], self.vehicles[0].t_end + self.vehicles[0].t_step)
                    ax.plot(x, y, 'go', markersize = 1)
        except:
            pass
            
        # Axis related stuff
        ax.set_title("Trajectories of the vehicles after iteration {} with seed {} in the frenet frame".format(iternum, seed))
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.1, 0)
        ax.set_xlabel("x axis")  
        ax.set_ylabel("y axis") 
        ax.set_aspect('equal', adjustable='box')
        # Saving figure to folder
        # fig.savefig(self.cwd + '/figures/' +'{:0>1d}'.format(self.stage) + '{:0>2d}'.format(iternum) +'.png', dpi = 200)
        fig.savefig(self.cwd + '/figures/' + '{:0>2d}'.format(iternum) +'.png', dpi = 200)
        ax.cla()
        # plt.show()
        # fig.clf()
        
        # fig.clear()
        return self