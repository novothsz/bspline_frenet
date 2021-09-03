from .vehicle import Vehicle
import numpy as np
import math
import matplotlib.pyplot as plt
from .frenet_path import FrenetPath
from .environment import Environment
import yaml

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
        
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
        
    def intermediate_position_generator(self):
        # get current final time
        # get obstacle positions at final time
        # get position of vehicles at the final time of the previously calculated horizon
        # rotate this formation with incremental +/- phi angles till no collision
        # if not found, shrink/blow up till no collision
        # if - found: set this as final position & phi for the vehicles
        # if - not found: all hope is lost, we throw in the towel
        
        # Step 1: get the current final time
        t_end = self.vehicles[0].t_end + self.vehicles[0].t_step #... well, maybe  - self.vehicles[0].t_step? Depends on when we call this method
        # Step 2: get obstacle positions at final time
        obstacle_corners = []
        for obstacle in self.vehicles[0].obstacles:
            corners = [ [corner[0](t_end).tolist()[0][0], corner[1](t_end).tolist()[0][0]] for corner in obstacle.corners_spline]
            obstacle_corners += [corners]
        # Step 3: get position of vehicles at the final time of the previously calculated horizon
        # t_previous = self.vehicles[0].t_end - self.vehicles[0].t_step
        vehicle_positions = []
        for vehicle in self.vehicles:
            idx_len = int(len(vehicle.DvX.y) / vehicle.n_dimensions)
            # try:
            #     vehicle_positions += [[vehicle.DvX.y[idx_len-1].full()[0][0], vehicle.DvX.y[idx_len*2 - 1].full()[0][0]]] # final predicted p and q positions
            # except:
            #     vehicle_positions += [[vehicle.DvX.y[idx_len-1], vehicle.DvX.y[idx_len*2 - 1]]] # final predicted p and q positions
                
                
            vehicle_positions += [vehicle.xf[:2]]
                
                
        # Step 4: check for collision
        all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions, obstacle_corners)   
        if collision == True:
            kappa = True
            
        # Step 5: rotating & scaling
        rotation_angle_new = 0
        scaling_factor_new = 1
        vehicle_positions_new = []
        costs = []
        vehicle_positions_new_saved = []
        if collision == True:
            degree_step = 10
            radian_step = degree_step/360*2 * math.pi
            rotation_angles = [[0 + radian_step * i, 0 - radian_step * i] for i in range(1, int(math.pi/radian_step))]
            rotation_angles = np.array(rotation_angles).reshape(-1).tolist()
            
            scaling_step = 1.1
            scaling_step = 2.0
            # scaling_factors = [  [1 * scaling_step ** i, 1 / (scaling_step ** i) ] for i in range(0, math.floor(abs(math.log(0.010) / math.log(scaling_step))))  ]
            # scaling_factors = np.array(scaling_factors).reshape(-1).tolist()
            scaling_factors_shrink = [  1 / (scaling_step ** i)  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
            scaling_factors_expand = [  1 * scaling_step ** i  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
            scaling_factors = scaling_factors_shrink + scaling_factors_expand
            scaling_factors = np.array(scaling_factors).reshape(-1).tolist()
            
            
            # Version 1: take the first non-colliding result
            
            # for scaling_factor in scaling_factors:
            #     vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
            #     for rotation_angle in rotation_angles:
            #         vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
            #         all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, obstacle_corners)
            #         if collision == False:
            #             vehicle_positions_new = vehicle_positions_scaled_rotated
            #             rotation_angle_new = rotation_angle
            #             break
            #     if collision == False:
            #         scaling_factor_new = scaling_factor
            #         break
            
            # if collision == True:
            #     import warnings
            #     warnings.warn("Warning...........Collision is still happening :/ Nothing we can do... Formation dissolve!")
            
            # Version 2: give costs and choose the least-cost formation change
            
            for scaling_factor in scaling_factors:
                vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
                for rotation_angle in rotation_angles:
                    vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
                    all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, obstacle_corners)
                    if collision == True:
                        costs += [math.inf]
                    else:
                        cost = self.formation_change_cost_calculator(vehicle_positions, vehicle_positions_scaled_rotated, rotation_angle, scaling_factor)
                        # check collision for previous and future steps too!
                        obstacle_corners_zizz = []
                        for t_zizz in np.linspace(t_end - self.vehicles[0].t_step / 10 * 10 , t_end + self.vehicles[0].t_step / 10 * 10, 6):
                        
                            for obstacle in self.vehicles[0].obstacles:
                                corners = [ [corner[0](t_zizz).tolist()[0][0], corner[1](t_zizz).tolist()[0][0]] for corner in obstacle.corners_spline]
                                obstacle_corners_zizz += [corners]
            
                        all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, obstacle_corners_zizz)   
                        if collision == True:
                            cost = math.inf
            
                        costs += [cost]
                    vehicle_positions_new_saved += [vehicle_positions_scaled_rotated]
                    
            # find the least-cost version
            cost_min = min(costs)
            cost_min_idx = costs.index(cost_min)
            vehicle_positions_new = vehicle_positions_new_saved[cost_min_idx]
            
            
            
        # Step 6: setting new positions
        self.vehicle_positions_new = vehicle_positions_new
        if vehicle_positions_new != []:
            for i, position in enumerate(vehicle_positions_new):
                self.vehicles[i].xf = [position[0], position[1]] + [self.vehicles[i].xf[2] + rotation_angle_new] + self.vehicles[i].xf[3:]
        if rotation_angle_new != 0:        
            print("rotation_angle_new = " + str(rotation_angle_new))
        if scaling_factor_new != 1:
            print("scaling_factor_new = " + str(scaling_factor_new))
                
        return self
            
          
     
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    def formation_change_cost_calculator(self, vehicle_positions_original, vehicle_positions_new, rotation_angle = 0, scaling_factor = 1):
        cost = 0
        alpha_distance = 0.1
        alpha_rotation = 0.0001
        alpha_scaling_up = 1000
        alpha_scaling_down = 1
        for original, new in zip(vehicle_positions_original, vehicle_positions_new):
            cost += alpha_distance * (original[0] - new[0])**2 + (original[1] - new[1])**2
            cost += alpha_rotation * abs(rotation_angle)
            # cost += alpha_scaling * abs(1-scaling_factor)
            if scaling_factor > 1:
                cost += alpha_scaling_up * abs(1-scaling_factor)
            if scaling_factor < 1:
                cost += alpha_scaling_down * abs(1-scaling_factor)
                
            # cost += alpha_scaling_up * scaling_factor * abs(1-scaling_factor)
            
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
                                               ellipse_rotation = math.pi / 2)
            
            
    # def ellipse_generator(self, centerpoint : list, n_positions : int, a : float, b : float, 
    #                       ellipse_rotation : float = 0, vehicles_rotation : float = 0,
    #                       ellipse_scale_x : float = 1, ellipse_scale_y : float = 1):
        
        
            if self.stage == 0:
                yaml_dict = {'crazyflies' : []}
                for i, vehicle in enumerate(self.vehicles):
                    # initialPosition = [positions[i][j].tolist() for j in range(len(positions[i]))] + [targetHeight]
                    initialPosition = [float(pos) for pos in positions[i]]
                    initialPosition = initialPosition + [float(targetHeight)]
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
                                               ellipse_rotation = math.pi / 2)

            if self.stage == 0:
                yaml_dict = {'crazyflies' : []}
                for i, vehicle in enumerate(self.vehicles):
                    # initialPosition = [positions[i][j].tolist() for j in range(len(positions[i]))] + [targetHeight]
                    initialPosition = [float(pos) for pos in positions[i]]
                    initialPosition = initialPosition + [float(targetHeight)]
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
    def ellipse_generator(self, centerpoint : list, n_positions : int, a : float, b : float, 
                          ellipse_rotation : float = 0, vehicles_rotation : float = 0,
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
        self.vehicles[0].plot_environment(ax)
        for obstacle in self.vehicles[0].obstacles:
            obstacle.plot_obstacle(ax)
        for vehicle in self.vehicles:
            ax.plot(vehicle.x0[0], vehicle.x0[1], 'ko')
            ax.plot(vehicle.xf[0], vehicle.xf[1], 'go')


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
    
    
    def plot_moovie_frames(self, iternum : int = 0, seed = ''):
        fig, ax = self.figures["figures"]
        ax.clear()
        frame_num = 0
        horizon_num = 0
        for t in np.linspace(0, 1, 80):

            # Then we plot the vehicles
            for i in range(len(self.vehicles)):
                t_start, ax = self.vehicles[i].plot_moovie_frames_mooving_horizon(ax, horizon_num)
            horizon_num += 1

            # Axis related stuff
            ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
            # Or setting the ax limits 
            ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
            ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
            # ax.set_xlim(self.vehicles[0].fp.fx_spline(t_start)[0][0] - 2, self.vehicles[0].fp.fx_spline(t_start)[0][0] + 2*2)
            # ax.set_ylim(self.vehicles[0].fp.fy_spline(t_start)[0][0] - 2, self.vehicles[0].fp.fy_spline(t_start)[0][0] + 2)
            ax.set_xlabel("x axis")
            ax.set_ylabel("y axis")
            ax.set_aspect('equal', adjustable='box')
            # Saving figure to folder
            fig.savefig(self.cwd + '/video/' + '{:0>1d}'.format(self.stage) + '{:0>2d}'.format(frame_num) +'.png', dpi = 200)
            # »
            frame_num += 1

        return self
    
    
    
    def plot_moovie_frames_old(self, iternum : int = 0, seed = ''):

        # self.check_collision()
        # self.calculate_formation_error()
        fig, ax = self.figures["figures"]

        frame_num = 0
        for t in np.linspace(0, 1, 20):
            # fig.clear()
            # fig, ax = plt.subplots()

            # First we plot the paths
            # for i in range(len(self.vehicles)):
                # ax = self.vehicles[i].plot_path_frames(ax, t)

            # Then we plot the vehicles
            for i in range(len(self.vehicles)):
                ax = self.vehicles[i].plot_moovie_frames(ax, t)

            # Axis related stuff
            ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
            # ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
            # ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
            # Or setting the ax limits 
            ax.set_xlim(self.vehicles[0].fp.fx_spline(t)[0][0] - 2*2, self.vehicles[0].fp.fx_spline(t)[0][0] + 2*2)
            ax.set_ylim(self.vehicles[0].fp.fy_spline(t)[0][0] - 1*2, self.vehicles[0].fp.fy_spline(t)[0][0] + 1*2)
            ax.set_xlabel("x axis")
            ax.set_ylabel("y axis")
            ax.set_aspect('equal', adjustable='box')
            # Saving figure to folder
            fig.savefig(self.cwd + '/video/' + '{:0>1d}'.format(self.stage) + '{:0>2d}'.format(frame_num) +'.png', dpi = 100)
            ax.clear()
            frame_num += 1

        return self

    def save_trajectory_to_csv(self, t_desired = 5, t_hover = 2):
        for i in range(len(self.vehicles)):
            self.vehicles[i].save_trajectory_to_csv(t_desired = t_desired, t_hover = t_hover)
        return self

    def frenet_plotter(self, iternum : int = 0, seed = ''):
        """This function plots the trajectories calculated by each of the agent.
        It also plots the Frenet path.
        """
        
        # Plotting trajectory of the vehicles
        fig, ax = self.figures["figures"] # plt.subplots()
        for i in range(len(self.vehicles)):
            ax = self.vehicles[i].plot_vehicle_frenet_trajectories(ax)
        
        if self.vehicle_positions_new != []:
            for position in self.vehicle_positions_new:
                x, y = self.fp.frenet_to_inertial(position[0], position[1], self.vehicles[0].t_end + self.vehicles[0].t_step)
                ax.plot(x, y, 'ro', markersize = 1)
        else:
            for vehicle in self.vehicles:
                position = vehicle.xf[0:2]
                x, y = self.fp.frenet_to_inertial(position[0], position[1], self.vehicles[0].t_end + self.vehicles[0].t_step)
                ax.plot(x, y, 'go', markersize = 1)
            
            
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
        fig.savefig(self.cwd + '/figures/' +'{:0>1d}'.format(self.stage) + '{:0>2d}'.format(iternum) +'.png', dpi = 200)
        ax.cla()
        # fig.clf()
        
        # fig.clear()
        return self