import numpy as np
from numpy import interp
import math
from time import time
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from casadi import MX, SX, Function, vertcat, nlpsol, cos, sin, norm_2 #, dot
from casadi import dot
from .spline import BSpline, BSplineBasis
from .spline_extra import definite_integral


from .vehicle_basis import VehicleBasis
from .param import ParamValX, ParamValZ, DecisionVarX, DecisionVarZ
import time

from matplotlib.collections import LineCollection
from matplotlib.pyplot import cm
import copy

class Vehicle(VehicleBasis):
    def __init__(self):
        super().__init__()
        
        self.plot_self_ID = 2
        self.plot_obst_IDX = 0 # 4
        self.plot_stage = 5
        
        self.t_start = 0.0
        self.t_step = 0.04 # 0.04 # Changed in simulation_step() upon first call
        self.t_window_size = 0.2
        self.t_end = self.t_start + self.t_window_size
        self.simulation = False
        self.shift_enabled = False
        
        self.n_of_saved_waypoints = int(self.t_window_size / self.t_step) + 1; epsilon = 10e-10; assert self.t_window_size % self.t_step > -epsilon and \
                                                                                                        self.t_window_size % self.t_step < epsilon# math.floor(self.t_window_size / self.t_step)
        
        self.n_of_saved_waypoints = 5
        self.waypoints = []
        self.waypoint_timestamps = []
        self.current_configuration_position = []
        

    def setup_z_update(self):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []
        


        # Define ADMM cost
        # Cost: x_i - z_i
        y = self.define_MX_spline(degree=self.state_degree, knot_intervals=self.knot_intervals, n_spl=self.n_dimensions,
                                lower_bound=self.y_min, upper_bound=self.y_max,
                                name=["y"] * self.n_dimensions,
                                category = 'parameter')

        z_i = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound=self.y_min, upper_bound=self.y_max,
                               initial_value = [[self.x0[i], self.xf[i]] for i in range(self.n_dimensions)],
                               name = ['z_i'] * self.n_dimensions)

        lambda_i  = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound = [], upper_bound = [],
                               name = ['lambda_i'] * self.n_dimensions,
                               category = 'parameter')
        
        # Group center should be in the (0, 0) position of the frenet frame.
        # The mean of our and the group-members position (coming later) should be (0, 0)
        p_sum = 0
        q_sum = 0
        p_sum += z_i[0]
        q_sum += z_i[1]
        
        
        
        for i in range(len(y)):
            self.J += dot(lambda_i[i].coeffs,  y[i].coeffs - z_i[i].coeffs)
            self.J += self.rho * dot(np.ones(y[i].coeffs.shape[0]), (y[i].coeffs - z_i[i].coeffs)**2)

        # Cost sum: x_j - z_ij
        for i in range(len(self.neighbours)):

            y_j = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound = [], upper_bound = [],
                               name = ['y_j'] * self.n_dimensions,
                               category = 'parameter')

            z_ij = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                                   lower_bound=self.y_min, upper_bound=self.y_max,
                                   initial_value = [[self.neighbours[i].x0[j], self.neighbours[i].xf[j]] for j in range(self.n_dimensions)],
                                   name = ['z_ij'] * self.n_dimensions)
            lambda_ij  = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                                   lower_bound = [], upper_bound = [],
                                   name = ['lambda_ij'] * self.n_dimensions,
                                   category = 'parameter')
            
            
            for j in range(len(y)):
                self.J += dot(lambda_ij[j].coeffs,  y_j[j].coeffs - z_ij[j].coeffs)
                self.J += self.rho * dot(np.ones(y_j[j].coeffs.shape[0]), (y_j[j].coeffs - z_ij[j].coeffs)**2)


            def cross_product(spline1, spline2):
                a1, a2 = spline1[0], spline1[1]
                b1, b2 = spline2[0], spline2[1]
                return a1 * b2 - a2 * b1
            
            def dot_product(spline1, spline2):
                a1, a2 = spline1[0], spline1[1]
                b1, b2 = spline2[0], spline2[1]
                return (a1 * b1 + a2 * b2)
            
            def vector_rotation(spline, alpha, t):
                a1, a2 = spline[0], spline[1]
                return (a1 * cos(alpha(t)) - a2 * sin(alpha(t)), \
                        a1 * sin(alpha(t)) + a2 * cos(alpha(t)))
                    
            # vec1: what is should be
            # vec2: what we have
            # cross: the cross product of the two vectors. It is a function of t.
            vec1 = z_i - z_ij
            vec2 = np.array(self.xf[:self.n_dimensions_old]) - np.array(self.neighbours[i].xf[:self.n_dimensions_old])
            
            
            # Formation constraint.
            for t in np.linspace(0, 1, self.t_resolution_length):
                self.define_constraint([cross_product(vec1, vector_rotation(vec2, z_i[2], t))(t)],
                                        [-self.slack * 1],
                                        [self.slack * 1],
                                        constraint_type='time',
                                        name=["formation_vehicle_" + str(i)] * self.n_dimensions_old)
                
            # New Phi constraint
            self.define_constraint([z_i[2] - z_ij[2]],
                                    [0],
                                    [0],
                                    constraint_type='overall',
                                    name=["phi_equality_constraint"])
            

            # Group center should be in the (0, 0) position of the frenet frame.
            p_sum += z_ij[0]
            q_sum += z_ij[1]
                
        # Spline-coeff version
        self.define_constraint([p_sum, q_sum],
                                [-self.slack,-self.slack],
                                [ self.slack, self.slack],
                                constraint_type='overall',
                                name=["frenet_zero_" + str(i)] * self.n_dimensions_old)
        


        # Containers
        self.DvZ = DecisionVarZ(self.w_list, self.g_list, self.lbg, self.ubg)
        self.PvZ = ParamValZ(self.P_list, self.P0)

        # Initializing values
        self.DvZ.extract(self.w0)

        "Initialize values with the pre-calculated values from initialize_x()"
        if self.initial_values != {}:
            self.DvZ.w0_z = self.initial_values["z_i"] + self.initial_values["z_ji"]
            self.DvZ.z_i = self.initial_values["z_i"]
            self.DvZ.z_ij = self.initial_values["z_ji"]
            self.PvZ.y = self.initial_values["y"]
            self.PvZ.y_j = self.initial_values["y_j"]
            self.PvZ.lambda_i = self.initial_values["lambda_i"]
            self.PvZ.lambda_ij = self.initial_values["lambda_ij"]
            self.message_in["y_j"] = self.initial_values["y_j"]
            self.message_in["z_ji"] = self.initial_values["z_ji"]
            self.message_in["lambda_ji"] = self.initial_values["lambda_ji"]
        
        # Creating solver class
        prob = {'f': self.J,
                'x': vertcat(*self.w),
                'g': vertcat(*self.g),
                'p': vertcat(*self.P)
                }

        self.solver_z = nlpsol('solver', 'ipopt', prob, self.options_z)

        # Assembling the argument dictionary
        self.arg_z = {'x0' : self.w0,
                   'lbx': self.lbw,
                   'ubx': self.ubw,
                   'lbg': self.lbg,
                   'ubg': self.ubg,
                   'p': self.PvZ.assemble()}

        return self
    
    def z_update_prior(self):
        self.update_PvZ()
        if self.shift_enabled == True:
            self.shift_DvZ()
            self.shift_PvZ()
        # Updating the necessary arguments for the solver
        self.arg_z['x0'] = self.DvZ.assemble()
        self.arg_z['p'] = self.PvZ.assemble()
        return self
    def z_update(self):
        start_time = time.time()
        self.solution_z = self.solver_z.call(self.arg_z)
        final_time = time.time()
        self.variable_history["z_update_time"] += [final_time - start_time]
        return self
    def z_update_posterior(self):
        # Extracting the solution
        self.DvZ.extract(self.solution_z)
        return self
    
    def check_feasibility_of_solution_x(self):
        "Reconstructing all the equations, constraints and checking where the indeasibility happens"
        PvX = copy.deepcopy(self.PvX)
        DvX = copy.deepcopy(self.DvX)
        
        feasibility_dict = {}
        
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        # p, q, phi
        coeffs = [DvX.y[len(basis)*i:len(basis)*(i+1)] for i in range(self.n_dimensions)]
        y = [BSpline(basis, np.array(coeffs_)) for coeffs_ in coeffs]
        y_dot = [y_.derivative() for y_ in y]
        
        p, q, phi = y[0], y[1], y[2]
        # Check initial constraint
        x0 = PvX.x0
        
        feasibility_dict = {}
        key, value = self.check_constraint(y,
                                x0[:self.n_dimensions],
                                x0[:self.n_dimensions],
                                constraint_type='initial',
                                name="y0")
        feasibility_dict[key] = value
        
        key, value = self.check_constraint(y_dot,
                                x0[self.n_dimensions:self.n_dimensions*2],
                                x0[self.n_dimensions:self.n_dimensions*2],
                                constraint_type='initial',
                                name="dy0")
        feasibility_dict[key] = value
        
        if self.MPC_version == False:
            raise NotImplementedError()
        elif self.MPC_version == True:
            raise NotImplementedError()
        elif self.MPC_version == 'MPC_param':
            n = self.n_of_saved_waypoints
            for i, t_intermediate in enumerate(PvX.t_intermediate):
                idx = np.arange(int(self.state_len/2)*i,int(self.state_len/2)*i+int(self.state_len/2))
                x_intermediate = np.array(PvX.x_intermediate)[idx].reshape(-1).tolist()
                
                key, value = self.check_constraint([(p(t_intermediate) - x_intermediate[0])**2, (q(t_intermediate) - x_intermediate[1])**2],
                                        [0, 0],
                                        [(self.radious*1)**2, (self.radious*1)**2],
                                        constraint_type='time',
                                        name="pq_intermediate" + str(i))
                feasibility_dict[key] = value
        
                key, value = self.check_constraint([(phi(t_intermediate) - x_intermediate[2])**2],
                                        [0], # self.slack, # 
                                        [5 / 360 * math.pi * 2],
                                        constraint_type='time',
                                        name="phi_intermediate" + str(i))
                feasibility_dict[key] = value
                
                
                
        # Vertical speed should be zero
        key, value = self.check_constraint([y_dot[1]],
                               [0],
                               [0],
                               constraint_type='final',
                               name="q_dot_final" + str(i))
        feasibility_dict[key] = value
        
        # a_list = []
        # Collision avoidance with obstacles (coefficient based)
        if self.MPC_version == 'MPC_param':
            # We have a different collision-avoidance constraint if we are using the MPC_param version.
            for i, obstacle in enumerate(self.obstacles):
                # corner1, 2, 3, 4
                basis = self.define_knots(degree = self.n_obstacle_cropped_degree, knot_intervals = self.n_obstacle_cropped_knot_intervals)
                search_length = len(basis) * 2 * 4 # 2, because x, y, 'splines' and 4, because each has 4 corners
                idx_current_obst = np.arange(search_length*i,search_length*i+search_length)
                obst_corners = []
                for j in range(4):
                    search_length = len(basis) * 2 # because x, y 'splines'
                    idx_current_corner = np.arange(search_length*j,search_length*j+search_length)
                    obst_corner_coeffs = np.array(PvX.obst)[idx_current_obst][idx_current_corner].reshape(-1).tolist()
                    coeffs = [obst_corner_coeffs[len(basis) * k :len(basis)*(k+1)] for k in range(2)]
                    obst_corners += [[BSpline(basis, np.array(coeffs_)) for coeffs_ in coeffs]]
                
                # hyperplane a
                basis = self.define_knots(degree = 3, knot_intervals = self.knot_intervals)
                search_length = len(basis) * (2)
                idx = np.arange(search_length*i,search_length*i+search_length)
                a_coeffs = np.array(DvX.a)[idx].reshape(-1).tolist()
                a_coeffs = [a_coeffs[0:len(basis)]] + [a_coeffs[len(basis):len(basis)*2]]
                a = [BSpline(basis, np.array(coeffs_)) for coeffs_ in a_coeffs]
                # hyperplane b
                search_length = len(basis) * (1)
                idx = np.arange(search_length*i,search_length*i+search_length)
                b_coeffs = np.array(DvX.b)[idx].reshape(-1).tolist()
                b = [BSpline(basis, np.array(b_coeffs))]
                # hyperplane d_tau
                search_length = len(basis) * (1)
                idx = np.arange(search_length*i,search_length*i+search_length)
                d_tau_coeffs = np.array(DvX.d_tau)[idx].reshape(-1).tolist()
                d_tau = [BSpline(basis, np.array(d_tau_coeffs))]
                hyperplane = [a, b, d_tau]
                
                # obst center
                search_length = len(basis) * 2
                idx_center = np.arange(search_length*i,search_length*i+search_length)
                obst_center_coeffs = np.array(PvX.obst_center)[idx_center].reshape(-1).tolist()
                coeffs = [obst_center_coeffs[len(basis) * k :len(basis)*(k+1)] for k in range(2)]
                obst_center = [BSpline(basis, np.array(coeffs_)) for coeffs_ in coeffs]
                center_circle = [obst_center, self.obstacles[i].max_dist_from_center]
                key, value = self.check_collision_avoidance_hyperplane([p, q], obst_corners, hyperplane,
                                                    radious=self.radious * 0, name="avoidance_obst_" + str(i) + '_',
                                                    constraint_type='obstacle',
                                                    center_circle = center_circle)
                feasibility_dict[key] = value
                # a_list += [tmp_a]
        
        return feasibility_dict
    
    
    def setup_x_update(self):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []
        # Parameters
        x0 = MX.sym('x0', self.state_len); self.P += [x0]; self.P_list += ['x0'] * self.state_len; self.P0 += [0] * self.state_len
        xf = MX.sym('xf', self.state_len); self.P += [xf]; self.P_list += ['xf'] * self.state_len; self.P0 += [0] * self.state_len

        pq = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=self.n_dimensions,
                                lower_bound=self.y_min, upper_bound=self.y_max,
                                initial_value = [[self.x0[i], self.xf[i]] for i in range(self.n_dimensions)],
                                name=["y"] * self.n_dimensions)

        y = pq # we basically rename the thing :)
        pq_dot = [pq_.derivative() for pq_ in pq]
        y_dot = pq_dot
        pq_dotdot = [pq_dot_.derivative() for pq_dot_ in pq]

        p = pq[0]
        q = pq[1]
        phi = pq[2]

        t_intermediate_idx = [] # In this list we save the indices of t_intermediate casadi variables, which are created in one loop 
        # (for way-points) but which we also want to use in another loop (when we specify the normal vector of the hyperplanes)
        # The indices are w.r.t to the self.P list.
        
        # Initial position constraint on y
        self.define_constraint(y,
                                x0[:self.n_dimensions],
                                x0[:self.n_dimensions],
                                constraint_type='initial_param',
                                name=["y0"] * self.n_dimensions)
        # Initial velocity constraint on dy
        self.define_constraint(y_dot,
                                x0[self.n_dimensions:self.n_dimensions*2],
                                x0[self.n_dimensions:self.n_dimensions*2],
                                constraint_type='initial_param',
                                name=["dy0"] * self.n_dimensions)
        
        "state suggestion"
        if self.MPC_version == False:
            for i, t_intermediate in enumerate(self.t_intermediate_list):
                idx = np.arange(int(self.state_len/2)*i,int(self.state_len/2)*i+int(self.state_len/2))
                x_intermediate = np.array(self.x_intermediate_list)[idx].tolist()
                
                
                self.define_constraint([p(t_intermediate) - x_intermediate[0],
                                        q(t_intermediate) - x_intermediate[1],
                                        phi(t_intermediate) - x_intermediate[2]],
                                        [0.0 - self.radious * 10, 0.0 - self.radious * 10, 0.0 - 0.1],
                                        [0.0 + self.radious * 10, 0.0 + self.radious * 10, 0.0 + 0.1],
                                        constraint_type='time',
                                        name=["guidence" + str(i)] * 1)
                if t_intermediate == 1:
                    self.define_constraint([p(t_intermediate) - x_intermediate[0],
                                            q(t_intermediate) - x_intermediate[1],
                                            phi(t_intermediate) - x_intermediate[2]],
                                            [0.0 - self.radious * 10, 0.0 - self.radious * 10, 0.0 - 0.1],
                                            [0.0 + self.radious * 10, 0.0 + self.radious * 10, 0.0 + 0.1],
                                            constraint_type='time',
                                            name=["guidence" + str(i)] * 1)
                    
        elif self.MPC_version == True:
            # Non-forgetting version       
            n = self.n_of_saved_waypoints
            assert n == len(self.t_intermediate_list) # Please generate the intermediate points first :)
            for i, t_intermediate in enumerate(self.t_intermediate_list):
                idx = np.arange(int(self.state_len/2)*i,int(self.state_len/2)*i+int(self.state_len/2))
                
                # Define the parameter
                x_intermediate = MX.sym('x_intermediate', int(self.state_len/2)); self.P += [x_intermediate]; self.P_list += ['x_intermediate'] * int(self.state_len/2); self.P0 += np.array(self.x_intermediate_list)[idx].tolist(); assert len(self.x_intermediate_list)/(self.state_len/2) == n  # [0] * int(self.state_len/2)
                lambda_ = np.power(np.linspace(1, 0, n), 1)
                self.J += self.rho_intermediate * lambda_[i] *(p(t_intermediate) - x_intermediate[0])**2
                self.J += self.rho_intermediate * lambda_[i] *(q(t_intermediate) - x_intermediate[1])**2
                self.J += self.rho_intermediate * lambda_[i] *(phi(t_intermediate) - x_intermediate[2])**2
                
            # Final state constraint
            self.define_constraint([p, q],
                                    xf[:self.n_dimensions_old] - [self.radious*1, self.radious*1],
                                    xf[:self.n_dimensions_old] + [self.radious*1, self.radious*1],
                                    constraint_type='final_param',
                                    name=["yf"] * self.n_dimensions)
            self.define_constraint([phi],
                                    xf[self.n_dimensions] - [5 / 360 * math.pi * 2], # self.slack, # 
                                    xf[self.n_dimensions] + [5 / 360 * math.pi * 2], # self.slack, # 
                                    constraint_type='final_param',
                                    name=["phif"] * self.n_dimensions)
                                            
            
        elif self.MPC_version == 'MPC_param':
            # Pretty much the same as the regular MPC version, but now we pass in t_intermediate as parameter and have hard-constraints, instead of cost function
            n = self.n_of_saved_waypoints
            assert n == len(self.t_intermediate_list) # Please generate the intermediate points first :)
            for i, t_intermediate in enumerate(self.t_intermediate_list):
                idx = np.arange(int(self.state_len/2)*i,int(self.state_len/2)*i+int(self.state_len/2))
                # Define the parameter
                x_intermediate = MX.sym('x_intermediate', int(self.state_len/2)); self.P += [x_intermediate]; self.P_list += ['x_intermediate'] * int(self.state_len/2);

                self.P0 += np.array(self.x_intermediate_list)[:2].tolist()
                self.P0 += [np.cos(self.x_intermediate_list[2]).tolist()]
                self.P0 += [np.sin(self.x_intermediate_list[2]).tolist()]
                t_intermediate = MX.sym('t_intermediate', 1); self.P += [t_intermediate]; self.P_list += ['t_intermediate'] * 1; self.P0 += [self.t_intermediate_list[i]]
                t_intermediate_idx += [len(self.P) - 1]
                                                
                # constraint on pq at t_intermediate
                self.define_constraint([(p(t_intermediate) - x_intermediate[0])**2, (q(t_intermediate) - x_intermediate[1])**2],
                                        [0, 0],
                                        [(self.radious*2)**2, (self.radious*2)**2],
                                        constraint_type='time',
                                        name=["pq_intermediate" + str(i)] * 2)
                
                # constraint on phi at t_intermediate
                self.define_constraint([(phi(t_intermediate) - x_intermediate[2])**2],
                                        [0], # self.slack, # 
                                        [5 / 360 * math.pi * 2],
                                        constraint_type='time',
                                        name=["phi_intermediate" + str(i)] * 1)
                
        else:
            raise NotImplementedError()
            
        self.J += (p(1) - x_intermediate[0])**2 + (q(1) - x_intermediate[1])**2  
        a_list = []
        # Collision avoidance with obstacles (coefficient based)
        if self.MPC_version == 'MPC_param':
            # We have a different collision-avoidance constraint if we are using the MPC_param version.
            for i, obstacle in enumerate(self.obstacles):
                obst_corners = []
                deg = self.n_obstacle_cropped_degree
                corner1 = self.define_MX_spline(degree = deg, knot_intervals = self.n_obstacle_cropped_knot_intervals, n_spl = 2,
                           lower_bound = [], upper_bound = [],
                           name = ['obst'] * 2,
                           category = 'parameter')
                
                corner2 = self.define_MX_spline(degree = deg, knot_intervals = self.n_obstacle_cropped_knot_intervals, n_spl = 2,
                           lower_bound = [], upper_bound = [],
                           name = ['obst'] * 2,
                           category = 'parameter')
                
                corner3 = self.define_MX_spline(degree = deg, knot_intervals = self.n_obstacle_cropped_knot_intervals, n_spl = 2,
                           lower_bound = [], upper_bound = [],
                           name = ['obst'] * 2,
                           category = 'parameter')
                
                corner4 = self.define_MX_spline(degree = deg, knot_intervals = self.n_obstacle_cropped_knot_intervals, n_spl = 2,
                           lower_bound = [], upper_bound = [],
                           name = ['obst'] * 2,
                           category = 'parameter')
                
                obst_corners += [corner1, corner2, corner3, corner4]
                
                # Let's put a circle inside the obstacle as well. The same hyperplane must be between this circle and the vehicle, 
                # that also separates the vehicle and the obstacle as well.
                center = self.define_MX_spline(degree = deg, knot_intervals = self.n_obstacle_cropped_knot_intervals, n_spl = 2,
                           lower_bound = [], upper_bound = [],
                           name = ['obst_center'] * 2,
                           category = 'parameter')
                # This circle also has a radious, defined by the half of the maximum extension of the obstacle
                circle_rad = obstacle.max_dist_from_center * 0.8
                center_circle = [center, circle_rad]
            
                tmp_a = self.collision_avoidance_hyperplane([p, q], obst_corners,
                                                    radious=self.radious * 1, name="obst_" + str(i),
                                                    constraint_type='obstacle',
                                                    center_circle = center_circle)
                a_list += [tmp_a]
                
            
        # Collision avoidance with obstacles (time-sampling based)
        else:
            for i, obstacle in enumerate(self.obstacles):
                obst_corners = []
                for j, t in enumerate(np.linspace(0, 1, self.t_resolution_length)):
                    corner1 = MX.sym('obst_' + str(i) + '_corner1', 2); self.P += [corner1]; self.P_list += ['obst'] * 2; self.P0 += [0] * 2
                    corner2 = MX.sym('obst_' + str(i) + '_corner2', 2); self.P += [corner2]; self.P_list += ['obst'] * 2; self.P0 += [0] * 2
                    corner3 = MX.sym('obst_' + str(i) + '_corner3', 2); self.P += [corner3]; self.P_list += ['obst'] * 2; self.P0 += [0] * 2
                    corner4 = MX.sym('obst_' + str(i) + '_corner4', 2); self.P += [corner4]; self.P_list += ['obst'] * 2; self.P0 += [0] * 2
                    obst_corners += [[corner1, corner2, corner3, corner4]]
            
                self.collision_avoidance_hyperplane([p, q], obst_corners,
                                                    radious=self.radious, name="obst_" + str(i),
                                                    constraint_type='spline_obstacle_param',
                                                    n_samples=self.t_resolution_length)
            
        # Cost for extra acceleration in the frenet frame
        cost = 0
        for i in range(len(pq_dotdot)):
            cost += pq_dotdot[i]**2
        self.J += self.rho_input * definite_integral(cost, 0, 1)

        # Cost x_i - z_i
        z_i = self.define_MX_spline(degree = 3, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound = [], upper_bound = [],
                               name = ['z_i'] * self.n_dimensions,
                               category = 'parameter')
        lambda_i  = self.define_MX_spline(degree = 3, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound = [], upper_bound = [],
                               name = ['lambda_i'] * self.n_dimensions,
                               category = 'parameter')


        for i in range(len(y)):
            self.J += dot(lambda_i[i].coeffs,y[i].coeffs - z_i[i].coeffs)
            self.J += self.rho * dot(np.ones(y[i].coeffs.shape[0]), (y[i].coeffs - z_i[i].coeffs)**2)

        # Cost sum: x_i - z_ji
        for i in range(len(self.neighbours)):
            z_ji = self.define_MX_spline(degree = 3, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                                   lower_bound = [], upper_bound = [],
                                   name = ['z_ji'] * self.n_dimensions,
                                   category = 'parameter')
            lambda_ji  = self.define_MX_spline(degree = 3, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                                   lower_bound = [], upper_bound = [],
                                   name = ['lambda_ji'] * self.n_dimensions,
                                   category = 'parameter')

            for j in range(len(y)):
                self.J += dot(lambda_ji[j].coeffs,y[j].coeffs - z_ji[j].coeffs)
                self.J += self.rho * dot(np.ones(y[j].coeffs.shape[0]), (y[j].coeffs - z_ji[j].coeffs)**2)


        # Containers
        self.DvX = DecisionVarX(self.w_list, self.g_list, self.lbg, self.ubg)
        self.PvX = ParamValX(self.P_list, self.P0)
        self.DvX.extract(self.w0)

        "Initialize values with the pre-calculated values from initialize_x()"
        if self.initial_values != {}:
            self.DvX.extract(self.initial_values["w0_initial"])
            self.PvX.z_i = self.initial_values["z_i"]
            self.PvX.z_ji = self.initial_values["z_ji"]
            self.PvX.lambda_i = self.initial_values["lambda_i"]
            self.PvX.lambda_ji = self.initial_values["lambda_ji"]
            self.message_in["y_j"] = self.initial_values["y_j"]
            self.message_in["z_ji"] = self.initial_values["z_ji"]
            self.message_in["lambda_ji"] = self.initial_values["lambda_ji"]


        # Creating solver class
        prob = {'f': self.J,
                'x': vertcat(*self.w),
                'g': vertcat(*self.g),
                'p': vertcat(*self.P)
                }

        self.solver = nlpsol('solver', 'ipopt', prob, self.options)

        # Assembling the argument dictionary
        self.arg = {'x0' : self.w0,
                   'lbx': self.lbw,
                   'ubx': self.ubw,
                   'lbg': self.lbg,
                   'ubg': self.ubg,
                   'p': self.PvX.assemble()}


        return 
    
    def visualize_x_problem(self, ax, stage):
        """This function visualizes the arguments of the x-optimizer. This is to check, if 
        everything looks okay.
        Elements, defining parameters will be plotted with filled colors.
        Elements, defining decision variables will be plotted with opaque colors.
        """
        DvX = self.variable_history["DvX_posterior"][stage]
        PvX = self.variable_history["PvX"][stage]
        
        self_ID = self.plot_self_ID
        self_ID = 2
        obst_IDX = self.plot_obst_IDX
        
        if self.stage == 4:
            kappa = True
    
        def draw_colored_pq(ax, p_, q_, i = -1):
            cols = np.linspace(0,1,len(p_))
            points = np.array([p_, q_]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis')
            lc = LineCollection(segments, cmap='Wistia')
            lc = LineCollection(segments, cmap='hot')
            if i == 0:
                lc = LineCollection(segments, cmap='brg', label='corner trajectory')
            else:
                lc = LineCollection(segments, cmap='brg')
            line = ax.add_collection(lc)
            lc.set_array(cols)
            lc.set_linewidth(2)
            ax.plot()
            return ax, line
            
        # plot obstacles trajectories (PvX)
        n_c = self.n_obstacle_cropped_coeffs
        all_corners = np.array(PvX.obst)
        basis = self.obstacle_cropped_basis
        p = BSpline(basis, np.zeros(len(basis)))
        q = BSpline(basis, np.zeros(len(basis)))
        t = np.linspace(0.001, 1-0.001, 100)
        if self.ID == self_ID:    
            
            
            # Colors
            color=cm.Wistia(np.linspace(0,1,len(t)))
            color=cm.YlOrRd(np.linspace(0,1,len(t)))
            
            
            for i in range(len(self.obstacles)):
                idx = np.arange(n_c*int(4*2)*i,n_c*int(4*2)*i+n_c*int(4*2))
                obstacle_i_corners = all_corners[idx]
                
                p_c, q_c = [], []
                for j in range(4):
                    idx2 = np.arange(n_c*int(2)*j,n_c*int(2)*j+n_c*int(2))
                    corner_x = obstacle_i_corners[idx2[:n_c]]
                    corner_y = obstacle_i_corners[idx2[n_c:]]
                    p = BSpline(basis, corner_x)
                    q = BSpline(basis, corner_y)
                    
                    # Plot colored lines
                    p_ = np.array([p(t_)[0] for t_ in t]).reshape(-1)
                    q_ = np.array([q(t_)[0] for t_ in t]).reshape(-1)
                    ax, line = draw_colored_pq(ax, p_, q_, i)
                    
                    p_c.append(p)
                    q_c.append(q)
                    
                    
                # # Plot rectangles
                numera = 3
                t_c = [interp(int(i),[0,numera-1],[0,1]) for i in np.linspace(0, numera-1, numera)]
                color=cm.brg(np.linspace(0,1,numera))
                c = color
                for j, t_ in enumerate(t_c):
                    p_corners, q_corners = [], []
                    for p, q in zip(p_c, q_c):
                        p_ = p(t_).reshape(-1).tolist()[0]
                        q_ = q(t_).reshape(-1).tolist()[0]
                        
                        p_corners.append(p_)
                        q_corners.append(q_)
                    p_corners.append(p_corners[0])
                    q_corners.append(q_corners[0])
                    ax.plot(p_corners,
                        q_corners,
                        c = c[j])
                    plt.plot()
                    
                    
            # plot danger zone
            s_danger = 0.6988905493709299    
            circle = plt.Circle((0, 0), s_danger, color='r', alpha=0.5, zorder = 0)
            ax.add_patch(circle)
            ax.set_aspect('equal', adjustable='box')
                    
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
        
        # plot vehicle position trajectories
        x0 = np.array(PvX.x0)
        ax.plot(x0[0], x0[1], 'bo', zorder = 2)
        xf = np.array(PvX.xf)
        ax.plot(xf[0], xf[1], 'g*', zorder = 2)
        
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        p_coeffs = np.array(DvX.y)[np.arange(0, len(basis))]
        q_coeffs = np.array(DvX.y)[np.arange(len(basis), len(basis)*2)]
        phi_coeffs = np.array(DvX.y)[np.arange(len(basis)*2, len(basis)*3)]
        p = BSpline(basis, p_coeffs)
        q = BSpline(basis, q_coeffs)
        phi = BSpline(basis, phi_coeffs)
        
        p_ = p(t).reshape(-1).tolist()
        q_ = q(t).reshape(-1).tolist()
        phi_ = phi(t).reshape(-1).tolist()
        
        # ax.plot(p_, q_, c = 'cornflowerblue',lw=0.8,alpha = 0.5, zorder = 3)
        ax, line = draw_colored_pq(ax, p_, q_, -1)
        
        # plot vehicle phi trajectories
        # plot trajectory suggestions
        # plot intermediate positions
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        x = np.array(PvX.x_intermediate)
        for i in range(len(PvX.t_intermediate)):
            idx = np.arange(3*int(1)*i,3*int(1)*i+3*int(1))
            p = x[idx[0]]
            q = x[idx[1]]
            phi = x[idx[2]]
            
            ax.plot(p, q, 'ko',alpha = 0.1)
        
        if self.ID == "Dont plot hyperplanes":
        # if self.ID == self_ID:
            # plot hyperplanes
            a1_coeffs = np.array(DvX.a)[np.arange(len(basis)*2*(obst_IDX) + 0, len(basis)*2*(obst_IDX) + len(basis))]
            a2_coeffs = np.array(DvX.a)[np.arange(len(basis)*2*(obst_IDX) + len(basis), len(basis)*2*(obst_IDX) + len(basis) * 2)]
            b_coeffs = np.array(DvX.b)[np.arange(len(basis)*1*(obst_IDX) + 0, len(basis)*1*(obst_IDX) + len(basis))]
            
            a1 = BSpline(basis, a1_coeffs)
            a2 = BSpline(basis, a2_coeffs)
            b = BSpline(basis, b_coeffs)
            
            shrink = 0.01
            c=cm.brg(np.linspace(0,1,len(t)))
            for i, t_ in enumerate(t):
                if True:
                # if not(i%2) or (i == len(t) - 1):
                    x1 = np.linspace(-3 + i * shrink, 3 - i * shrink, 100)
                    if a2(t_) == 0:
                        x2 = x1
                    else:
                        x2 = (b(t_) - a1(t_) * x1) / a2(t_)
                    # Plot the lines
                    x1 = x1.reshape(-1).tolist()
                    x2 = x2.reshape(-1).tolist()
                    
                    
                    # Plot
                    ax.plot(x1, x2, c = c[i], zorder = 0)
                
                
            self.plot_drone(ax, x0 = x0[0],
                            y0 = x0[1],
                            theta_c = 0)
                
        for i, obst in enumerate(self.obstacles):
            if i == obst_IDX:
                search_length = len(self.obstacle_cropped_basis) * 2 # len(basis) * 2
                idx_center = np.arange(search_length*i,search_length*i+search_length)
                obst_center_coeffs = np.array(PvX.obst_center)[idx_center].reshape(-1).tolist()
                coeffs = [obst_center_coeffs[len(self.obstacle_cropped_basis) * k :len(self.obstacle_cropped_basis)*(k+1)] for k in range(2)]
                obst_center = [BSpline(basis, np.array(coeffs_)) for coeffs_ in coeffs]
                center_circle = [obst_center, self.obstacles[i].max_dist_from_center]
                
                p, q = obst_center
                p_ = p(t).reshape(-1).tolist()
                q_ = q(t).reshape(-1).tolist()
                # ax.plot(p_, q_, c = 'cornflowerblue',lw=0.8,alpha = 0.5, zorder = 3)
                ax, line = draw_colored_pq(ax, p_, q_, -1)
                
                s_danger = obst.max_dist_from_center * 0.9    
                for j, (p__, q__) in enumerate(zip(p_, q_)):
                    if not(j%10):
                        circle = plt.Circle((p__, q__), s_danger, color='r', alpha=0.1, zorder = 10)
                        ax.add_patch(circle)
                        ax.legend([circle, line], ['collision radious', 'corner trajectory'])
                        ax.set_aspect('equal', adjustable='box')
        return ax
    
    def distributed_x_update(self, list_):
        args, idx = list_
        return {idx: self.solver.call(args)}
    
    def distributed_z_update(self, list_):
        args, idx = list_
        return {idx: self.solver_z.call(args)}
        

    def x_update_prior(self):
        
        self.update_PvX()
        if self.shift_enabled == True:
            self.shift_DvX()
            self.shift_PvX()
        # Updating the necessary arguments for the solver
        self.arg['x0'] = self.DvX.assemble()
        self.arg['p'] = self.PvX.assemble()
        self.variable_history["DvX_prior"] += [copy.deepcopy(self.DvX)]
        self.variable_history["PvX"] += [copy.deepcopy(self.PvX)]
        return self
    
    def x_update(self):
        # Solving the problem
        start_time = time.time()
        self.solution = self.solver.call(self.arg)
        final_time = time.time()
        
        if self.solver.stats()['return_status'] == 'Solve_Succeeded':
            self.variable_history['first_time_success'] += [True]
        else:
            self.variable_history['first_time_success'] += [False]
            print(f"First time solver failed for vehicle {self.ID}, trying again.")
            self.arg['x0'] = self.solution["x"]
            self.solution = self.solver.call(self.arg)
            
            if self.solver.stats()['return_status'] != 'Solve_Succeeded':
                print(f"Second time solver failed for vehicle {self.ID}, trying again.")
                self.arg['x0'] = self.solution["x"]
                self.solution = self.solver.call(self.arg) 
            
        self.variable_history["x_update_time"] += [final_time - start_time]
        self.variable_history["solution"] += [self.solution]
        self.variable_history["solver_stats"] += [self.solver.stats()]
        
        return self
    
    def x_update_posterior(self):
        # Extracting the solution
        self.DvX.extract(self.solution)
        self.variable_history["DvX_posterior"] += [copy.deepcopy(self.DvX)]
        feasibility_dict = self.check_feasibility_of_solution_x()
        feasibility_dict["IPOPT_SUCCESS"] = self.solver.stats()['success']
        feasibility_dict["IPOPT_RETURN_STATUS"] = self.solver.stats()['return_status']
        self.variable_history["feasibility_dict"] += [feasibility_dict]
        self.variable_history['y'] += [self.DvX.y]
        self.variable_history['t_start'] += [self.t_start]
        self.variable_history['t_end'] += [self.t_end]
        self.variable_history['xf'] += [self.xf]
        self.variable_history['a'] += [list(self.DvX.a)]
        self.variable_history['b'] += [list(self.DvX.b)]
        return self
    
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    "Plotting"
    
    def plot_vehicle_frenet_trajectories(self, ax):
        flatten = lambda t: [item for sublist in t for item in sublist]

        # Plotting the environment
        ax = self.plot_environment(ax, self.t_start)

        # Creating the splines
        basis = self.define_knots(degree = 3, knot_intervals = self.knot_intervals)
        solution = self.solution['x'].full()
        coeffs1 = flatten([solution[x] for x in np.arange(0, len(basis))])
        coeffs2 = flatten([solution[x] for x in np.arange(len(basis), len(basis)*2)])

        p_solution = BSpline(basis, coeffs1)
        q_solution = BSpline(basis, coeffs2)
        # Sampling
        t_solution = np.linspace(0, 1, 100)
        t = np.linspace(self.t_start, self.t_end, 100)
        
        x_t, y_t = [], []
        for t_, t_solution_ in zip(t, t_solution):
            p_solution_, q_solution_ = p_solution(t_solution_)[0], q_solution(t_solution_)[0]
            x_, y_ = self.fp.frenet_to_inertial(p_solution_, q_solution_, t_)
            # x_, y_ = p_solution_, q_solution_
            x_t += [x_]
            y_t += [y_]
        ax.plot(x_t, y_t, 'k')

        # Plotting of obstacle
        for obstacle in self.obstacles:
            obstacle.plot_obstacle(ax)

        return ax
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    "Plotting formation error"
    
    def calculate_formation_error(self, intermediate_ADMM_idx = []):
        horizon0 = 0
        if self.t_step != 0:
            horizonf = int(1 / self.t_step)  
        else:
            horizonf = 1
        # Containers
        x_t_saved = []
        y_t_saved = []
        t_saved = []
        x_j_saved = []
        y_j_saved = []
        
        if intermediate_ADMM_idx == []:
            intermediate_ADMM_idx = self.n_intermediate_ADMM - 1
        "Collecting all the data"
        for horizon_range in range(horizon0, horizonf):
            horizon_num = int(horizon_range * self.n_intermediate_ADMM + intermediate_ADMM_idx)
            
            # Creating the splines
            basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
            solution = self.variable_history['y'][horizon_num]

            t_start = self.variable_history['t_start'][horizon_num]
            coeffs1 = solution[0:len(basis)]
            coeffs2 = solution[len(basis):len(basis)*2]
            p_solution = BSpline(basis, coeffs1)
            q_solution = BSpline(basis, coeffs2)
            x_t, y_t = [], []
            for t_pq, t_frenet in zip(np.linspace(0, self.t_step*1/self.t_window_size, 100), np.linspace(t_start, t_start + self.t_step, 100)):
                p_solution_, q_solution_ = p_solution(t_pq)[0], q_solution(t_pq)[0]
                x_, y_ = p_solution_, q_solution_
                x_t += [x_]
                y_t += [y_]
                
            x_t_saved += [x_t]
            y_t_saved += [y_t]
            t_saved += [np.linspace(t_start, t_start + self.t_step, 100)]
            
            y_j_all = self.variable_history['y_j'][horizon_num]
            y_j_all = np.array(y_j_all)
            
            x_j_saved_tmp = []
            y_j_saved_tmp = []
            for i in range(len(self.neighbours)):
                idx = np.arange(len(basis)*int(self.state_len/2)*i,len(basis)*int(self.state_len/2)*i+len(basis)*int(self.state_len/2))
                all_coeffs = y_j_all[idx]
                coeffs1 = all_coeffs[0:len(basis)].tolist()
                coeffs2 = all_coeffs[len(basis):len(basis)*2].tolist()
                p_solution = BSpline(basis, coeffs1)
                q_solution = BSpline(basis, coeffs2)
                x_t, y_t = [], []
                for t_pq, t_frenet in zip(np.linspace(0, self.t_step*1/self.t_window_size, 100), np.linspace(t_start, t_start + self.t_step, 100)):
                    p_solution_, q_solution_ = p_solution(t_pq)[0], q_solution(t_pq)[0]
                    x_, y_ = p_solution_, q_solution_
                    x_t += [x_]
                    y_t += [y_]
                    
                x_j_saved_tmp += [x_t]
                y_j_saved_tmp += [y_t]
            
            x_j_saved += [x_j_saved_tmp]
            y_j_saved += [y_j_saved_tmp]
            
            
        "Doing the calculation"
        # desired angle w.r.t. the two neighbours:
        vec1 = np.array(self.neighbours[0].xf[:2]) - np.array(self.xf[:2])
        vec2 = np.array(self.neighbours[-1].xf[:2]) - np.array(self.xf[:2])
        angle_original = math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        if angle_original > math.pi:
            angle_original = 2 * math.pi - angle_original
                    
        angle_error_saved = []
        for hr in range(horizon0, horizonf):
            angle_error = []
            for i in range(100):
                # current angle w.r.t. the two neighbours:
                vec1 = np.array([x_j_saved[hr][0][i], y_j_saved[hr][0][i]]) - np.array([x_t_saved[hr][i], y_t_saved[hr][i]])
                vec2 = np.array([x_j_saved[hr][-1][i], y_j_saved[hr][-1][i]]) - np.array([x_t_saved[hr][i], y_t_saved[hr][i]])
                angle_current = math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))    
                if angle_current > math.pi:
                    angle_current = 2 * math.pi - angle_current
                angle_error_tmp = abs(angle_original - angle_current)
                angle_error += [angle_error_tmp]
            angle_error_saved += [angle_error]
            
        
        return t_saved, angle_error_saved
                
            
    
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################

    "Plotting video"
    
    def plot_rotation(self, x, y, theta):
        x_new = x * cos(theta) - y * sin(theta)
        y_new = x * sin(theta) + y * cos(theta)
        return x_new, y_new
    
    def plot_moovie_frames_mooving_horizon(self, ax, horizon_num):
        t_steps = 100     
        self_ID = self.ID
        obst_IDX = self.plot_obst_IDX
        horizon_num_original = int(horizon_num)    
        horizon_num = int(horizon_num * self.n_intermediate_ADMM + self.n_intermediate_ADMM - 1)
        # Creating the splines
        basis = self.define_knots(degree = 3, knot_intervals = self.knot_intervals)
        solution = self.variable_history['y'][horizon_num]
        t_start = self.variable_history['t_start'][horizon_num]
        t_end = self.variable_history['t_end'][horizon_num]
        coeffs1 = solution[0:len(basis)]
        coeffs2 = solution[len(basis):len(basis)*2]
        p_solution = BSpline(basis, coeffs1)
        q_solution = BSpline(basis, coeffs2)
        x_t, y_t = [], []
        for t_pq, t_frenet in zip(np.linspace(0, 1, t_steps), np.linspace(t_start, t_end, t_steps)):
            p_solution_, q_solution_ = p_solution(t_pq)[0], q_solution(t_pq)[0]
            x_, y_ = self.fp.frenet_to_inertial(p_solution_, q_solution_, t_frenet)
            x_t += [x_]
            y_t += [y_]
            
            
        assert not((self.t_step * 100) % 1)
        
        # This is so that we can plot the part of the trajectory, which will be covered during the next simulation step, with one color,
        # and the remainder with a different color. (Black and cornflowerblue respectively.)
        if t_end > t_start:
            a = t_end - t_start
            virtual_idx = self.t_window_size * t_steps / a
            idx = math.ceil(virtual_idx * self.t_step * 1 / self.t_window_size)
        else:
            idx = t_steps
        
        # Plotting future trajectories (with two different colors)
        ax.plot(x_t[0:idx], y_t[0:idx], c = 'k',lw=0.8,alpha = 1, zorder = 5)
        if self.ID != self_ID:
            ax.plot(x_t[idx:], y_t[idx:], c = 'whitesmoke',lw=0.8,alpha = 0.5, zorder = 3)
        else:
            ax.plot(x_t[idx:], y_t[idx:], c = 'cornflowerblue',lw=0.8,alpha = 0.5, zorder = 3)

        # Plotting intermediate positions
        for i in range(len(self.variable_history['t_real_intermediate_list'][horizon_num_original])): 
            idx = np.arange(int(self.state_len/2)*i,int(self.state_len/2)*i+int(self.state_len/2))
            idx = np.arange(3*i,3*i+3)
            x_, y_ = self.fp.frenet_to_inertial(np.array([self.variable_history['x_intermediate_list'][horizon_num_original]]).reshape(-1)[idx][0], 
                                                np.array([self.variable_history['x_intermediate_list'][horizon_num_original]]).reshape(-1)[idx][1],
                                                self.variable_history['t_real_intermediate_list'][horizon_num_original][i])
            ax.plot(x_,
                    y_
                    , 'go', markersize = 2, zorder = 4)
        
        # Also, we plot the current configuration positions
        x_, y_ = self.fp.frenet_to_inertial(np.array([self.variable_history['current_configuration_position'][horizon_num_original]]).reshape(-1)[0], 
                                                np.array([self.variable_history['current_configuration_position'][horizon_num_original]]).reshape(-1)[1],
                                                t_start)
        ax.plot(x_,
                y_
                , 'bo', markersize = 2, zorder = 4)
        
        # Plotting the drone itself
        if self.ID == self_ID:
            x0, y0 = self.fp.frenet_to_inertial(p_solution(0)[0], q_solution(0)[0], t_start)
            
            self.plot_drone(ax, x0 = x0,
                            y0 = y0,
                            theta_c = math.atan2(self.fp.fy_d_spline(t_start)[0][0] + q_solution.derivative()(0),
                                        self.fp.fx_d_spline(t_start)[0][0] + p_solution.derivative()(0)))
        
        if self.ID == 0:
            # Plotting the environment
            ax = self.plot_environment(ax, t_start)
    
            # Plotting of obstacle
            for obstacle in self.obstacles:
                obstacle.plot_obstacle(ax)
        
        return t_start, ax
        
    def plot_drone(self, ax, x0, y0, theta_c):
        
        # Draw body
        radious = 0.05
        radious = self.radious * 0.5 #* 0.9
        height = radious
        width = radious

        # Draw arms
        l = 0.1
        l = radious * 2

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4)
        x1 = x0 + rot_x
        y1 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*3)
        x2 = x0 + rot_x
        y2 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*5)
        x3 = x0 + rot_x
        y3 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*7)
        x4 = x0 + rot_x
        y4 = y0 + rot_y


        # Rotors
        # https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
        r_rotor = 0.03
        r_rotor = l * 0.3

        # Adding stuff to ax

        # Lines
        line1 = Line2D([x0, x1], [y0, y1], zorder = 8)
        line2 = Line2D([x0, x2], [y0, y2], zorder = 8)
        line3 = Line2D([x0, x3], [y0, y3], zorder = 8)
        line4 = Line2D([x0, x4], [y0, y4], zorder = 8)
        ax.add_line(line1)
        ax.add_line(line2)
        ax.add_line(line3)
        ax.add_line(line4)

        # Circles
        circle1 = plt.Circle((x1, y1), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle2 = plt.Circle((x2, y2), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle3 = plt.Circle((x3, y3), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle4 = plt.Circle((x4, y4), r_rotor, color='k', alpha=0.5, zorder = 10)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        ax.add_patch(circle4)



        rect_x = - width/2
        rect_y = - height/2
        rot_x, rot_y = self.plot_rotation(rect_x, rect_y, theta_c)

        rectangle = patches.Rectangle((x0 + rot_x, y0 + rot_y), height, width,
                                      linewidth=1, edgecolor='k', facecolor='k',
                                      zorder = 9, angle = theta_c / np.pi * 180)
        ax.add_patch(rectangle)

        return ax
        
    def plot_configurations(self, ax):
        
        # Plotting of obstacle
        for obstacle in self.obstacles:
            obstacle.plot_obstacle(ax)
            
        x_t, y_t = [], []
        i = 0
        for t_intermediate_list, x_intermediate_list in zip(self.variable_history['t_real_intermediate_list'], self.variable_history['x_intermediate_list']):
            for i in range(self.n_of_saved_waypoints):
                idx = np.arange(int(self.state_len/2)*i,int(self.state_len/2)*i+int(self.state_len/2))
                print(idx)
                x = np.array([x_intermediate_list]).reshape(-1)[idx].tolist()
                print(x[0])
                print(x[1])
                print(i)
                print(t_intermediate_list)
                x_, y_ = self.fp.frenet_to_inertial(x[0], x[1], t_intermediate_list[i])
                x_t += [x_]
                y_t += [y_]
                
        ax.plot(x_t, y_t, 'ro')

    def plot_moovie_frames(self, ax, t):
        # https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html

        flatten = lambda t: [item for sublist in t for item in sublist]

        # Plotting the environment
        ax = self.plot_environment(ax, 0)

        # Plotting of obstacle
        for obstacle in self.obstacles:
            obstacle.plot_obstacle(ax)


        # Creating the splines
        basis = self.define_knots(degree = 3, knot_intervals = self.knot_intervals)
        solution = self.solution['x'].full()
        coeffs1 = flatten([solution[x] for x in np.arange(0, len(basis))])
        coeffs2 = flatten([solution[x] for x in np.arange(len(basis), len(basis)*2)])

        p_solution = BSpline(basis, coeffs1)
        q_solution = BSpline(basis, coeffs2)
        
        # Sampling
        x_t, y_t = [], []
        for i, t_ in enumerate(self.t_intermediate_list):
            idx = np.arange(int(self.state_len/2)*i,int(self.state_len/2)*i+int(self.state_len/2))
            p_solution_, q_solution_ = np.array(self.x_intermediate_list)[idx][0], np.array(self.x_intermediate_list)[idx][1]
            x_, y_ = self.fp.frenet_to_inertial(p_solution_, q_solution_, t_)
            x_t += [x_]
            y_t += [y_]
            
        ax.plot(x_t, y_t, 'ro', markersize = 2)   
        
        x_t, y_t = [], []
        for t_ in np.linspace(0, t):
            p_solution_, q_solution_ = p_solution(t_)[0], q_solution(t_)[0]
            x_, y_ = self.fp.frenet_to_inertial(p_solution_, q_solution_, t_)
            x_t += [x_]
            y_t += [y_]
            
        # ax.plot(x_t, y_t, 'k', linewidth = 0.5)    
        ax.plot(x_t, y_t, c = 'cornflowerblue',lw=1.0,alpha = 0.9, zorder = 7)

        theta_c = 0
        theta_c = self.fp.fy_d_spline(t)[0][0] / self.fp.fx_d_spline(t)[0][0]

        # Draw body
        radious = 0.05
        radious = self.radious * 0.5 #* 0.9
        height = radious
        width = radious


        # Draw arms
        l = 0.1
        l = radious * 2

        p_solution_, q_solution_ = p_solution(t)[0], q_solution(t)[0]
        x0, y0 = self.fp.frenet_to_inertial(p_solution_, q_solution_, t)

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4)
        x1 = x0 + rot_x
        y1 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*3)
        x2 = x0 + rot_x
        y2 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*5)
        x3 = x0 + rot_x
        y3 = y0 + rot_y
        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*7)
        x4 = x0 + rot_x
        y4 = y0 + rot_y

        # Rotors
        # https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
        r_rotor = 0.03
        r_rotor = l * 0.3

        # Adding stuff to ax

        # Lines
        line1 = Line2D([x0, x1], [y0, y1], zorder = 8)
        line2 = Line2D([x0, x2], [y0, y2], zorder = 8)
        line3 = Line2D([x0, x3], [y0, y3], zorder = 8)
        line4 = Line2D([x0, x4], [y0, y4], zorder = 8)
        ax.add_line(line1)
        ax.add_line(line2)
        ax.add_line(line3)
        ax.add_line(line4)

        # Circles
        circle1 = plt.Circle((x1, y1), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle2 = plt.Circle((x2, y2), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle3 = plt.Circle((x3, y3), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle4 = plt.Circle((x4, y4), r_rotor, color='k', alpha=0.5, zorder = 10)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        ax.add_patch(circle4)

        rect_x = - width/2
        rect_y = - height/2
        rot_x, rot_y = self.plot_rotation(rect_x, rect_y, theta_c)

        rectangle = patches.Rectangle((x0 + rot_x, y0 + rot_y), height, width,
                                      linewidth=1, edgecolor='k', facecolor='k',
                                      zorder = 9, angle = theta_c / np.pi * 180)
        ax.add_patch(rectangle)

        return ax