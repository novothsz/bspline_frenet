import numpy as np
from numpy import interp
import math
from time import time
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from casadi import MX, SX, Function, vertcat, nlpsol, cos, sin, norm_2 #, dot
from .spline import BSpline, BSplineBasis
from .spline_extra import definite_integral


from .vehicle_basis import VehicleBasis
from .param import ParamValX, ParamValZ, DecisionVarX, DecisionVarZ


class Vehicle(VehicleBasis):
    def __init__(self):
        super().__init__()
        
        self.t_start = 0.0
        self.t_step = 0.01 # 0.04 # Changed in simulation_step() upon first call
        # self.t_step = 1 # 0.04 # Changed in simulation_step() upon first call
        # self.t_step = 0.5 # 0.04 # Changed in simulation_step() upon first call
        self.t_window_size = 0.2
        # self.t_window_size = 0.5
        # self.t_window_size = 1.0
        self.t_end = self.t_start + self.t_window_size
        self.simulation = False
        self.shift_enabled = False
        
    


    def setup_z_update(self):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []
        


        # Define ADMM cost
        # x_i - z_i
        y = self.define_MX_spline(degree=self.state_degree, knot_intervals=self.knot_intervals, n_spl=self.n_dimensions,
                                lower_bound=self.y_min, upper_bound=self.y_max,
                                name=["y"] * self.n_dimensions,
                                category = 'parameter')

        z_i = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound=self.y_min, upper_bound=self.y_max,
                               initial_value = [[self.x0[0], self.xf[0]], [self.x0[1], self.xf[1]], [self.x0[2], self.xf[2]]],
                               name = ['z_i'] * self.n_dimensions)

        lambda_i  = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound = [], upper_bound = [],
                               name = ['lambda_i'] * self.n_dimensions,
                               category = 'parameter')
        
        """ ---- Frenet ---- """
        """ ---- Frenet ---- """
        p_sum = 0
        q_sum = 0
        p_sum += z_i[0]
        q_sum += z_i[1]
        """ ---- Frenet ---- """
        """ ---- Frenet ---- """
            

        for i in range(len(y)):
            self.J += definite_integral(lambda_i[i] * (y[i] - z_i[i]), 0, 1)
            self.J += definite_integral(self.rho * (y[i] - z_i[i])**2, 0, 1)

        # Cost sum: x_j - z_ij
        for i in range(len(self.neighbours)):

            y_j = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                               lower_bound = [], upper_bound = [],
                               name = ['y_j'] * self.n_dimensions,
                               category = 'parameter')

            z_ij = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                                   lower_bound=self.y_min, upper_bound=self.y_max,
                                   initial_value = [[self.neighbours[i].x0[0], self.neighbours[i].xf[0]], [self.neighbours[i].x0[1], self.neighbours[i].xf[1]], [self.neighbours[i].x0[2], self.neighbours[i].xf[2]]],
                                   name = ['z_ij'] * self.n_dimensions)
            lambda_ij  = self.define_MX_spline(degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = self.n_dimensions,
                                   lower_bound = [], upper_bound = [],
                                   name = ['lambda_ij'] * self.n_dimensions,
                                   category = 'parameter')


            for j in range(len(y)):
                self.J += definite_integral(lambda_ij[j] * (y_j[j] - z_ij[j]), 0, 1)
                self.J += definite_integral(self.rho * (y_j[j] - z_ij[j])**2, 0, 1)


            def cross_product(spline1, spline2):
                a1, a2 = spline1[0], spline1[1]
                b1, b2 = spline2[0], spline2[1]
                return (a1 * b2 - a2 * b1)**2
            
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
            # cross = cross_product(vec1, vec2)
            # dot = dot_product(vec1, vec2)
            # usage: cross(t), dot(t)
            
            
            # Formation constraint.
            for t in np.linspace(0, 1, self.t_resolution_length):
                self.define_constraint([cross_product(vec1, vector_rotation(vec2, z_i[2], t))(t)],
                                        [-self.slack * 1],
                                        [self.slack * 1],
                                        constraint_type='time',
                                        name=["formation_vehicle_" + str(i)] * self.n_dimensions_old)
            # But the dot product should be > 0, to avoid the vehicles switching place and still
            # fulfilling the formation requirements (at least for those two vehicles)

            for t in np.linspace(0, 1, self.t_resolution_length):
                self.define_constraint([dot_product(vec1, vector_rotation(vec2, z_i[2], t))(t)],
                                        [0],
                                        [math.inf],
                                        constraint_type='time',
                                        name=["formation_dot_vehicle_" + str(i)] * self.n_dimensions_old)

            # TODO: I think this is not really needed anymore, but need to check
            # for j in range(len(y)):
            #     self.J += self.rho_formation * definite_integral( ((vec1[j] - vec2[j])**2), 0, 1)


            """Special distance-constraint"""
            """Special distance-constraint"""
            # frenet_zero = MX((0, 0))
            xf = np.array(self.xf[:self.n_dimensions_old])
            xf_j = np.array(self.neighbours[i].xf[:self.n_dimensions_old])

            dist_we_have = (z_i[0] - z_ij[0])**2 \
                            + (z_i[1] - z_ij[1])**2
            dist_we_want = (xf[0] - xf_j[0])**2 \
                            + (xf[1] - xf_j[1])**2
            dist_difference = (dist_we_have * 1 - dist_we_want * 0.25) * 1  # 0.5 means we can shrink to the quarter of the size

            ""
            for t in np.linspace(0, 1, self.t_resolution_length):
                self.define_constraint([dist_difference(t)],
                                        [0.0],
                                        [math.inf],
                                        constraint_type='time',
                                        name=["formation_vehicle_" + str(i)] * self.n_dimensions_old)

                # We can add collision avoidance here too :)
                self.define_constraint([dist_we_have(t)],
                                        [(self.radious * 2 * self.vehicle_avoidnce_multiplier)**2],
                                        [math.inf],
                                        constraint_type='time',
                                        name=["formation_vehicle_" + str(i)] * self.n_dimensions_old)
            """Special distance-constraint"""
            """Special distance-constraint"""

            

            """ ---- Frenet ---- """
            """ ---- Frenet ---- """
            # We add an extra constraint, if we are in the Frenet-frame.
            
            # Péni féle
            # If we add the p components and if we add the q components together, 
            # each of these should equal to zero. Meaning, the center of gravity 
            # should be in the (0, 0) point of the Frenet-frame.
            
            # This is defined outside of this loop :)
            # p_sum = z_i[0]
            # q_sum = z_i[1]
            
            p_sum += z_ij[0]
            q_sum += z_ij[1]
                
            """ ---- Frenet ---- """
            """ ---- Frenet ---- """
                
                    
        
        """ ---- Frenet ---- """
        """ ---- Frenet ---- """
        for t in np.linspace(0, 1, self.t_resolution_length):
            self.define_constraint([p_sum(t), q_sum(t)],
                                    [-self.slack,-self.slack],
                                    [ self.slack, self.slack],
                                    constraint_type='time',
                                    name=["frenet_zero_" + str(i)] * self.n_dimensions_old)
        """ ---- Frenet ---- """
        """ ---- Frenet ---- """


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
    
    def z_update(self):
        self.update_PvZ()
        if self.shift_enabled == True:
            self.shift_DvZ()
            self.shift_PvZ()
        # Updating the necessary arguments for the solver
        self.arg_z['x0'] = self.DvZ.assemble()
        self.arg_z['p'] = self.PvZ.assemble()

        # Solving the problem
        # t1 = time.time()
        self.solution_z = self.solver_z.call(self.arg_z)
        # t2 = time.time()
        # self.z_update_time += [t2-t1]

        # Extracting the solution
        self.DvZ.extract(self.solution_z)

        return self
    
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
                                initial_value = [ [self.x0[0], self.xf[0]], [self.x0[1], self.xf[1]], [self.x0[2], self.xf[2]] ],
                                name=["y"] * self.n_dimensions)

        y = pq # we basically rename the thing :)
        pq_dot = [pq_.derivative() for pq_ in pq]
        pq_dotdot = [pq_dot_.derivative() for pq_dot_ in pq]

        p = pq[0]
        q = pq[1]
        phi = pq[2]
        
        p_dot = pq_dot[0]
        q_dot = pq_dot[1]
        phi_dot = pq_dot[2]
        
        # p_dotdot = pq_dotdot[0]
        # q_dodott = pq_dotdot[1]
        
        # Initial position constraint on y
        self.define_constraint([p, q, phi],
                                x0[:self.n_dimensions],
                                x0[:self.n_dimensions],
                                constraint_type='initial_param',
                                name=["y0"] * self.n_dimensions)
        # Initial velocity constraint on dy
        self.define_constraint([p_dot, q_dot, phi_dot],
                                x0[self.n_dimensions:self.n_dimensions*2],
                                x0[self.n_dimensions:self.n_dimensions*2],
                                constraint_type='initial_param',
                                name=["dy0"] * self.n_dimensions)
        # Version 2
        
        # self.J += self.rho_final_value * ((p.coeffs[-1] - self.xf[0])**2 \
        #                                 + (p.derivative().coeffs[-1] - self.xf[2])**2 \
        #                                 + (q.coeffs[-1] - self.xf[1])**2 \
        #                                 + (q.derivative().coeffs[-1] - self.xf[3])**2)
        
        "final_param"
        # self.J += self.rho_final_value * ((p.coeffs[-1] - self.xf[0])**2)
        # self.J += self.rho_final_value * ((q.coeffs[-1] - self.xf[1])**2)
        # self.J += self.rho_final_value * ((phi.coeffs[-1] - self.xf[2])**2)
        
        lambda_ = np.power(np.linspace(1, 0, p.coeffs.shape[0]), 1)
        for i in range(p.coeffs.shape[0]):
            self.J += self.rho_final_value * ((p.coeffs[i] -      (lambda_[i] * x0[0] + (1 - lambda_[i]) * xf[0])     )**2)
            self.J += self.rho_final_value * ((q.coeffs[i] -      (lambda_[i] * x0[1] + (1 - lambda_[i]) * xf[1])     )**2)
            self.J += self.rho_final_value * ((phi.coeffs[i] -      (lambda_[i] * x0[2] + (1 - lambda_[i]) * xf[2])     )**2)
            
            
        # "initial_param"    
        # lambda_ = np.power(np.linspace(0, 1, p.coeffs.shape[0]), 8)
        # for i in range(p.coeffs.shape[0]):
        #     self.J += self.rho_final_value * ((p.coeffs[i] -      (lambda_[i] * x0[0] + (1 - lambda_[i]) * xf[0])     )**2)
        #     self.J += self.rho_final_value * ((q.coeffs[i] -      (lambda_[i] * x0[1] + (1 - lambda_[i]) * xf[1])     )**2)
            
        # self.J += self.rho_final_value * ((p.coeffs[-1] - xf[0])**2) * 10
        # self.J += self.rho_final_value * ((q.coeffs[-1] - xf[1])**2) * 10
        
        # self.J += self.rho_final_value * ((phi.coeffs[-1] - self.xf[2])**2)
        
        # a = p.coeffs[-1] - self.xf[0]
        # b = q.coeffs[-1] - self.xf[1]
        # self.J += self.rho_final_value * (a**2 + b**2)**2
        # self.J += self.rho_final_value * ((phi.coeffs[-1] - self.xf[2])**2)
        
        self.define_constraint([phi],
                                xf[self.n_dimensions] - [5 / 360 * math.pi * 2],
                                xf[self.n_dimensions] + [5 / 360 * math.pi * 2],
                                constraint_type='final_param',
                                name=["phif"] * self.n_dimensions)
                                        
        # Version 1
        # Final position constraint on y
        self.define_constraint([p, q],
                                xf[:self.n_dimensions_old] - [self.radious*1, self.radious*1],
                                xf[:self.n_dimensions_old] + [self.radious*1, self.radious*1],
                                constraint_type='final_param',
                                name=["yf"] * self.n_dimensions)
        
        
        # self.define_constraint([p, q],
        #                         vertcat(xf[0] - self.radious*1, xf[1] - self.radious*1),
        #                         vertcat(xf[0] + self.radious*1, xf[1] + self.radious*1),
        #                         constraint_type='final_param',
        #                         name=["yf"] * self.n_dimensions)

        # # Final velocity constraint on dy
        # self.define_constraint([p_dot, q_dot, phi_dot],
        #                         xf[self.n_dimensions:self.n_dimensions*2],
        #                         xf[self.n_dimensions:self.n_dimensions*2],
        #                         constraint_type='final_param',
        #                         name=["dyf"] * self.n_dimensions)

        # TODO: overall constraints on y, dy and u
        
        "Version 2"
        # """
        v_s = MX.sym('v_s', self.t_resolution_length); self.P += [v_s]; self.P_list += ['v_s'] * self.t_resolution_length; self.P0 += [0] * self.t_resolution_length
        curvature = MX.sym('curvature', self.t_resolution_length); self.P += [curvature]; self.P_list += ['curvature'] * self.t_resolution_length; self.P0 += [0] * self.t_resolution_length
        equation_min_p = MX.sym('equation_min_p', self.t_resolution_length); self.P += [equation_min_p]; self.P_list += ['equation_min_p'] * self.t_resolution_length; self.P0 += [0] * self.t_resolution_length
        equation_max_p = MX.sym('equation_max_p', self.t_resolution_length); self.P += [equation_max_p]; self.P_list += ['equation_max_p'] * self.t_resolution_length; self.P0 += [0] * self.t_resolution_length
        equation_min_q = MX.sym('equation_min_q', self.t_resolution_length); self.P += [equation_min_q]; self.P_list += ['equation_min_q'] * self.t_resolution_length; self.P0 += [0] * self.t_resolution_length
        equation_max_q = MX.sym('equation_max_q', self.t_resolution_length); self.P += [equation_max_q]; self.P_list += ['equation_max_q'] * self.t_resolution_length; self.P0 += [0] * self.t_resolution_length
        # obst_corners
        
        "p_dot equation"
        for i, t in enumerate(np.linspace(0, 1, self.t_resolution_length)):
            expression1 = -p_dot(t) -v_s[i] * (1 - curvature[i] * q(t)) + equation_min_p[i]
            expression2 = -p_dot(t) -v_s[i] * (1 - curvature[i] * q(t)) + equation_max_p[i]
            
            self.define_constraint([expression1],
                                    [-math.inf],
                                    [0],
                                    constraint_type='time',
                                    name=["p_dot_min"])
            
            self.define_constraint([expression2],
                                    [0],
                                    [math.inf],
                                    constraint_type='time',
                                    name=["p_dot_max"])
            
        "q_dot equation"
        for i, t in enumerate(np.linspace(0, 1, self.t_resolution_length)):
            expression1 = -q_dot(t) -v_s[i] * p(t) * curvature[i] + equation_min_q[i]
            expression2 = -q_dot(t) -v_s[i] * p(t) * curvature[i] + equation_max_q[i]
            
            self.define_constraint([expression1],
                                    [-math.inf],
                                    [0],
                                    constraint_type='time',
                                    name=["q_dot_min"])
            
            self.define_constraint([expression2],
                                    [0],
                                    [math.inf],
                                    constraint_type='time',
                                    name=["q_dot_max"])
            
        # Collision avoidance with obstacles and neighbours or w.t.f.?
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
            
            
        # """    
        
        """
        "Version 1"
        # Invoke the frenet frame so that we can fill up the parameters
        v_s  = self.fp.fx_d_spline + self.fp.fy_d_spline
        # sin_theta_c = self.fp.sin_f_theta_spline
        # cos_theta_c = self.fp.cos_f_theta_spline
        curvature = self.fp.fy_c_spline
        
        
        
        # This is what we are going to do... We will be searching for p_dot and q_dot values
        # that are not equal to the value that the system dynamics dictates. Instead 
        # we replace vx & vy with their minimum and maximum values. This gives us constraints
        # on how we are allowed to choose p_dot and q_dot. Thus, we will find optimal values for
        # the states of the system that adhere to the minimum and maximum constraints of vx & vy.
        # This, of course is not enough, later we need to do something similar for p_dotdot, q_dotdot / ax, ay.
        # Furthermore, we will plot the constraints just so we can see wether we even can find a solution or not.
        # And using these values will we calculate vx, vy and ax, ay after the fact :)
        
        
        [equation_min_p, equation_max_p] = self.fp.equation_min_max('p')
        [equation_min_q, equation_max_q] = self.fp.equation_min_max('q')
        
        # Okay... well... coefficient reduction wasn't enough, because the problem is
        # the multiplication: curvature * p.
        # For this reason: we have to time-sample the whole thing right here and now. Meh...
        
        "p_dot equation"
        for t in np.linspace(0, 1, self.t_resolution_length):
            expression1 = -p_dot(t) -v_s(t) * (1 - curvature(t) * q(t)) + equation_min_p(t)
            expression2 = -p_dot(t) -v_s(t) * (1 - curvature(t) * q(t)) + equation_max_p(t)
            
            self.define_constraint([expression1],
                                    [-math.inf],
                                    [0],
                                    constraint_type='time',
                                    name=["p_dot_min"])
            
            self.define_constraint([expression2],
                                    [0],
                                    [math.inf],
                                    constraint_type='time',
                                    name=["p_dot_max"])
            
            
        "q_dot equation"
        for t in np.linspace(0, 1, self.t_resolution_length):
            expression1 = -q_dot(t) -v_s(t) * p(t) * curvature(t) + equation_min_q(t)
            expression2 = -q_dot(t) -v_s(t) * p(t) * curvature(t) + equation_max_q(t)
            
            self.define_constraint([expression1],
                                    [-math.inf],
                                    [0],
                                    constraint_type='time',
                                    name=["q_dot_min"])
            
            self.define_constraint([expression2],
                                    [0],
                                    [math.inf],
                                    constraint_type='time',
                                    name=["q_dot_max"])
            
            
        # Collision avoidance with obstacles and neighbours or w.t.f.?
        for i, obstacle in enumerate(self.obstacles):
            self.collision_avoidance_hyperplane([p, q], obstacle.corners_spline,
                                                radious=self.radious, name="obst_" + str(i),
                                                constraint_type='spline_obstacle_spline_t',
                                                n_samples=self.t_resolution_length)
        """
        
        # Cost function
        # TODO In frenet frame this cost function does not apply! It needs to be changed!
        cost = 0
        for i in range(len(pq_dotdot)):
            cost += pq_dotdot[i]**2
        # Cost for: acceleration in the frenet frame
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
            self.J += definite_integral(lambda_i[i] * (y[i] - z_i[i]), 0, 1)
            self.J += definite_integral(self.rho * (y[i] - z_i[i])**2, 0, 1)

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
                self.J += definite_integral(lambda_ji[j] * (y[j] - z_ji[j]), 0, 1)
                self.J += definite_integral(self.rho * (y[j] - z_ji[j])**2, 0, 1)

            # # Inter-vehicle collision avoidance
            # for i, neighbour in enumerate(self.neighbours):
            #     self.collision_avoidance_hyperplane(y, z_ji, radious = self.radious/10, name = "inter_vehicle" + str(i), constraint_type="inter_vehicle")


        # Containers
        self.DvX = DecisionVarX(self.w_list, self.g_list, self.lbg, self.ubg)
        self.PvX = ParamValX(self.P_list, self.P0)
        # self.PvX.obst = tempOOO

        # Initializing values - below this is overwritten by the initial values
        # found during initialize_x()
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


        return self

    def x_update(self):
        self.update_PvX()
        if self.shift_enabled == True:
            self.shift_DvX()
            self.shift_PvX()
        # Updating the necessary arguments for the solver
        self.arg['x0'] = self.DvX.assemble()
        self.arg['p'] = self.PvX.assemble()
        # Solving the problem
        self.solution = self.solver.call(self.arg)
        # Extracting the solution
        self.DvX.extract(self.solution)
        self.variable_history['y'] += [self.DvX.y]
        self.variable_history['t_start'] += [self.t_start]
        self.variable_history['t_end'] += [self.t_end]
        self.variable_history['xf'] += [self.xf]
        return self
    
    
    
    # basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
    # solution = self.DvX.y
    # coeffs1 = solution[0:len(basis)]
    # coeffs2 = solution[len(basis):len(basis)*2]
    # p_solution = BSpline(basis, coeffs1)
    # q_solution = BSpline(basis, coeffs2)
    # p_solution_, q_solution_ = p_solution(self.t_step*1/self.t_window_size)[0], q_solution(self.t_step*1/self.t_window_size)[0]
    # print(p_solution_, q_solution_)
    
    # print(self.PvX.x0[:2])
    
    
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
        
        # from .spline_extra import shift_spline
        # p_solution.coeffs = shift_spline(p_solution.coeffs, 0.4, p_solution.basis)
        # q_solution.coeffs = shift_spline(q_solution.coeffs, 0.4, q_solution.basis)
        
        
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
        """
        "Below is the same as for the generic plot_vehicle_trajecotries()"
        
        # Fitting a polynome of degree 7 onto the spline
        poly7_x = np.poly1d(np.polyfit(t, x_t, deg=7))
        poly7_y = np.poly1d(np.polyfit(t, y_t, deg=7))

        # Now, the 7 degree polynomials are calculated such that upon evaluation
        # between t = [0, 1] we get correct values. Outside this range, they don't
        # represent the trajectories we calculated.
        # Below we rescale the polynomials such that they give correct values
        # between t = [0, t_desired]
        # Remember: x^7 --> x^7 / t_desired^7
        power = 0
        for i in range(len(poly7_x.coeffs)-1, 0-1, -1):
            poly7_x.coeffs[i] = poly7_x.coeffs[i] / pow(self.T, power)
            poly7_y.coeffs[i] = poly7_y.coeffs[i] / pow(self.T, power)
            power += 1

        # Let's now sample from this, for the sake of plotting
        tp7 = np.linspace(0, self.T, 100)
        poly7_x_t = [poly7_x(t_) for t_ in tp7]
        poly7_y_t = [poly7_y(t_) for t_ in tp7]

        # The path !!! now with correct arrangement of the coefficients !!!
        # Storing the coefficients in the format, that crazyswarm requires
        # (x^0, x^1, x^2, ...)
        poly7_x = poly7_x.coeffs.tolist()
        poly7_x.reverse()
        poly7_y = poly7_y.coeffs.tolist()
        poly7_y.reverse()
        # self.write_csv(self.T, poly7_x, poly7_y)

        # Adding an extra 7 degree polynomial for hoowering at the end
        final_x = poly7_x_t[-1]
        final_y = poly7_y_t[-1]
        however_x = [final_x] + [0] * 7
        however_y = [final_y] + [0] * 7

        # Combining the polinomials into a list
        T_list = [[self.T], [2]]
        poly7_x_list = [poly7_x, however_x]
        poly7_y_list = [poly7_y, however_y]

        # Writing the list to file
        self.write_csv(T_list, poly7_x_list, poly7_y_list)
        """
        # ax.plot(poly7_x_t, poly7_y_t, 'k*')

        # Plotting of spline & control points
        # ax.plot(coeffs1, coeffs2, 'ro')
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
        horizonf = int(1 / self.t_step)  
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
            # horizon_num = intermediate_ADMM_idx
            # Creating the splines
            basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
            solution = self.variable_history['y'][horizon_num]
            # print(len(self.variable_history['y']))
            # print(horizon_num)
            t_start = self.variable_history['t_start'][horizon_num]
            # t_end = self.variable_history['t_end'][horizon_num]
            coeffs1 = solution[0:len(basis)]
            coeffs2 = solution[len(basis):len(basis)*2]
            p_solution = BSpline(basis, coeffs1)
            q_solution = BSpline(basis, coeffs2)
            x_t, y_t = [], []
            for t_pq, t_frenet in zip(np.linspace(0, self.t_step*1/self.t_window_size, 100), np.linspace(t_start, t_start + self.t_step, 100)):
                p_solution_, q_solution_ = p_solution(t_pq)[0], q_solution(t_pq)[0]
                # x_, y_ = self.fp.frenet_to_inertial(p_solution_, q_solution_, t_frenet)
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
                    # x_, y_ = self.fp.frenet_to_inertial(p_solution_, q_solution_, t_frenet)
                    x_, y_ = p_solution_, q_solution_
                    x_t += [x_]
                    y_t += [y_]
                    
                x_j_saved_tmp += [x_t]
                y_j_saved_tmp += [y_t]
            
            x_j_saved += [x_j_saved_tmp]
            y_j_saved += [y_j_saved_tmp]
            
            
        # plt.figure()
        # for x_t, y_t in zip(x_t_saved, y_t_saved):
        #     plt.plot(x_t, y_t)
        # plt.show()
        # plt.figure()
        # for x_t, y_t in zip(x_j_saved, y_j_saved):
        #     plt.plot(x_t[0], y_t[0])
        # plt.show()
            
            
        "Doing the calculation"
        # desired angle w.r.t. the two neighbours:
        vec1 = np.array(self.neighbours[0].xf[:2]) - np.array(self.xf[:2])
        vec2 = np.array(self.neighbours[-1].xf[:2]) - np.array(self.xf[:2])
        angle_original = math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        if angle_original > math.pi:
            angle_original = 2 * math.pi - angle_original
                    
        # print(angle_original)
        angle_error_saved = []
        for hr in range(horizon0, horizonf):
            angle_error = []
            for i in range(100):
                # for j in range(len(self.neighbours)):
                # current angle w.r.t. the two neighbours:
                vec1 = np.array([x_j_saved[hr][0][i], y_j_saved[hr][0][i]]) - np.array([x_t_saved[hr][i], y_t_saved[hr][i]])
                vec2 = np.array([x_j_saved[hr][-1][i], y_j_saved[hr][-1][i]]) - np.array([x_t_saved[hr][i], y_t_saved[hr][i]])
                angle_current = math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))    
                if angle_current > math.pi:
                    angle_current = 2 * math.pi - angle_current
                angle_error_tmp = abs(angle_original - angle_current)
                angle_error += [angle_error_tmp]
            angle_error_saved += [angle_error]
            
            
        # plt.figure()
        # for time, angle in zip(t_saved, angle_error_saved):
        #     plt.plot(time, angle)
        # plt.show()
        
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
        
        t_steps = 10     
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
        # idx = int(100 * self.t_step * 1 / self.t_window_size + 1) # int(1 / self.t_step - 1)
        # print("idx_old:" + str(idx))
        # idx = int(100 - abs(self.t_step - t_start) / (t_end - t_start))
        # print("idx_new:" + str(idx))
        # idx = int((self.t_step + t_start) * (100))
        # print("idx_new:" + str(idx))
        
        if t_end > t_start:
            a = t_end - t_start
            virtual_idx = self.t_window_size * t_steps / a
            idx = math.ceil(virtual_idx * self.t_step * 1 / self.t_window_size)
        else:
            idx = t_steps
        
        ax.plot(x_t[0:idx], y_t[0:idx], c = 'k',lw=0.8,alpha = 1, zorder = 5)
        # ax.plot(x_t[idx], y_t[idx], c = 'r', marker = 'o', markersize = 1, lxw=1,alpha = 1, zorder = 6)
        ax.plot(x_t[idx:], y_t[idx:], c = 'cornflowerblue',lw=0.8,alpha = 0.5, zorder = 3)
        
        if self.vehicle_positions_new['vehicle_positions_new'][horizon_num_original] != []:
            # ax.plot(x_t[-1], y_t[-1], 'go', markersize = 1, zorder = 3)
            # print(horizon_num_original)
            
            x_, y_ = self.fp.frenet_to_inertial(self.vehicle_positions_new['vehicle_positions_new'][horizon_num_original][0], 
                                                self.vehicle_positions_new['vehicle_positions_new'][horizon_num_original][1],
                                                t_end)
            ax.plot(x_,
                    y_,
                    'ro', markersize = 1, zorder = 3)
        else:
            # ax.plot(x_t[-1], y_t[-1], 'ro', markersize = 1, zorder = 3)
            # ax.plot(self.xf[0], self.xf[1], 'go', markersize = 1, zorder = 3)
            x_, y_ = self.fp.frenet_to_inertial(self.variable_history['xf'][horizon_num_original][0], 
                                                self.variable_history['xf'][horizon_num_original][1],
                                                t_end)
            ax.plot(x_,
                    y_
                    , 'go', markersize = 1, zorder = 3)
        
        
        x0, y0 = self.fp.frenet_to_inertial(p_solution(0)[0], q_solution(0)[0], t_start)
        
        self.plot_drone(ax, x0 = x0,
                        y0 = y0,
                        theta_c = math.atan2(self.fp.fy_d_spline(t_start)[0][0] + q_solution.derivative()(0),
                                    self.fp.fx_d_spline(t_start)[0][0] + p_solution.derivative()(0)))
        
        # theta_c = self.fp.fy_d_spline(t_start)[0][0] / self.fp.fx_d_spline(t_start)[0][0] + \
        #                     q_solution.derivative()(0) / p_solution.derivative()(0)
        
        
        
        # Plotting the environment
        ax = self.plot_environment(ax, t_start)

        # Plotting of obstacle
        for obstacle in self.obstacles:
            obstacle.plot_obstacle(ax)
        
        return t_start, ax
        
    def plot_drone(self, ax, x0, y0, theta_c):
        
        # Draw body
        radious = 0.05
        radious = self.radious * 0.5 * 0.9
        height = radious
        width = radious


        # Draw arms
        l = 0.1
        l = radious * 2


        # x0 = x(t) + self.fp.fx_spline(t)
        # y0 = y(t) + self.fp.fy_spline(t)
        # p_solution_, q_solution_ = p_solution(t)[0], q_solution(t)[0]
        # x0, y0 = self.fp.frenet_to_inertial(p_solution_, q_solution_, t)



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
        

    def plot_moovie_frames(self, ax, t):
        # https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html

        flatten = lambda t: [item for sublist in t for item in sublist]

        # Plotting the environment
        ax = self.plot_environment(ax)

        # Plotting of obstacle
        for obstacle in self.obstacles:
            obstacle.plot_obstacle(ax)


        # Creating the splines
        basis = self.define_knots(degree = 3, knot_intervals = self.knot_intervals)
        solution = self.solution['x'].full()
        coeffs1 = flatten([solution[x] for x in np.arange(0, len(basis))])
        coeffs2 = flatten([solution[x] for x in np.arange(len(basis), len(basis)*2)])

        # x = BSpline(basis, coeffs1)
        # y = BSpline(basis, coeffs2)
        
        # # Plotting the trajectory
        # x_trajectory = [x(t_)[0] + self.fp.fx_spline(t_)[0][0] for t_ in np.linspace(0, t, 100)]
        # y_trajectory = [y(t_)[0] + self.fp.fy_spline(t_)[0][0] for t_ in np.linspace(0, t, 100)]
        # ax.plot(x_trajectory, y_trajectory, 'k')
        
        
        
        # Creating the splines
        # basis = self.define_knots(degree = 3, knot_intervals = self.knot_intervals)
        # solution = self.solution['x'].full()
        # coeffs1 = flatten([solution[x] for x in np.arange(0, len(basis))])
        # coeffs2 = flatten([solution[x] for x in np.arange(len(basis), len(basis)*2)])

        p_solution = BSpline(basis, coeffs1)
        q_solution = BSpline(basis, coeffs2)
        
        # Sampling
        # t = np.linspace(0, 1, 100)
        
        x_t, y_t = [], []
        for t_ in np.linspace(0, t):
            p_solution_, q_solution_ = p_solution(t_)[0], q_solution(t_)[0]
            x_, y_ = self.fp.frenet_to_inertial(p_solution_, q_solution_, t_)
            x_t += [x_]
            y_t += [y_]
            
        ax.plot(x_t, y_t, 'k')    
            

        theta_c = 0
        theta_c = self.fp.fy_d_spline(t)[0][0] / self.fp.fx_d_spline(t)[0][0]

        # Draw body
        radious = 0.05
        radious = self.radious * 0.5 * 0.9
        height = radious
        width = radious


        # Draw arms
        l = 0.1
        l = radious * 2


        # x0 = x(t) + self.fp.fx_spline(t)
        # y0 = y(t) + self.fp.fy_spline(t)
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