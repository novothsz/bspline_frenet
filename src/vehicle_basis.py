import numpy as np
from numpy import interp
import math

from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import csv
import time
import pickle
import random



from casadi import MX, SX, Function, vertcat, dot, nlpsol, cos, sin, norm_2
from .spline import BSpline, BSplineBasis
from .spline_extra import definite_integral, shift_spline, shift_knot1_fwd, shift_knot1_bwd, shift_over_knot, extrapolate
from .environment import Environment


        


class VehicleBasis(Environment):
    def __init__(self):
        super().__init__()
        
        self.stage = []
        self.n_dimensions = 3 # this is considering a third state, the phi rotation angle
        self.state_len = 6 
        
        
        self.n_dimensions_old = 2
        self.state_len_old = self.n_dimensions_old * 2 # [position_x, velocity_x] * 2 if we are in 2D
        
        
        self.state_degree = 3
        self.radious = 0.08
        self.T = 5

    
        self.x0 = []
        self.xf = []
        self.x_reference = [] # this value gives the reference position of the vehicle in the formation
        # We need this value, because when mooving using an MPC the xf positions might change.
        # This is only a problem, because the formation constraint says, that the formation
        # has to be kept. The formation can scale and rotate freely. The rotation is given as a spline.
        # If we abrupty change the reference, then this spline will also has to change abruptly.
        # Because it is a negotiated variable, it would lead to poor results. So let's not do that, and have a 
        # fix x_reference_value. Actually... We don't need an x_reference. It is enough to hard-set this vector in the solver
        # and that is it :)
        # Oh, wow! It is already hard-set :))
        self.neighbours = []
        self.obstacles = []
        self.ID = -1
        self.iteration_count = 0

        # Temporary containers
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []

        self.P0_assemble = []
        self.P0_z_assemble = []


        # Test hyperparam
        self.slack = 0.00001
        # Hyperparams
        self.rho = 50# /5 # /50
        self.rho_formation = 100# /5 # /50
        self.rho_input = 0.1 * 10 * 2
        self.rho_final_value = 0.1 * 10000
        
        self.epsilon = 0.001 # try to keep minimum epsilon distance from the obstacle
        # self.epsilon = self.radious # try to keep minimum epsilon distance from the obstacle
        self.safety_weight = 1 # cost parameter for epsilon
        self.knot_intervals = 5 # number of knots for the output (position) spline of the vehicle
        self.t_resolution_length = 6
        self.t_resolution_length = self.knot_intervals + 1
        
        self.obstacle_avoidance_multiplier = 1.5
        self.vehicle_avoidnce_multiplier = 2.0
        
        # Constraints on decision variables
        # self.y_min = [self.border_x[0], self.border_y[0]]
        # self.y_max = [self.border_x[1], self.border_y[1]]
        self.y_min = [-1, -1, -math.pi]
        self.y_max = [1, 1, math.pi]
        
        
        self.u_min = [-50, -50]
        self.u_max = [50, 50]
        self.u_min = [-250, -250]
        self.u_max = [250, 250]
        
        self.options = {'print_time': False, 'ipopt': {'print_level' : 0, 'max_iter': 1000, 'max_cpu_time': 100}}
        self.options_z = {'print_time': False, 'ipopt': {'print_level' : 0, 'max_iter': 1000, 'max_cpu_time': 100}}

        self.initial_values = {}
        # variable_history
        self.variable_history =  {'y' : [],         # x_update
                                  'y_j' : [],       # data_exchange_x_receive
                                  't_start' : [],
                                  't_end' : []
                }
        self.n_intermediate_ADMM = 1
        # message
        self.message_in = {}
        self.message_out = {}
        
        # for spline fitting
        
        
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
    
    

    def set_position(self, position : list, position_type : str):
        # 2D
        if position_type == 'initial':
            self.x0 = position + [0, 0, 0]
        elif position_type == 'final':
            self.xf = position + [0, 0, 0]
        else:
            raise NotImplementedError()

        return self
    
    
        
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################

    def update_PvZ(self):
        """Updating P0_z parameter. Values, that are commented out are not
        currently updated. This can be changed later allowing additional functionality.
        Updated values are: x_i, x_j
        """
        self.PvZ.y = self.DvX.y
        try:
            self.PvZ.y_j = self.message_in['y_j']
        except:
            pass

        return self

    def update_PvX(self):
        """Updating P0 parameter. Values, that are commented out are not
        currently updated. This can be changed later allowing additional functionality.
        Updated values are: T, x0, z_i, z_ji, lambda_ji
        """
        self.PvX.z_i = self.DvZ.z_i
        self.PvX.x0 = self.x0
        self.PvX.xf = self.xf
        
        # updating values because we are following the mooving Frenet-frame
        t_evaluation = np.linspace(self.t_start, self.t_start + self.t_window_size, self.t_resolution_length)
        self.PvX.v_s = [self.fp.fx_d_spline(t_).tolist()[0][0] + self.fp.fy_d_spline(t_).tolist()[0][0] for t_ in t_evaluation]
        self.PvX.curvature = [self.fp.fy_c_spline(t_).tolist()[0][0] for t_ in t_evaluation]
        self.PvX.equation_min_p = [self.fp.equation_min_p(t_).tolist()[0][0] for t_ in t_evaluation]
        self.PvX.equation_max_p = [self.fp.equation_max_p(t_).tolist()[0][0] for t_ in t_evaluation]
        self.PvX.equation_min_q = [self.fp.equation_min_q(t_).tolist()[0][0] for t_ in t_evaluation]
        self.PvX.equation_max_q = [self.fp.equation_max_q(t_).tolist()[0][0] for t_ in t_evaluation]
        self.PvX.obst = []
        for obstacle in self.obstacles:
            for t_ in t_evaluation:
                for corner in obstacle.corners_spline:
                    self.PvX.obst += [corner[0](t_).tolist()[0][0], corner[1](t_).tolist()[0][0]]
            
        try:
            self.PvX.z_ji = self.message_in['z_ji']
            self.PvX.lambda_ji = self.message_in['lambda_ji']
            
        except:
            pass

        return self
    
    def shift_DvZ(self):
        # shift DvZ comes first, then shift_PvZ! Same for x!
        # Values, that need to be shifted: z_i, z_ij
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        
        
        z_i_coeffs_shifted = []
        for i in range(3):
            idx = np.arange(len(basis)*i,len(basis)*i+len(basis)) # 4 values, step by step
            z_i_coeffs_shifted += shift_spline(self.DvZ.z_i[idx[0]:idx[-1]+1], self.t_step, basis).tolist()
        self.DvZ.z_i = z_i_coeffs_shifted
        
        
        z_ij_coeffs_shifted = []
        for i in range(len(self.neighbours) * 3):
            idx = np.arange(len(basis)*i,len(basis)*i+len(basis)) # 4 values, step by step
            z_ij_coeffs_shifted += shift_spline(self.DvZ.z_ij[idx[0]:idx[-1]+1], self.t_step, basis).tolist()
        self.DvZ.z_ij = z_ij_coeffs_shifted
        
        return self
        
    
    def shift_PvZ(self):
        # Values, that need to be shifted are:
        # y, y_j, lambda_i, lambda_ij
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        
        # shifting y, lambda_i NO!!!!
        # This is already shifted!!!! Don't shift again
        
        # shifting lambda_i as the shifted version, which is found in self.DvX ....
        self.PvX.lambda_i = self.PvX.lambda_i
        
        # shifting lambda_ij
        lambda_ij_coeffs_shifted = []
        for i in range(len(self.neighbours) * 3):
            idx = np.arange(len(basis)*i,len(basis)*i+len(basis)) # 4 values, step by step
            lambda_ij_coeffs_shifted += shift_spline(self.PvZ.lambda_ij[idx[0]:idx[-1]+1], self.t_step, basis).tolist()
        self.PvZ.lambda_ij = lambda_ij_coeffs_shifted
        
        return self

    def shift_PvX(self):
        # Values, which are not being shifted, are:
        # v_s, curvature, equation_min/max_p/q -> these are re-evaluated according the current time-window inside update_PvX
        # Values, which need to be shifted, are:
        # x0, z_i, z_ji, lambda_i, lambda_ji
        
        # x0
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        # coeffs1 = self.DvX.y[0:len(basis)]
        # coeffs2 = self.DvX.y[len(basis):len(basis)*2]
        sol = self.solution['x'].full().reshape(-1).tolist()
        coeffs1 = sol[0:len(basis)]
        coeffs2 = sol[len(basis):len(basis)*2]
        coeffs3 = sol[len(basis)*2:len(basis)*3]
        p = BSpline(basis, coeffs1)
        q = BSpline(basis, coeffs2)
        phi = BSpline(basis, coeffs3)
        # p_ = [p(t_).tolist()[0] for t_ in np.linspace(0, 1, 100)]
        # q_ = [q(t_).tolist()[0] for t_ in np.linspace(0, 1, 100)]
        p0 = p(self.t_step*1/self.t_window_size).tolist()[0]
        q0 = q(self.t_step*1/self.t_window_size).tolist()[0]
        phi0 = phi(self.t_step*1/self.t_window_size).tolist()[0]
        p_dot0 = p.derivative()(self.t_step*1/self.t_window_size).tolist()[0]
        q_dot0 = q.derivative()(self.t_step*1/self.t_window_size).tolist()[0]
        phi_dot0 = phi.derivative()(self.t_step*1/self.t_window_size).tolist()[0]
        # updating x0 in PvX
        self.x0 = [p0, q0, phi0, p_dot0, q_dot0, phi_dot0]
        self.PvX.x0 = self.x0
        
        # shifting z_i, lambda_i
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        z_i_coeffs_shifted = []
        lambda_i_coeffs_shifted = []
        for i in range(3):
            idx = np.arange(len(basis)*i,len(basis)*i+len(basis)) # 4 values, step by step
            z_i_coeffs_shifted += shift_spline(self.PvX.z_i[idx[0]:idx[-1]+1], self.t_step, basis).tolist()
            lambda_i_coeffs_shifted += shift_spline(self.PvX.lambda_i[idx[0]:idx[-1]+1], self.t_step, basis).tolist()
        self.PvX.z_i = z_i_coeffs_shifted
        self.PvX.lambda_i = lambda_i_coeffs_shifted
        
        
        # shifting z_i, lambda_i
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        z_ji_coeffs_shifted = []
        lambda_ji_coeffs_shifted = []
        for i in range(len(self.neighbours) * 3):
            idx = np.arange(len(basis)*i,len(basis)*i+len(basis)) # 4 values, step by step
            z_ji_coeffs_shifted += shift_spline(self.PvX.z_ji[idx[0]:idx[-1]+1], self.t_step, basis).tolist()
            lambda_ji_coeffs_shifted += shift_spline(self.PvX.lambda_ji[idx[0]:idx[-1]+1], self.t_step, basis).tolist()
        self.PvX.z_ji = z_ji_coeffs_shifted
        self.PvX.lambda_ji = lambda_ji_coeffs_shifted
        
        # Also, we need to update some other parameters
        
        
        
        
        
    def shift_DvX(self):
        # shifting y
        basis_y = self.define_knots(degree = 3, knot_intervals = self.knot_intervals)
        
        coeffs1 = self.DvX.y[0:len(basis_y)]
        coeffs2 = self.DvX.y[len(basis_y):len(basis_y)*2]
        coeffs3 = self.DvX.y[len(basis_y)*2:len(basis_y)*3]
        coeffs1 = shift_spline(coeffs1, self.t_step, basis_y).tolist()
        coeffs2 = shift_spline(coeffs2, self.t_step, basis_y).tolist()
        coeffs3 = shift_spline(coeffs3, self.t_step, basis_y).tolist()
        self.DvX.y = coeffs1
        self.DvX.y += coeffs2
        self.DvX.y += coeffs3
        
        # shifting a
        basis_a = self.define_knots(degree = 1, knot_intervals = self.knot_intervals)
        a_coeffs_shifted = []
        for i in range(len(self.obstacles) * 2): # * 2, because a is 2 dimensional 
            idx = np.arange(len(basis_a)*i,len(basis_a)*i+len(basis_a)) # 4 values, step by step
            a_coeffs_shifted += shift_spline(self.DvX.a[idx[0]:idx[-1]+1], self.t_step, basis_a).tolist()
        self.DvX.a = a_coeffs_shifted
        
        # shifting b, d_tau
        b_coeffs_shifted = []
        d_tau_coeffs_shifted = []
        for i in range(len(self.obstacles)):
            idx = np.arange(len(basis_a)*i,len(basis_a)*i+len(basis_a)) # 4 values, step by step
            b_coeffs_shifted += shift_spline(self.DvX.b[idx[0]:idx[-1]+1], self.t_step, basis_a).tolist()
            d_tau_coeffs_shifted += shift_spline(self.DvX.d_tau[idx[0]:idx[-1]+1], self.t_step, basis_a).tolist()
        self.DvX.b = b_coeffs_shifted
        self.DvX.d_tau = d_tau_coeffs_shifted
    
    
        
    
    def simulation_step(self):
        # self.t_step = 0.01
        self.t_start = self.t_start + self.t_step
        self.t_end = self.t_start + self.t_window_size
        
        self.shift_enabled = True
        
        return self
                
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    "Data exchange"
    def data_exchange_z_send(self):
        """Sharing lambda_ij, z_ij data

        Parameters
        ----------
        none

        Returns
        message : list
            form: [{'sender' : senderID, 'receiver': receiverID, 'data_name': data}]
        -------
        """
        lambda_ij = np.array(self.PvZ.lambda_ij).reshape((len(self.neighbours), -1))
        z_ij = np.array(self.DvZ.z_ij).reshape((len(self.neighbours), -1))

        message = []
        for i in range(len(self.neighbours)):
            message += [{'sender': self.ID, 'receiver': self.neighbours[i].ID, 'lambda_ij': lambda_ij[i, :].tolist(), \
                                      'z_ij': z_ij[i, :].tolist()} ]

        return message

    def data_exchange_z_receive(self, message):
        """Receiving message to update: lambda_ji and z_ji values
        incoming: lambda_ij -> updated: lambda_ji
        incoming: z_ij -> updated: z_ji

        Parameters
        ----------
        message : list
            form: [{'sender' : senderID, 'receiver': receiverID, 'data_name': data}]

        Returns
        self
        -------
        """

        lambda_ji = []
        z_ji = []

        for neighbour_ in self.neighbours:
            for dict_ in message:
                if neighbour_.ID == dict_['sender'] and self.ID == dict_['receiver']:
                    lambda_ji += dict_['lambda_ij']
                    z_ji += dict_['z_ij']

        self.message_in['lambda_ji'] = lambda_ji
        self.message_in['z_ji'] = z_ji

        return self


    def data_exchange_x_send(self):
        """Sharing our own x_i data
        receiverID = -1, because we send it to everyone. In the receive function,
        every agent ignores messages that are not sent from neighboring agents.

        Parameters
        ----------
        none

        Returns
        message : list
            form: [{'sender' : senderID, 'receiver': receiverID = -1, 'data_name': data}]
        -------
        """
        message = [{'sender': self.ID, 'receiver': -1, 'y_i': self.DvX.y}] # the first 4 belongs to x0. We don't optimize for that in the z update

        return message


    def data_exchange_x_receive(self, message):
        """Receiving message to update: x_j values
        receiverID = -1, because x_j is sent to everyone. We ignore messages
        that are not sent from neighboring agents.

        Parameters
        ----------
        message : list
            form: [{'sender' : senderID, 'receiver': receiverID = -1, 'data_name': data}]

        Returns
        self
        -------
        """
        y_j = []
        for neighbour_ in self.neighbours:
            for dict_ in message:
                if neighbour_.ID == dict_['sender']:
                    y_j += dict_['y_i']

        self.message_in['y_j'] = y_j

        # Saving some variables
        self.variable_history['y_j'] += [y_j]
        return self

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    def lambda_update(self):
        """Updating lambda_i and lambda_ij values (in P0 and P0_z)"""
        lambda_i_old = np.array(self.PvZ.lambda_i)
        # lambda_i_new = lambda_i_old * 0
        y = np.array(self.DvX.y)
        z_i = np.array(self.DvZ.z_i)

        lambda_ij_old = np.array(self.PvZ.lambda_ij)
        # lambda_ij_new = lambda_ij_old * 0
        y_j = np.array(self.message_in['y_j'])
        z_ij = np.array(self.DvZ.z_ij)

        lambda_i_new = lambda_i_old + self.rho * (y - z_i)
        lambda_ij_new = lambda_ij_old + self.rho * (y_j - z_ij)

        self.PvX.lambda_i = lambda_i_new.tolist()
        self.PvZ.lambda_i = lambda_i_new.tolist()
        self.PvZ.lambda_ij = lambda_ij_new.tolist()

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    "Preparation"
    
    def prepare0(self):
        "TODO"

    def prepare1(self):
        self.setup_x_update()
        self.setup_z_update()

    def prepare2(self):
        "TODO"

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    "Spline related functions"
    def define_knots(self, degree = 3, **kwargs):
        """This function defines the knots and creates the
        B-spline basis function with the prescribed degree.
        Input:
            degree: degree of the B-spline basis functions
            knot_intervals: number of knot intervals
            knots (optional): knot vector. If not given, calculated using the
            number of knot_intervals
        Returns:
            basis: array of B-spline basis functions
            knots: the knot vector
            knot_intervals
        """

        if 'knot_intervals' in kwargs:
            knot_intervals = kwargs['knot_intervals']
            knots = np.r_[np.zeros(degree),
                          np.linspace(0, 1, knot_intervals+1),
                          np.ones(degree)]
        if 'knots' in kwargs:
            knot_intervals = len(knots) - 2*degree - 1
            knots = kwargs['knots']

        basis = BSplineBasis(knots, degree)

        return basis

    def define_MX_spline(self, degree, knot_intervals, n_spl, lower_bound, upper_bound, initial_value = [], name = '', category = 'variable'):
        """This function defines a set of splines.
        Input:
            basis: the basis function to define the spline with. If not provided,
            self.basis will be used.
            n_spl: number of splines to define
        Returns:
            a B-spline class
        """
        basis = self.define_knots(degree = degree, knot_intervals = knot_intervals)

        splines = []
        for k in range(n_spl):
            coeffs = MX.sym(name[k], len(basis))
            if category == 'variable':
                self.w += [coeffs]
                self.w_list += [name[k] for i in range(len(basis))]
                # if initial_value is not an empty list
                """initial value can be: empty, a list of starting and final values and a list of coefficients
                empty: we initialize it with uniform random variables
                list of starting and final values: we initialize it with np.linspace and adding some noise to it.
                    If any of the starting or final values for the given spline is None, we also initialize it with
                    uniform random variables.
                list of coefficients: we initialize it with the coefficients provided
                """
                if initial_value:
                    if len(initial_value[k]) == len(basis):
                        self.w0 += initial_value[k]
                        # assert True == print("Lefutott LOL")
                    elif initial_value[k][0] is not None or initial_value[k][1] is not None:
                        # self.w0 += np.linspace(initial_value[k][0], initial_value[k][1], len(basis)).tolist()
                        w0_noise_added = np.linspace(initial_value[k][0], initial_value[k][1], len(basis))
                        import random
                        for i in range(len(w0_noise_added)):
                            w0_noise_added[i] = w0_noise_added[i] + w0_noise_added[i] * 0.05 * random.uniform(-0.5, 0.5)

                        self.w0 += w0_noise_added.tolist()
                    else:
                        # self.w0 += np.zeros((1, len(basis)))[0].tolist()
                        import random
                        self.w0 += [random.uniform(-0.5, 0.5) for i in range(len(basis))]

                else:
                    # self.w0 += np.zeros((1, len(basis)))[0].tolist()
                    import random
                    self.w0 += [random.uniform(-0.5, 0.5) for i in range(len(basis))]

            elif category == 'parameter':
                self.P += [coeffs]
                self.P_list += [name[k] for i in range(len(basis))]
                self.P0 += [0 for i in range(len(basis))]
            else:
                raise NotImplementedError()
            splines += [BSpline(basis, coeffs)]


        for i in range(len(splines)):
            for j in range(splines[i].coeffs.shape[0]):
                if category == 'variable':
                    self.lbw += [lower_bound[i]]
                    self.ubw += [upper_bound[i]]

        return np.array(splines)

    def define_constraint(self, constraint, lower_bound, upper_bound, constraint_type = 'overall', name = ''):
        """This function defines constraint on the b_spline coefficients
        Input:
            constraint (list): a spline constraint. We will set upper and lower bounds on its coefficients
            lower_bound (list): the lower bound of the coefficients
            upper_bound (list): the upper bound of the coefficients
            constraint_type: 'overall', 'initial', 'final'
        Returns:

        """
        if constraint_type == 'overall':
            for i in range(len(constraint)):
                for j in range(constraint[i].coeffs.shape[0]):
                    self.g += [constraint[i].coeffs[j]]
                    self.g_list += [name]
                    self.lbg += [lower_bound[i]]
                    self.ubg += [upper_bound[i]]
            return self

        elif constraint_type == 'time':
            for i in range(len(constraint)):
                # for j in range(constraint[i].coeffs.shape[0]):
                self.g += [constraint[i]]
                for j in range(constraint[i].shape[0]):
                    self.g_list += [name]
                    self.lbg += [lower_bound[i]]
                    self.ubg += [upper_bound[i]]
            return self

        elif constraint_type == 'initial':
            for i in range(len(constraint)):
                self.g += [constraint[i].coeffs[0]] # we restrict the first coefficient
                self.g_list += [name[i]]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self

        elif constraint_type == 'final':
            for i in range(len(constraint)):
                self.g += [constraint[i].coeffs[-1]] # we restrict the last coefficient
                self.g_list += [name[i]]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self

        # If we want to equal the first/last coefficient to a parameter value
        # we have to bring the MX type parameter to the constraint and set
        # lbg&ubg to 0
        elif constraint_type == 'initial_param':
            for i in range(lower_bound.shape[0]):
                self.g += [constraint[i].coeffs[0] - lower_bound[i]] # we restrict the first coefficient
                self.g_list += [name[i]]
                self.lbg += [0]
                self.ubg += [0]
            return self

        elif constraint_type == 'final_param':
            for i in range(lower_bound.shape[0]):
                self.g += [constraint[i].coeffs[-1] - lower_bound[i]] # we restrict the last coefficient
                self.g_list += [name[i]]
                self.lbg += [0]
                self.ubg += [0]
            return self

        else:
            raise NotImplementedError()

    def collision_avoidance_circular(self, splines, center, radious, name):
        """This function defines constraints on the splines to avoid the space arodund
        a certaint point with a given radious.
        Input:
            splines (list): a spline on which we want to set final constraint
            center: the point to avoid
            radious: the minimum distance from the center point
        Returns:

        """

        constraint = (splines[0] - center[0])**2 + (splines[1] - center[1])**2
        self.define_constraint([constraint], lower_bound = [radious**2], upper_bound = [math.inf], name = name)

    def collision_avoidance_hyperplane(self, splines, points, radious, name, constraint_type='obstacle', n_samples = 10):
        """Collision avoidance using the separating hyperplane theorem"""

        # a
        # a points towards the obstacle. This means, that if we use
        # initial_value = [[1, -1], [0.1, 0.1]]
        # the vehicle will try to avoid the obstacle from the bottom direction (because
        # a always points "up" a little bit, it's y coordinate is 0.1)
        a = self.define_MX_spline(degree = 1, knot_intervals = self.knot_intervals, n_spl = len(splines),
                                lower_bound = [-math.inf] * len(splines),
                                upper_bound = [math.inf] * len(splines),
                                initial_value = [[1, -1], [0, 0]],
                                # name = ["a"+ str(i) for i in range(len(splines))])
                                name = ["a"] * self.n_dimensions_old)

        # b
        b = self.define_MX_spline(degree = 1, knot_intervals = self.knot_intervals, n_spl = 1,
                                lower_bound = [-math.inf],
                                upper_bound = [math.inf],
                                # initial_value = [[0, 0]],
                                name = ["b"])
        # initial_value = [[self.x0[0], self.xf[0]], [self.x0[1], self.xf[1]]],
        # d_tau
        d_tau = self.define_MX_spline(degree = 1, knot_intervals = self.knot_intervals, n_spl = 1,
                                lower_bound = [0],
                                upper_bound = [math.inf],
                                name = ["d_tau"])


        # ---- Constraint 1
        const1 = a[0]*splines[0] + a[1]*splines[1] - b[0]
        self.define_constraint([const1], lower_bound = [-math.inf], upper_bound = [-radious], name = "eq1")

        # ---- Constraint 2 (various versions)
        const2 = []
        if constraint_type == 'obstacle':
            for point in points:
                const2 += [a[0] * point[0] + a[1] * point[1] - b[0]  - d_tau[0] ]
            # ----
            for i in range(len(points)):
                self.define_constraint([const2[i]], lower_bound = [0], upper_bound = [math.inf], name = "eq2" + "_corn_" + str(i))

        elif constraint_type == 'spline_obstacle_t':
            t = np.linspace(0, 1, n_samples)
            for t_ in t: # 100
                for point in points: # 4
                    # const2 += [a[0](t) * point[0](t) + a[1](t) * point[1](t) - b[0](t)]
                    const2 += [a[0](t) * point[0] + a[1](t) * point[1] - b[0](t)  - d_tau[0](t) ]

            # ----
            for i in range(len(const2)):
                self.define_constraint([const2[i]], lower_bound = [0], upper_bound = [math.inf], constraint_type = "time", name = "eq2" + "_corn_" + str(i))


        elif constraint_type == 'spline_obstacle_spline_t':
            t = np.linspace(0, 1, n_samples)
            for t_ in t:
                for point in points:
                    # const2 += [a[0](t) * point[0](t) + a[1](t) * point[1](t) - b[0](t)]
                    const2 += [a[0](t) * point[0](t) + a[1](t) * point[1](t) - b[0](t)  - d_tau[0](t) ]

            # ----
            for i in range(len(const2)):
                self.define_constraint([const2[i]], lower_bound = [0], upper_bound = [math.inf], constraint_type = "time", name = "eq2" + "_corn_" + str(i))

        # Dude, these namings... Oh well, whatever...
        elif constraint_type == 'spline_obstacle_param':
            t = np.linspace(0, 1, n_samples)
            for t_, points_t in zip(t, points): # 100
                for point in points_t: # 4
                        # const2 += [a[0](t) * point[0](t) + a[1](t) * point[1](t) - b[0](t)]
                        const2 += [a[0](t_) * point[0] + a[1](t_) * point[1] - b[0](t_)  - d_tau[0](t_) ]

            # ----
            for i in range(len(const2)):
                self.define_constraint([const2[i]], lower_bound = [0], upper_bound = [math.inf], constraint_type = "time", name = "eq2" + "_corn_" + str(i))


        elif constraint_type == 'inter_vehicle':
            for i in range(points[0].coeffs.shape[0]):
                const2 += [a[0] * points[0].coeffs[i] + a[1] * points[1].coeffs[i] - b[0]  - d_tau[0] ]

            # ----
            for i in range(points[0].coeffs.shape[0]):
                self.define_constraint([const2[i]], lower_bound = [0], upper_bound = [math.inf], name = "eq2" + "_" + str(i))

        elif constraint_type == 'inter_vehicle_t':
            t = np.linspace(0, 1, n_samples)
            for t_ in t:
                const2 += [a[0](t) * points[0](t) + a[1](t) * points[1](t) - b[0](t)  - d_tau[0](t) ]

            # ----
            for i in range(len(const2)):
                self.define_constraint([const2[i]], lower_bound = [0], upper_bound = [math.inf], constraint_type = "time", name = "eq2" + "_" + str(i))
        else:
            raise NotImplementedError()


        # ---- Constraint 3
        const3 = a[0] * a[0] + a[1] * a[1]
        self.define_constraint([const3], lower_bound = [0.0], upper_bound = [1.0], name = "eq3")

        self.J += self.safety_weight * definite_integral((self.epsilon - d_tau[0])**2, 0, 1)

        return self



    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    "Writing data"
    def write_csv_for_stage(self, T, px, py):
        """This function writes a csv for only the specific stage.
        It won't include any other stages, but can be used later to assemble
        trajectories for all stages."""
        
        pz = [0] * len(px[0])
        pj = [0] * len(px[0])
        first_line = ['duration', 'x^0', 'x^1', 'x^2', 'x^3', 'x^4', 'x^5', 'x^6', 'x^7', 'y^0', 'y^1', 'y^2', 'y^3', 'y^4', 'y^5', 'y^6', 'y^7', 'z^0', 'z^1', 'z^2', 'z^3', 'z^4', 'z^5', 'z^6', 'z^7', 'yaw^0', 'yaw^1', 'yaw^2', 'yaw^3', 'yaw^4', 'yaw^5', 'yaw^6', 'yaw^7']
        mode = 'w'
        with open(self.cwd + '/csv/' + 'stage_' + str(self.stage) + '_vehicle' + str(self.ID) + '.csv', mode = mode) as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(first_line)
            for i in range(len(T)):
                writer.writerow(T[i] + px[i] + py[i] + pz + pj)
                
        return self
    
    def write_csv(self, T, px, py):
        """This function writes the provided time and 7d polinome coefficients [each as lists]
        to a csv file, so that the crazyswarm code can upload the data to the drones.
        OR actually, don't really remember, what is does :D :S. Read the comments pls.
        """
        
        pz = [0] * len(px[0])
        pj = [0] * len(px[0])

        # Writing this first line is required by crazyswarm
        first_line = ['duration', 'x^0', 'x^1', 'x^2', 'x^3', 'x^4', 'x^5', 'x^6', 'x^7', 'y^0', 'y^1', 'y^2', 'y^3', 'y^4', 'y^5', 'y^6', 'y^7', 'z^0', 'z^1', 'z^2', 'z^3', 'z^4', 'z^5', 'z^6', 'z^7', 'yaw^0', 'yaw^1', 'yaw^2', 'yaw^3', 'yaw^4', 'yaw^5', 'yaw^6', 'yaw^7']

        # If we are at the first stage, we will create a new csv file
        append_row = []
        if self.stage == 0:
            mode = 'w'
        # If we have already completed the second stage, we will add new rows to the csv file.
        # For this reason, first we:
        # 1: read in the lines.
        # 2: store them in append_row and swith back to writing mode
        else:
            mode = 'r'
            with open(self.cwd + '/csv/' + 'vehicle' + str(self.ID) + '.csv', mode = mode) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    append_row += [row]
            mode = 'w'

        with open(self.cwd + '/csv/' + 'vehicle' + str(self.ID) + '.csv', mode = mode) as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # 3: write the first line
            if self.stage == 0:
                writer.writerow(first_line)
        # 4: OR write the lines associated with previous stages in apend-row
        # (basically except the last two lines)
            for i, row in enumerate(append_row):
                if i < self.stage*2 + 1:
                    writer.writerow(row)
        # 5: because instead of the last two lines we will write
        # the polynome values belonging to the most recent solution.
            for i in range(len(T)):
                writer.writerow(T[i] + px[i] + py[i] + pz + pj)
        return self
    
    

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################

    # ========================================================================
    # Methods required to override
    # ========================================================================

    def setup_z_update(self):
        raise NotImplementedError('Please implement this method!')

    def setup_x_update(self):
        raise NotImplementedError('Please implement this method!')

    def z_update(self):
        raise NotImplementedError('Please implement this method!')

    def x_update(self):
        raise NotImplementedError('Please implement this method!')
        
    
        
    def initialize_values(self):
        """This function initializes saves some values into a dictionary.
        These values will be used to initialize all decision variables & parameters
        for the x & z update respectively. (in setup_x_update() and setup_z_update() )"""
        
        self.initial_values["y"] = self.DvX.y
        self.initial_values["z_i"] = self.DvX.y
        self.initial_values["z_ji"] = self.DvX.y * len(self.neighbours)
        self.initial_values["y_j"] = self.message_in["y_j"]
        self.initial_values["z_j"] = self.message_in["y_j"]
        self.initial_values["z_ij"] = self.message_in["y_j"]
        self.initial_values["lambda_i"] = [1] * len(self.DvX.y)
        self.initial_values["lambda_ij"] = [1] * len(self.DvX.y) * len(self.neighbours)
        self.initial_values["lambda_ji"] = [1] * len(self.DvX.y) * len(self.neighbours)
        
        # Setting back every variable that belong to the ADMM iterations
        # (this step is actually not necessary)
        self.DvX = []
        self.message_in = {}
        self.variable_history =  {'y' : [],         # x_update
                                  'y_j' : [],       # data_exchange_x_receive
                                  't_start' : [],
                                  't_end' : []
        }
        
    def initialize_x(self):
        # tmp_obstacles = self.obstacles
        # self.obstacles = []
        tmp_neighbours = self.neighbours
        self.neighbours = []
        tmp_t_resolution_length = self.t_resolution_length
        self.t_resolution_length = 30
        self.setup_x_update()
        self.setup_z_update()
        self.x_update()
        self.initial_values["w0_initial"] = self.solution['x']# .full().reshape(1, -1).tolist()[0]
        self.initial_values["DvX"] = self.DvX
        # self.obstacles = tmp_obstacles
        self.neighbours = tmp_neighbours
        self.t_resolution_length = tmp_t_resolution_length
        print("Should be working, but please implement this method properly")
        # raise NotImplementedError('Please implement this method!')








