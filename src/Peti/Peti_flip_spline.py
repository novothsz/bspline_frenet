from casadi import MX, SX, vertcat, dot, nlpsol
from spline import BSpline, BSplineBasis
from spline_extra import definite_integral
import math
from Peti_param import DecisionVarX
import matplotlib.pyplot as plt
import numpy as np

class Flipper():
    def __init__(self):
        print("I am a happy Flipper :)")
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []
        
        # self.state_degree = 3
        self.knot_intervals = 30
        self.nt = 100
        self.slack = 0.00
        self.zero_slack = 0.0
        self.gravity = np.array([0, 0, 9.81])
        # self.n_dimensions = 1
        
        "r"
        self.r_min = [-math.inf, -math.inf, -math.inf]
        self.r_max = [ math.inf,  math.inf,  math.inf]
        self.r0 = [0, 0, 0]
        self.rf = [0, 0, 1]
        
        "q"
        self.q_min = [-1, -self.zero_slack, -1, -self.zero_slack]
        self.q_max = [ 1, self.zero_slack, 1, self.zero_slack]
        self.q_min = [-1, -1, -1, -1]
        self.q_max = [ 1, 1, 1, 1]
        self.q0 = [1, 0, 0, 0]
        self.qf = [1, 0, 0, 0]
        
        "omega"
        self.v_min = -1400 / 360 * 2 * math.pi
        self.v_max = 1400 / 360 * 2 * math.pi
        self.omega_min = [self.v_min, -self.zero_slack, -self.zero_slack]
        self.omega_max = [self.v_max,  self.zero_slack, self.zero_slack]
        self.omega_min = [self.v_min, self.v_min, self.v_min]
        self.omega_max = [self.v_max, self.v_max, self.v_max]
        self.omega0 = [0, 0, 0]
        self.omegaf = [0, 0, 0]
        
        
        "F"
        self.F_min = [0, 0, -math.inf]
        self.F_max = [0, 0,  math.inf]
        
        "tau"
        self.u_min = -0.16 * 2 * 0.034
        self.u_max =  0.16 * 2 * 0.034
        self.tau_min = [self.u_min, -self.zero_slack, -self.zero_slack]
        self.tau_max = [self.u_max, self.zero_slack, self.zero_slack]
        # self.tau_min = [-math.inf, 0, 0]
        # self.tau_max = [math.inf, 0, 0]
        
        
        self.m = 0.027 # weight of cf_v2
        
        self.J11 = 1.4 * 1e-5
        self.J22 = 1.4 * 1e-5
        self.J33 = 2.17 * 1e-5
        self.I = np.eye(3) * np.array([self.J11, self.J22, self.J33]) #inertia
        
        
        self.arg = {}
        self.DvX = []
        
        # self.T = 0.2 + 0.18 + 0.2
        
        
    def setup_solver(self):
        
        # Decision variables
        # ---- States ---- #
        "r, dr, ddr (derivative)"
        r = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.r_min, upper_bound=self.r_max,
                                name=["r"] * 3)
        # initial_value = [[self.x0[0], self.xf[0]], [self.x0[1], self.xf[1]]],
        
        dr = np.array([r_.derivative() for r_ in r])
        ddr = np.array([dr_.derivative() for dr_ in dr])
        
        "q"
        q = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=4,
                                lower_bound=self.q_min, upper_bound=self.q_max,
                                name=["q"] * 4)
        
        # Let's quickly add some initial values to q.
        # import os
        my_data = np.load('/Users/szilard/Dropbox/Sztaki/code/code_examples/bspline_static/src/trajs.npy')
        q0 = my_data['q0']
        q2 = my_data['q1']
        
        # Downsampling q0
        idx_len = len(q0)
        idx_new = np.linspace(0, idx_len-1, q[0].coeffs.shape[0], endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_q0 = [q0[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        
        # Downsampling q2
        idx_len = len(q2)
        idx_new = np.linspace(0, idx_len-1, q[2].coeffs.shape[0], endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_q2 = [q2[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        # Initializing q with the values provided by Peti
        new_w0_for_q = downsampled_q0 + [0] * len(downsampled_q0) + downsampled_q2 + [0] * len(downsampled_q2)
        self.w0[-len(new_w0_for_q):] = new_w0_for_q
        dq = np.array([q_.derivative() for q_ in q])
        
        
        
        "omega"
        omega = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.omega_min, upper_bound=self.omega_max,
                                name=["omega"] * 3)
        domega = np.array([omega_.derivative() for omega_ in omega])
        
        # ---- Inputs ---- #
        "F"
        F = self.define_MX_spline(degree=0, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.F_min, upper_bound=self.F_max,
                                name=["F"] * 3)
        
        # Also, we may want to initialize F
        F3_0 = my_data['input'][0, :]
        # Downsampling F3_0
        idx_len = len(F3_0)
        idx_new = np.linspace(0, idx_len-1, F[0].coeffs.shape[0], endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_F3_0 = [F3_0[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        # Initializing q with the values provided by Peti
        new_w0_for_F3_0 = [0] * len(downsampled_F3_0) + [0] * len(downsampled_F3_0) + downsampled_F3_0
        self.w0[-len(new_w0_for_F3_0):] = new_w0_for_F3_0
        
        
        
        "tau"
        tau = self.define_MX_spline(degree=0, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.tau_min, upper_bound=self.tau_max,
                                name=["tau"] * 3)
        
        
        # Also, we may want to initialize tau
        tau_x_0 = my_data['input'][1, :]
        # Downsampling F3_0
        idx_len = len(tau_x_0)
        idx_new = np.linspace(0, idx_len-1, tau[0].coeffs.shape[0], endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_tau_x_0 = [tau_x_0[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        # Initializing q with the values provided by Peti
        new_w0_for_tau_x_0 = downsampled_tau_x_0 + [0] * len(downsampled_tau_x_0) + [0] * len(downsampled_tau_x_0)
        self.w0[-len(new_w0_for_tau_x_0):] = new_w0_for_tau_x_0
        
        # We only do this, because it will be useful in the plotter() function
        self.r, self.q, self.omega, self.F, self.tau = r, q, omega, F, tau
        
        "Dynamics"
        """ 
        "Sanity check"
        # Step 1: get data
        # Step 2: fit a spline to it
        # Step 3: get coeffs
        # Step 4: do the dynamics equations with these coeffs
        # Step 5: plot the result
        
        t_inp = my_data["time"]
        inp = my_data["input"]
        F3_inp = inp[0, :]
        F1_inp = np.zeros(len(F3_inp))
        F2_inp = np.zeros(len(F3_inp))
        F_inp = np.array([F1_inp, F2_inp, F3_inp])
        tau_inp = inp[1:, :]
        
        
        from frenet_spline import SplineFitter
        fitter = SplineFitter(knot_intervals = self.knot_intervals)
        F1_spline = fitter.fitting(t_inp, F1_inp, degree=0)[1]
        F2_spline = fitter.fitting(t_inp, F2_inp, degree=0)[1]
        F3_spline = fitter.fitting(t_inp, F3_inp, degree=0)[1]
        
        tau1_spline = fitter.fitting(t_inp, tau_inp[0, :], degree=0)[1]
        tau2_spline = fitter.fitting(t_inp, tau_inp[1, :], degree=0)[1]
        tau3_spline = fitter.fitting(t_inp, tau_inp[2, :], degree=0)[1]
        
        # t = np.linspace(0, 1, 100)
        
        # F1_spline_t = [F1_spline(t_) for t_ in t]
        # F2_spline_t = [F2_spline(t_) for t_ in t]
        # F3_spline_t = [F3_spline(t_) for t_ in t]
        
        # F1_spline_t = np.array(F1_spline_t).reshape(1, -1).tolist()[0]
        # F2_spline_t = np.array(F2_spline_t).reshape(1, -1).tolist()[0]
        # F3_spline_t = np.array(F3_spline_t).reshape(1, -1).tolist()[0]
        
        # plt.close("all")
        # plt.plot(F1_spline_t, ':')
        # plt.plot(F2_spline_t, ':')
        # plt.plot(F3_spline_t)
        # plt.show()
        
        
        
        
        # tau1_spline_t = [tau1_spline(t_) for t_ in t]
        # tau2_spline_t = [tau2_spline(t_) for t_ in t]
        # tau3_spline_t = [tau3_spline(t_) for t_ in t]
        
        # tau1_spline_t = np.array(tau1_spline_t).reshape(1, -1).tolist()[0]
        # tau2_spline_t = np.array(tau2_spline_t).reshape(1, -1).tolist()[0]
        # tau3_spline_t = np.array(tau3_spline_t).reshape(1, -1).tolist()[0]
        
        
        # plt.close("all")
        # plt.plot(tau1_spline_t)
        # plt.plot(tau2_spline_t, ':')
        # plt.plot(tau3_spline_t, ':')
        # plt.show()
        
        F[0].coeffs = F1_spline.coeffs
        F[1].coeffs = F2_spline.coeffs
        F[2].coeffs = F3_spline.coeffs
        tau[0].coeffs = tau1_spline.coeffs
        tau[1].coeffs = tau2_spline.coeffs
        tau[2].coeffs = tau3_spline.coeffs
        
        # Step 4: do the dynamics equations with these coeffs
        
        # hhh.... going to be painful...
        
        # omega integration
        omega_plus = [np.array([0, 0, 0])]
        omega_dot_plus = [np.array([0, 0, 0])]
        diff_t_inp = np.diff(t_inp)
        b_scaling = 0.1
        for i, dt in enumerate(diff_t_inp):
            tmp = np.dot(np.diag(np.array([2.2314, 2.2314, b_scaling])), tau_inp[:, i]) - np.cross(omega_plus[-1], np.dot(self.I, omega_plus[-1]))
            omega_dot_plus += [np.dot(np.linalg.inv(self.I), tmp)]
            omega_plus += [omega_plus[-1] + omega_dot_plus[-1] * dt]
            
            
        
        plt.close("all")
        plt.plot(np.array(omega_plus)[:, 0])
        plt.plot(np.array(omega_plus)[:, 1])
        plt.plot(np.array(omega_plus)[:, 2])
        plt.show()
        
        # q integration
        q_plus = [np.array([1, 0, 0, 0])]
        q_dot_plus = [np.array([0, 0, 0, 0])]
        for i, dt in enumerate(diff_t_inp):
            q_dot_plus += [ 0.5 * self.quaternion_product(omega_plus[i], q_plus[-1] ) ]
            q_plus += [q_plus[-1] + q_dot_plus[-1] * dt]
            
            
        
        plt.close("all")
        plt.plot(np.array(q_plus)[:, 0])
        plt.plot(np.array(q_plus)[:, 1])
        plt.plot(np.array(q_plus)[:, 2])
        plt.plot(np.array(q_plus)[:, 3])
        plt.show()
        
        # r integration
        r_plus = [np.array([0, 0, 0])]
        r_dot_plus = [np.array([0, 0, 0])]
        r_dotdot_plus = [np.array([0, 0, 0])]
        for i, dt in enumerate(diff_t_inp):
            tmp = self.quaternion_product(q_plus[i], - 2.2314 * F_inp[:, i] / self.m)
            r_dotdot_plus += [self.quaternion_product(tmp, self.quaternion_conjugate(q_plus[i]))[1:]  + self.gravity ]
            r_dot_plus += [r_dot_plus[-1] + r_dotdot_plus[-1] * dt]
            r_plus += [r_plus[-1] + r_dot_plus[-1] * dt]
            # tmp = tau_inp[:, i] - np.cross(omega_plus[-1], np.dot(self.I, omega_plus[-1]))
            # omega_dot_plus += [np.dot(np.linalg.inv(self.I), tmp)]
            # omega_plus += [omega_plus[-1] + omega_dot_plus[-1] * dt]

         
        
        # plt.close("all")
        # plt.plot(np.array(r_dotdot_plus)[:, 0])
        # plt.plot(np.array(r_dotdot_plus)[:, 1])
        # plt.plot(np.array(r_dotdot_plus)[:, 2])
        # plt.figure()
        # plt.plot(np.array(r_dot_plus)[:, 0])
        # plt.plot(np.array(r_dot_plus)[:, 1])
        # plt.plot(np.array(r_dot_plus)[:, 2])
        # plt.show()       
        # plt.figure()
        # plt.plot(np.array(r_plus)[:, 0])
        # plt.plot(np.array(r_plus)[:, 1])
        # plt.plot(np.array(r_plus)[:, 2])
        # plt.show()       
        
        
        
        
        
        # from casadi import integrator, cross, dot, transpose
        # omega_int = MX.sym('omega', 3)
        # tau_int = MX.sym('tau', 3)
        # tmp = tau_int - cross(omega_int, np.array(self.I) * omega_int)
        # omega_dot_int = np.linalg.inv(self.I) * tmp
        
        # tmp = self.quaternion_product(q, F*(1/self.m))
        # r_dotdot = self.quaternion_product(tmp, self.quaternion_conjugate(q))[1:] + self.gravity
        # q_dot = 0.5 * self.quaternion_product(q, omega)
        # tmp = tau - self.cross_product(omega, np.dot(self.I, omega))
        # omega_dot = np.dot(np.linalg.inv(self.I), tmp)
        # Step 5: plot the result
        
        
        print("kappa")
        "Sanity check - END"
        
        """
        # r_dot = dr
        tmp = self.quaternion_product(q, F*(1/self.m))
         # [1:] because we ignore the first element of the quaterion, because the results should be a 3D vector
         # Nevertheless, later we impose a constraint for q, for making sure, we are dealing with a unit quaternion
        r_dotdot = self.quaternion_product(tmp, self.quaternion_conjugate(q))[1:] + self.gravity
        q_dot = 0.5 * self.quaternion_product(omega, q)
        tmp = tau - self.cross_product(omega, np.dot(self.I, omega))
        omega_dot = np.dot(np.linalg.inv(self.I), tmp)
        
        "Dynamics constraints"
        # slack: to make it easier for the solver
        slack = self.slack
        # The two values should be the same (left hand side of the equation = right hand side of the equation)
        # self.define_constraint(ddr - r_dotdot,
        #                         lower_bound = [0 - slack] * 3,
        #                         upper_bound = [0 + slack] * 3,
        #                         constraint_type='overall',
        #                         name=["ddr_constraint"] * 3)
        # self.define_constraint(dq - q_dot,
        #                         lower_bound = [0 - slack] * 4,
        #                         upper_bound = [0 + slack] * 4,
        #                         constraint_type='overall',
        #                         name=["dq_constraint"] * 4)
        # self.define_constraint(domega - omega_dot,
        #                         lower_bound = [0 - slack] * 3,
        #                         upper_bound = [0 + slack] * 3,
        #                         constraint_type='overall',
        #                         name=["domega_constraint"] * 3)
        t = np.linspace(0, 1, self.nt)
        for t_ in t:
            self.define_constraint([ddr_(t_) - r_dotdot_(t_) for ddr_, r_dotdot_ in zip(ddr, r_dotdot)],
                                    lower_bound = [0 - slack] * 3,
                                    upper_bound = [0 + slack] * 3,
                                    constraint_type='time',
                                    name=["ddr_constraint"] * 3)
            
            self.define_constraint([dq_(t_) - q_dot_(t_) for dq_, q_dot_ in zip(dq, q_dot)],
                                    lower_bound = [0 - slack] * 4,
                                    upper_bound = [0 + slack] * 4,
                                    constraint_type='time',
                                    name=["dq_constraint"] * 4)
            
            self.define_constraint([domega_(t_) - omega_dot_(t_) for domega_, omega_dot_ in zip(domega, omega_dot)],
                                    lower_bound = [0 - slack] * 3,
                                    upper_bound = [0 + slack] * 3,
                                    constraint_type='time',
                                    name=["domega_constraint"] * 3)
        
        "Quaternion-related constraints"
        # Let's enforce unit quaternion
        # self.define_constraint([q[0]**2 + q[2]**2],
        #                         lower_bound = [1 - slack],
        #                         upper_bound = [1 + slack],
        #                         constraint_type='overall',
        #                         name=["unit_quaternion"] * 1)
        t = np.linspace(0, 1, self.nt)
        for t_ in t:
            self.define_constraint([q[0](t_)**2 + q[1](t_)**2 + q[2](t_)**2 + q[3](t_)**2],
                                    lower_bound = [1 - slack],
                                    upper_bound = [1 + slack],
                                    constraint_type='time',
                                    name=["unit_quaternion"] * 1)
        
        
        
        # Cost function
        # We put cost on using tau
        # cost = 0
        # for i in range(len(tau)):
        #     cost += tau[i]**2
        # self.J += 1 * definite_integral(cost, 0, 1)
        
        
        # And on using F
        # cost = 0
        # for i in range(len(F)):
        #     cost += F[i]**2
        # self.J += 1 * definite_integral(cost, 0, 1)
        
        
        
        self.J += 1.0 * definite_integral(tau[0]**2, 0, 1)
        self.J += 1.0 * definite_integral(F[2]**2, 0, 1)
        
        
        # q0 = my_data['q0']
        # q2 = my_data['q1']
        
        # # Downsampling q0 --again--
        # idx_len = len(q0)
        # idx_new = np.linspace(0, idx_len-1, 100, endpoint=True)
        # idx_new = np.floor(idx_new)
        # downsampled_q0 = [q0[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        
        
        # t = np.linspace(0, 1, 100)
        # for i, t_ in enumerate(t):
        #     self.J += 1.0 * (q[0](t_) - downsampled_q0[i])**2
        
        "Starting & final time constraint"
        
        self.define_constraint(r,
                                lower_bound = self.r0,
                                upper_bound = self.r0,
                                constraint_type='initial',
                                name=["r0"] * 3)
        self.define_constraint(r,
                                lower_bound = self.rf,
                                upper_bound = self.rf,
                                constraint_type='final',
                                name=["rf"] * 3)
        
        self.define_constraint(dr,
                                lower_bound = [0, 0, 0],
                                upper_bound = [0, 0, 0],
                                constraint_type='initial',
                                name=["dr0"] * 3)
        self.define_constraint(dr,
                                lower_bound = [0, 0, 0],
                                upper_bound = [0, 0, 0],
                                constraint_type='final',
                                name=["drf"] * 3)
        
        
        self.define_constraint(q,
                                lower_bound = np.array(self.q0) - slack,
                                upper_bound = np.array(self.q0) + slack,
                                constraint_type='initial',
                                name=["q0"] * 4)
        self.define_constraint(q,
                                lower_bound = np.array(self.qf) - slack,
                                upper_bound = np.array(self.qf) + slack,
                                constraint_type='final',
                                name=["qf"] * 4)
        
        
        idx_len = len(q0)
        idx_new = np.linspace(0, idx_len-1, 100, endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_q0 = [q0[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        
        idx_len = len(q0)
        idx_new = np.linspace(0, idx_len-1, 100, endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_q2 = [q2[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        
        
        # for i, t_ in enumerate(t):
        #     self.define_constraint([  (q[0](t_) - downsampled_q0[i])**2   +   (q[2](t_) - downsampled_q2[i])**2  ],
        #                             lower_bound = [0 - slack, 0 - slack],
        #                             upper_bound = [0 + slack, 0 + slack],
        #                             constraint_type='time',
        #                             name=["qf"] * 4)
        
        
        for i, t_ in enumerate(t):
            self.J += 1.0 / len(t) * (  (q[0](t_) - downsampled_q0[i])**2   +   (q[2](t_) - downsampled_q2[i])**2  )
        # self.J += 1000.0 * np.sum(np.array([(q_(0) - q0_)**2 for q_, q0_ in zip(q, self.q0)]))
        # self.J += 1000.0 * np.sum(np.array([(q_(1) - qf_)**2 for q_, qf_ in zip(q, self.qf)]))
        
        "Omega constraints"
        self.define_constraint(omega,
                                lower_bound = self.omega0,
                                upper_bound = self.omega0,
                                constraint_type='initial',
                                name=["omega0"] * 3)
        self.define_constraint(omega,
                                lower_bound = self.omegaf,
                                upper_bound = self.omegaf,
                                constraint_type='final',
                                name=["omegaf"] * 3)
        
        # The turning rate should be positive in the middle of the flip
        # self.define_constraint([omega[0](0.5)],
        #                         lower_bound = [0],
        #                         upper_bound = [math.inf],
        #                         constraint_type='time',
        #                         name=["omega0(t = 0.5)"] * 1)
        # And the orientation should be defind by the data pr
        # self.define_constraint([q[0](0.5)],
        #                         lower_bound = [0 - slack],
        #                         upper_bound = [0 + slack],
        #                         constraint_type='time',
        #                         name=["q0(t = 0.5)"] * 1)
        # self.define_constraint([q[2](0.5)],
        #                         lower_bound = [1 - slack],
        #                         upper_bound = [1 + slack],
        #                         constraint_type='time',
        #                         name=["q2(t = 0.5)"] * 1)
        
        prob = {'f': self.J,
                'x': vertcat(*self.w),
                'g': vertcat(*self.g),
                'p': vertcat(*self.P)
                }

        self.solver = nlpsol('solver', 'ipopt', prob)
        
        self.arg = {'x0' : self.w0,
                   'lbx': self.lbw,
                   'ubx': self.ubw,
                   'lbg': self.lbg,
                   'ubg': self.ubg}
        self.solution = self.solver.call(self.arg)
        
        
        self.DvX = DecisionVarX(self.w_list, self.g_list, self.lbg, self.ubg)
        # Extracting the solution to self.DvX. 
        # It will now contain the solution for the state vectors in a 1D list, like rx, rx, ...., ry, ry, ..., rz, rz, ...
        # We will reshape this list into 3x? and 4x? matrices in the plotter() function
        self.DvX.extract(self.solution)
        


    def plotter(self):
        # solution = self.solution['x'].full().reshape(1, -1).tolist()[0]
        "r, q, omega, F, tau"
        figs, axs = plt.subplots(2, 3)
        t = np.linspace(0, 1, 100)
        sampler = lambda x: [x(t_) for t_ in t]
        
        # Getting the coeffs
        r = np.array(flipper.DvX.r).reshape(3, -1)
        for i, spline_coeffs in enumerate(r):
            # Updating the coefficients
            self.r[i].coeffs = spline_coeffs
            # Plotting
            axs[0, 0].plot(sampler(self.r[i]))
        axs[0, 0].legend(['x', 'y', 'z'])
        
        # q
        q = np.array(flipper.DvX.q).reshape(4, -1)
        for i, spline_coeffs in enumerate(q):
            self.q[i].coeffs = spline_coeffs
            axs[0, 1].plot(sampler(self.q[i]))
        axs[0, 1].legend(['q0', 'q1', 'q2', 'q3'])
        
        # omega
        omega = np.array(flipper.DvX.omega).reshape(3, -1)
        for i, spline_coeffs in enumerate(omega):
            self.omega[i].coeffs = spline_coeffs
            axs[0, 2].plot(sampler(self.omega[i]))
        axs[0, 2].legend(['omega_x', 'omega_y', 'omega_z'])
        
        # F
        F = np.array(flipper.DvX.F).reshape(3, -1)
        for i, spline_coeffs in enumerate(F):
            self.F[i].coeffs = spline_coeffs
            axs[1, 0].plot(sampler(self.F[i]))
        axs[1, 0].legend(['F_x', 'F_y', 'F_z'])
        
        # tau
        tau = np.array(flipper.DvX.tau).reshape(3, -1)
        for i, spline_coeffs in enumerate(tau):
            self.tau[i].coeffs = spline_coeffs
            axs[1, 1].plot(sampler(self.tau[i]))
        axs[1, 1].legend(['tau_x', 'tau_y', 'tau_z'])
        
        # Saving figure
        plt.savefig("flip_values.png")
        
        
    def cross_product(self, spline1, spline2):
        a1, a2, a3 = spline1[0], spline1[1], spline1[2]
        b1, b2, b3 = spline2[0], spline2[1], spline2[2]
        
        c1 = a2 * b3 - b2 * a3
        c2 = -(a1 * b3 - b1 * a3)
        c3 = a1 * b2 - b1 * a2
        
        return np.array([c1, c2, c3])
        
    def quaternion_conjugate(self, q):
        # Only holds, if we are dealing with unit quaternions
        return [q[0], -q[1], -q[2], -q[3]]
    
    def quaternion_product(self, q, w):
        if len(q) == 4:
            q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        else:
            q0, q1, q2, q3 = 0, q[0], q[1], q[2]
        if len(w) == 4:
            w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
        else:
            w0, w1, w2, w3 = 0, w[0], w[1], w[2]
        q1 * w1
        x0 = q0 * w0 - (q1 * w1 + q2 * w2 + q3 * w3)
        
        x1 = w0 * q1 + q0 * w1 + (q2 * w3 - q3 * w2)
        x2 = w0 * q2 + q0 * w2 - (q1 * w3 - q3 * w1)
        x3 = w0 * q3 + q0 * w3 + (q1 * w2 - q2 * w1)
        
        return np.array([x0, x1, x2, x3])

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
                        self.w0 += np.linspace(initial_value[k][0], initial_value[k][1], len(basis)).tolist()
                        # w0_noise_added = np.linspace(initial_value[k][0], initial_value[k][1], len(basis))
                        # import random
                        # for i in range(len(w0_noise_added)):
                        #     w0_noise_added[i] = w0_noise_added[i] + w0_noise_added[i] * 0.05 * random.uniform(-0.5, 0.5)
    
                        # self.w0 += w0_noise_added.tolist()
                    else:
                        self.w0 += np.zeros((1, len(basis)))[0].tolist()
                        # import random
                        # self.w0 += [random.uniform(-0.5, 0.5) for i in range(len(basis))]
    
                else:
                    self.w0 += np.zeros((1, len(basis)))[0].tolist()
                    # import random
                    # self.w0 += [random.uniform(-0.5, 0.5) for i in range(len(basis))]
    
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
    
        else:
            raise NotImplementedError()
            
flipper = Flipper()
flipper.setup_solver()
flipper.plotter()
# flipper.solution['x'].full().reshape(1, -1).tolist()[0]