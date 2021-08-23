from casadi import MX, SX, vertcat, dot, nlpsol, Function, integrator
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
        self.knot_intervals = 20
        self.N = 50
        self.T = 1
        self.options = {'print_time': False, 'ipopt': {'print_level' : 1, 'max_iter': 250, 'max_cpu_time': 100}}
        self.options = {'print_time': False, 'ipopt': {'max_iter': 35, 'max_cpu_time': 100}}
        self.options = {}
        # self.n_dimensions = 1
        
        "r"
        self.r_min = [-math.inf, -math.inf, -math.inf]
        self.r_max = [ math.inf,  math.inf,  math.inf]
        self.r_min = [-10, -10, -10]
        self.r_max = [10,  10,  10]
        self.r0 = [0, 0, 0]
        self.rf = [0, 0, 0]
        # self.r0 = [1000,1000,1000]
        # self.rf = [1000,1000,1000]
        # self.rf = [0, 0, 0.1]
        
        "q"
        self.q_min = [-1, 0, -1, 0]
        self.q_max = [ 1, 0, 1, 0]
        self.q0 = [1, 0, 0, 0]
        self.qf = [-1, 0, 0, 0]
        # self.qf = [1, 0, 0, 0]
        
        "omega"
        self.v_min = -2000 / 360 * 2 * math.pi
        self.v_max = 2000 / 360 * 2 * math.pi
        self.omega_min = [self.v_min, self.v_min, self.v_min]
        self.omega_max = [self.v_max,  self.v_max,  self.v_max]
        self.omega_min = [self.v_min, 0, 0]
        self.omega_max = [self.v_max,  0, 0]
        self.omega0 = [0, 0, 0]
        self.omegaf = [0, 0, 0]
        
        
        "F"
        self.F_min = [-math.inf, -math.inf, -math.inf]
        self.F_max = [math.inf, math.inf,  math.inf]
        
        "tau"
        self.u_min = -0.16 * 2 * 0.034
        self.u_max =  0.16 * 2 * 0.034
        self.tau_min = [self.u_min, self.u_min, self.u_min]
        self.tau_max = [self.u_max, self.u_max, self.u_max]
        # self.tau_min = [self.u_min, 0, 0]
        # self.tau_max = [self.u_max, 0, 0]
        # self.tau_min = [-math.inf, 0, 0]
        # self.tau_max = [math.inf, 0, 0]
        
        
        self.m = 0.027 # weight of cf_v2
        
        self.J11 = 1.4 * 1e-5
        self.J22 = 1.4 * 1e-5
        self.J33 = 2.17 * 1e-5
        self.I = np.eye(3) * np.array([self.J11, self.J22, self.J33]) #inertia
        self.gravity = np.array([0, 0, 9.81])
        
        self.arg = {}
        self.DvX = []
        
        # self.T = 0.2 + 0.18 + 0.2
        
    def setup_solver(self):
        N = self.N
        w, lbw, ubw = self.w, self.lbw, self.ubw
        g, lbg, ubg = self.g, self.lbg, self.ubg
        J = self.J
        P = self.P
        P0 = self.P0
        w0 = self.w0
        w_list, g_list, P_list = self.w_list, self.g_list, self.P_list
        
        T = MX.sym('T'); w += [T]; w_list += ["r_k"]
        w0 += [self.T]; lbw += [self.T]; ubw += [self.T]
        
        
        
        
        "Sanity check"
        my_data = np.load('/Users/szilard/Dropbox/Sztaki/code/code_examples/bspline_static/src/trajs.npy')
        
        # Also, we may want to initialize F
        F3_0 = my_data['input'][0, :]
        # Downsampling F3_0
        idx_len = len(F3_0)
        idx_new = np.linspace(0, idx_len-1, self.N, endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_F3_0 = [F3_0[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        
        
        
        
        # Also, we may want to initialize tau
        tau_x_0 = my_data['input'][1, :]
        # Downsampling F3_0
        idx_len = len(tau_x_0)
        idx_new = np.linspace(0, idx_len-1, self.N, endpoint=True)
        idx_new = np.floor(idx_new)
        downsampled_tau_x_0 = [tau_x_0[int(i)] for i in idx_new] # This is only a rough estimation of where the knot coeffs should be.
        
        
        # Initializing q with the values provided by Peti
        new_w0_for_F3_0 = [0] * len(downsampled_F3_0) + [0] * len(downsampled_F3_0) + downsampled_F3_0
        new_w0_for_tau_x_0 = downsampled_tau_x_0 + [0] * len(downsampled_tau_x_0) + [0] * len(downsampled_tau_x_0)
        
        state_history = []
        state_k = []
        state_k += [self.r0, self.r0, self.q0, self.omega0]
        state_len = len(vertcat(*state_k).full().reshape(1,-1).tolist()[0])
        for i in range(N):
            state_k = vertcat(*state_k).full().reshape(1,-1).tolist()[0]
            input_k = [0, 0, downsampled_F3_0[i] * 2.2314 * 1, 0, downsampled_tau_x_0[i], 0]
            # new_state = F(state = state_k[-state_len:], )
            
            F = self.setup_integrator()
            # res = F(state = state_k[-state_len:],
            #            state_input = input_k,
            #            dt = self.T/N)
            res = F(x0 = state_k[-state_len:],
                       p = input_k)
            
            
            # state_k += res['state_next'].full().reshape(1,-1).tolist()[0]
            # state_history += [res['state_next'].full().reshape(1,-1).tolist()[0]]
            # intQ = res['Q'].full().reshape(1,-1).tolist()[0]
            state_k += res['xf'].full().reshape(1,-1).tolist()[0]
            state_history += [res['xf'].full().reshape(1,-1).tolist()[0]]
            intQ = res['qf'].full().reshape(1,-1).tolist()[0]
            
        self.state_history = state_history
        self.plotter()
        
        "Sanity check - END"
        
        
        
        plt.show()
        for i in range(N):
            if i == 0:
                # Start position
                # r, r_dot, q, omega
                r_k = MX.sym('r_k', 3); w += [r_k]; w_list += ["r"] * 3
                w0 += [self.r0]; lbw += [self.r0]; ubw += [self.r0]
                
                r_dot_k = MX.sym('r_dot_k', 3); w += [r_dot_k]; w_list += ["r_dot"] * 3
                w0 += [self.r0]; lbw += [0, 0, 0]; ubw += [0, 0, 0]
                
                q_k = MX.sym('q_k', 4); w += [q_k]; w_list += ["q"] * 4
                w0 += [self.q0]; lbw += [self.q0]; ubw += [self.q0]
                
                omega_k = MX.sym('omega_k', 3); w += [omega_k]; w_list += ["omega"] * 3
                w0 += [self.omega0]; lbw += [self.omega0]; ubw += [self.omega0]
                
                # unit_q_constraint = q_k[0]**2 + q_k[1]**2 + q_k[2]**2 + q_k[3]**2
                # g += [unit_q_constraint]; lbg += [1]; ubg += [1]
                
                Xk = vertcat(r_k, r_dot_k, q_k, omega_k)
                
                
        
            F_k = MX.sym('F_k', 3); w += [F_k]; w_list += ["F"] * 3
            w0 += [[0] * 3]; lbw += [self.F_min]; ubw += [self.F_max]
            
            tau_k = MX.sym('F_k', 3); w += [tau_k]; w_list += ["tau"] * 3
            w0 += [[0] * 3]; lbw += [self.tau_min]; ubw += [self.tau_max]
        
            Uk = vertcat(F_k, tau_k)
            
            F = self.setup_integrator()
            # res = F(state = Xk,
            #            state_input = Uk,
            #            dt = T/N)
        
            # Xk_end = res['state_next']
            # intQ = res['Q']
            res = F(x0 = Xk,
                       p = Uk)
        
            Xk_end = res['xf']
            intQ = res['qf']
            
            
            # res = F(x0 = state_k[-state_len:],
            #            p = input_k)
            # state_history += [res['xf'].full().reshape(1,-1).tolist()[0]]
            # intQ = res['qf'].full().reshape(1,-1).tolist()[0]
            
            
            J = J + 1.0 * intQ
            
                
            
            
            if i == (N-1):
                # Final position
                # r, r_dot, q, omega
                r_k = MX.sym('r_k', 3); w += [r_k]; w_list += ["r"] * 3
                w0 += [self.rf]; lbw += [self.rf]; ubw += [self.rf]
                
                r_dot_k = MX.sym('r_dot_k', 3); w += [r_dot_k]; w_list += ["r_dot"] * 3
                w0 += [self.rf]; lbw += [0, 0, 0]; ubw += [0, 0, 0]
                
                q_k = MX.sym('q_k', 4); w += [q_k]; w_list += ["q"] * 4
                w0 += [self.qf]; lbw += [self.qf]; ubw += [self.qf]
                
                omega_k = MX.sym('omega_k', 3); w += [omega_k]; w_list += ["omega"] * 3
                w0 += [self.omegaf]; lbw += [self.omegaf]; ubw += [self.omegaf]
                
            else:
                # New symbolic
                r_k = MX.sym('r_k', 3); w += [r_k]; w_list += ["r"] * 3
                w0 += [self.r0]; lbw += [self.r_min]; ubw += [self.r_max]
                
                r_dot_k = MX.sym('r_dot_k', 3); w += [r_dot_k]; w_list += ["r_dot"] * 3
                w0 += [self.r0]; lbw += [self.r_min]; ubw += [self.r_max]
                
                q_k = MX.sym('q_k', 4); w += [q_k]; w_list += ["q"] * 4
                w0 += [self.q0]; lbw += [self.q_min]; ubw += [self.q_max]
                
                omega_k = MX.sym('omega_k', 3); w += [omega_k]; w_list += ["omega"] * 3
                w0 += [self.omega0]; lbw += [self.omega_min]; ubw += [self.omega_max]

                
            # unit_q_constraint = q_k[0]**2 + q_k[1]**2 + q_k[2]**2 + q_k[3]**2
            # g += [unit_q_constraint]; lbg += [1]; ubg += [1]
            
            Xk = vertcat(r_k, r_dot_k, q_k, omega_k)
            g += [Xk_end - Xk]; lbg += [[0] * Xk.shape[0]]; ubg += [[0] * Xk.shape[0]]
    
        
        
                
        # for i in range(N):
        #     if i == 0:
        #         # Start position
        #         # r, r_dot, q, omega
        #         Xk = MX.sym('X0', self.state_len); w += [Xk]; w_list += ["Xk"] # decision variable
        #         x0 = MX.sym('x0', self.state_len); P += [x0]; P_list += ["x0"] # parameter
                
        #         w0 += list(self.x0); lbw += self.state_min; \
        #                         ubw += self.state_max
        #         P0 += list(x0)      
                
        #         x_reference = MX.sym('x_reference', self.state_len); P += [x_reference] # parameter
        #         g += [Xk - x0]                  # dec. var. == parameter (meaning: we can set the starting position)

        #     Uk = MX.sym('U_' + str(i), self.input_len); w += [Uk] # decision variable



        # vertcat(*state_k).full().reshape(1,-1).tolist()[0]
        w, lbw, ubw = vertcat(*w), vertcat(*lbw), vertcat(*ubw)
        g, lbg, ubg = vertcat(*g), vertcat(*lbg), vertcat(*ubg)
        w0 = vertcat(*w0)
        self.w, self.lbw, self.ubw = w, lbw, ubw
        self.g, self.lbg, self.ubg = g, lbg, ubg
        self.J = J
        self.P = P
        self.P0 = P0
        self.w0 = w0
        self.w_list, self.g_list, self.P_list = w_list, g_list, P_list
        
        
        # prob = {'f': self.J,
        #         'x': vertcat(*self.w),
        #         'g': vertcat(*self.g),
        #         'p': vertcat(*self.P)
        #         }
        prob = {'f': self.J,
                'x': w,
                'g': g,
                'p': P
                }

        self.solver = nlpsol('solver', 'ipopt', prob, self.options)
        
        # self.arg = {'x0' : vertcat(*self.w0),
        #            'lbx': vertcat(*self.lbw),
        #            'ubx': vertcat(*self.ubw),
        #            'lbg': vertcat(*self.lbg),
        #            'ubg': vertcat(*self.ubg)}
        self.arg = {'x0' : w0,
                   'lbx': lbw,
                   'ubx': ubw,
                   'lbg': lbg,
                   'ubg': ubg}
        self.solution = self.solver.call(self.arg)
        
        self.DvX = DecisionVarX(self.w_list, self.g_list, self.lbg, self.ubg)
        # Extracting the solution to self.DvX. 
        # It will now contain the solution for the state vectors in a 1D list, like rx, rx, ...., ry, ry, ..., rz, rz, ...
        # We will reshape this list into 3x? and 4x? matrices in the plotter() function
        self.DvX.extract(self.solution)
        self.DvXplotter()
        
    def DvXplotter(self):
        # solution = self.solution['x'].full().reshape(1, -1).tolist()[0]
        "r, q, omega, F, tau"
        figs, axs = plt.subplots(2, 3)
        
        # Getting the coeffs
        DvX = self.DvX
        r = np.array(DvX.r).reshape(-1, 3)
        for i in range(r.shape[1]):
            # Plotting
            axs[0, 0].plot(r[:, i])
        axs[0, 0].legend(['x', 'y', 'z'])
        
        
        dr = np.array(DvX.r_dot).reshape(-1, 3)
        for i in range(dr.shape[1]):
            # Plotting
            axs[0, 1].plot(dr[:, i])
        axs[0, 1].legend(['dx', 'dy', 'dz'])
        
        
        # q
        q = np.array(DvX.q).reshape(-1, 4)
        for i in range(q.shape[1]):
            # Plotting
            axs[0, 2].plot(q[:, i])
        axs[0, 2].legend(['q0', 'q1', 'q2', 'q3'])
        
        # omega
        omega = np.array(DvX.omega).reshape(-1, 3)
        for i in range(omega.shape[1]):
            # Plotting
            axs[1, 0].plot(omega[:, i])
        axs[1, 0].legend(['omega_x', 'omega_y', 'omega_z'])
        
        # F
        F = np.array(DvX.F).reshape(-1, 3)
        for i in range(F.shape[1]):
            # Plotting
            axs[1,1].plot(F[:, i])
        axs[1, 1].legend(['F_x', 'F_y', 'F_z'])
        
        # tau
        tau = np.array(DvX.tau).reshape(-1, 3)
        for i in range(omega.shape[1]):
            # Plotting
            axs[1, 2].plot(tau[:, i])
        axs[1, 2].legend(['tau_x', 'tau_y', 'tau_z'])
        
        # Saving figure
        plt.savefig("flip_values_DvX.png")
            
    def plotter(self):
        # solution = self.solution['x'].full().reshape(1, -1).tolist()[0]
        "r, q, omega, F, tau"
        figs, axs = plt.subplots(2, 3)
        
        # Getting the coeffs
        hist = self.state_history
        r = np.array([hist[i][0:3] for i in range(len(hist))])
        for i in range(r.shape[1]):
            # Plotting
            axs[0, 0].plot(r[:, i])
        axs[0, 0].legend(['x', 'y', 'z'])
        
        
        dr = np.array([hist[i][3:6] for i in range(len(hist))])
        for i in range(dr.shape[1]):
            # Plotting
            axs[0, 1].plot(dr[:, i])
        axs[0, 1].legend(['dx', 'dy', 'dz'])
        
        
        # q
        q = np.array([hist[i][6:10] for i in range(len(hist))])
        for i in range(q.shape[1]):
            # Plotting
            axs[0, 2].plot(q[:, i])
        axs[0, 2].legend(['q0', 'q1', 'q2', 'q3'])
        
        # omega
        omega = np.array([hist[i][10:] for i in range(len(hist))])
        for i in range(omega.shape[1]):
            # Plotting
            axs[1, 0].plot(omega[:, i])
        axs[1, 0].legend(['omega_x', 'omega_y', 'omega_z'])
        
        # # F
        # F = np.array(flipper.DvX.F).reshape(3, -1)
        # for i, spline_coeffs in enumerate(F):
        #     self.F[i].coeffs = spline_coeffs
        #     axs[1, 0].plot(sampler(self.F[i]))
        # axs[1, 0].legend(['F_x', 'F_y', 'F_z'])
        
        # # tau
        # tau = np.array(flipper.DvX.tau).reshape(3, -1)
        # for i, spline_coeffs in enumerate(tau):
        #     self.tau[i].coeffs = spline_coeffs
        #     axs[1, 1].plot(sampler(self.tau[i]))
        # axs[1, 1].legend(['tau_x', 'tau_y', 'tau_z'])
        
        # Saving figure
        plt.savefig("flip_values.png")
        
        
    """
    Simple integrator
    """
    def setup_integrator(self):
        # Dimension
        # Symbolic variables
        T = MX.sym('T')
        
        # System state
        r = MX.sym('r', 3)
        dr = MX.sym('dr', 3)
        q = MX.sym('q', 4)
        omega = MX.sym('omega', 3)
        # System input
        F = MX.sym('F', 3)
        tau = MX.sym('tau', 3)
        
        # State equations
        r_dot = dr
        tmp = self.quaternion_product(q, F*(1/self.m))
        r_dotdot = self.quaternion_product(tmp, self.quaternion_conjugate(q))[1:] + self.gravity
        q_dot = - 0.5 * self.quaternion_product(omega, q)
        tmp = tau - self.cross_product(omega, vertcat(self.J11, self.J22, self.J33) * omega)
        omega_dot = vertcat(1 / self.J11, 1 / self.J22, 1 / self.J33) * tmp
        
        # Inputs
        state = [r, dr, q, omega]
        state_input = [F, tau]
        # Outputs
        state_dot = [r_dot, r_dotdot, q_dot, omega_dot]
        
        j = F[0]**2 + F[1]**2 + F[2]**2 + tau[0]**2 + tau[1]**2 + tau[2]**2
        j += 100 * (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2) - 1
        
        
        f = Function('f', [vertcat(*state),vertcat(*state_input)], \
                     [vertcat(*state_dot), j], \
                     ['state', 'state_input'],
                     ['state_dot', 'j'] \
                    )
            
            
        
        state = SX.sym('state', vertcat(*state).shape[0])
        state_input = SX.sym('state_input', vertcat(*state_input).shape[0])
        
        
        [state_dot, j] = f(state, state_input)
        
        dt = SX.sym('dt', 1)
        state_next = state + dt * state_dot
        Q = dt * j
        F = Function('F',
                     [state, state_input, dt],
                     [state_next, Q],
                     ['state', 'state_input', 'dt'],
                     ['state_next', 'Q'])
        
        
        # CVODES from the SUNDIALS suite
        dae = {'x':state, 'p':state_input, 'ode':state_dot, 'quad':j}
        opts = {'tf':1/self.N}
        F = integrator('F', 'cvodes', dae, opts)
        
        
        # Q = dt * j
        
        # # First equation
        # L = 0
        # state = r
        # u = []
        # state_dot = r_dot
        # f1 = Function('f1', [state, u], [state_dot, L])
        # # f1 = Function('f1', [vertcat(*state), vertcat(*u)], [vertcat(*state_dot), L])
        
        # X0 = MX.sym('X0', 3)
        # U = []
        # # U = MX.sym('U', 3)
        # [x_dot, Jk] = f1(X0, U)
        # dt = T / self.N
        # Xf = X0 + dt * x_dot
        # Q = dt * Jk
        # F1 = Function('F', [X0, U, T], [Xf, Q], ['x0', 'p', 'tf'], ['xf', 'qf'])
        
        # return F1
        
        return F
    
    
    def cross_product(self, spline1, spline2):
        a1, a2, a3 = spline1[0], spline1[1], spline1[2]
        b1, b2, b3 = spline2[0], spline2[1], spline2[2]
        
        c1 = a2 * b3 - b2 * a3
        c2 = -(a1 * b3 - b1 * a3)
        c3 = a1 * b2 - b1 * a2
        
        # return np.array([c1, c2, c3])
        return vertcat(c1, c2, c3)
        
    def quaternion_conjugate(self, q):
        # Only holds, if we are dealing with unit quaternions
        return [q[0], -q[1], -q[2], -q[3]]
    
    def quaternion_product(self, q, w):
        if isinstance(q, (MX,SX)):
            if q.shape[0] == 4:
                q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
            else:
                q0, q1, q2, q3 = 0, q[0], q[1], q[2]
        else:
            if len(q) == 4:
                q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
            else:
                q0, q1, q2, q3 = 0, q[0], q[1], q[2]
        if isinstance(w, (MX,SX)):    
            if w.shape[0] == 4:
                w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
            else:
                w0, w1, w2, w3 = 0, w[0], w[1], w[2]
        else:        
            if len(w) == 4:
                w0, w1, w2, w3 = w[0], w[1], w[2], w[3]
            else:
                w0, w1, w2, w3 = 0, w[0], w[1], w[2]
        
        
        x0 = q0 * w0 - (q1 * w1 + q2 * w2 + q3 * w3)
        
        x1 = w0 * q1 + q0 * w1 + (q2 * w3 - q3 * w2)
        x2 = w0 * q2 + q0 * w2 - (q1 * w3 - q3 * w1)
        x3 = w0 * q3 + q0 * w3 + (q1 * w2 - q2 * w1)
        
        return vertcat(x0, x1, x2, x3)

            
flipper = Flipper()
flipper.setup_solver()            
            
            
            
            
            
        # x0 = q0 * w0 - (q1 *  + q2 * w2 + q3 * )
        
        # x1 = w0 * q1 + q0 * w1 + (q2 * w3 - q3 * w2)
        # x2 = w0 * q2 + q0 * w2 - (q1 * w3 - q3 * w1)
        # x3 = w0 * q3 + q0 * w3 + (q1 * w2 - q2 * w1)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            