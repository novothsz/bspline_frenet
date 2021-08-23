from casadi import *
from spline import BSpline, BSplineBasis
from spline_extra import definite_integral
import math

w, lbw, ubw = [], [], []
g, lbg, ubg = [], [], []
J = 0
w0 = []
w_list, g_list = [], []







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
        
        self.state_degree = 3
        self.knot_intervals = 20
        self.n_dimensions = 1
        
        "r"
        self.r_min = [-math.inf, -math.inf, -math.inf]
        self.r_max = [ math.inf,  math.inf,  math.inf]
        self.r0 = [0, 0, 0]
        self.rf = [0, 0, 0]
        
        "q"
        self.q_min = [-math.inf, -math.inf, -math.inf, -math.inf]
        self.q_max = [ math.inf,  math.inf,  math.inf,  math.inf]
        self.q0 = [0, 0, 0, 0]
        self.qf = [0, 0, 0, 0]
        
        "omega"
        self.omega_min = [-math.inf, -math.inf, -math.inf]
        self.omega_max = [ math.inf,  math.inf,  math.inf]
        self.omega0 = [0, 0, 0]
        self.omegaf = [0, 0, 0]
        
        
        "F"
        self.F_min = [-math.inf, -math.inf, -math.inf]
        self.F_max = [ math.inf,  math.inf,  math.inf]
        self.F0 = [0, 0, 0]
        self.Ff = [0, 0, 0]
        
        "tau"
        self.tau_min = [-math.inf, -math.inf, -math.inf]
        self.tau_max = [ math.inf,  math.inf,  math.inf]
        self.tau0 = [0, 0, 0]
        self.tauf = [0, 0, 0]
        
        
        self.m = 0.03
        
        self.J11 = 1.0968 * 10e-5
        self.J22 = 1.0195 * 10e-5
        self.J33 = 1.3905 * 10e-5
        self.I = np.eye(3) * np.array([self.J11, self.J22, self.J33])
        self.gravity = np.array([0, 0, 9.81])
        
        self.arg = {}
        
        self.u_min = -0.16 * 2 * 0.034
        self.u_max =  0.16 * 2 * 0.034
        self.v_min = -1400 / 360 * 2 * math.pi
        self.v_max = 1400 / 360 * 2 * math.pi
        self.T = 0.2 + 0.18 + 0.2
        
        
    def setup_solver(self):
        
        # Decision variables
        # ---- States ---- #
        "r, r_dot"
        r = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.r_min, upper_bound=self.r_max,
                                name=["r"] * 4)
        # initial_value = [[self.x0[0], self.xf[0]], [self.x0[1], self.xf[1]]],
        
        dr = np.array([r_.derivative() for r_ in r])
        ddr = np.array([dr_.derivative() for dr_ in dr])
        
        "q"
        q = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=4,
                                lower_bound=self.q_min, upper_bound=self.q_max,
                                name=["q"] * 4)
        
        dq = np.array([q_.derivative() for q_ in q])
        
        "omega"
        omega = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.omega_min, upper_bound=self.omega_max,
                                name=["omega"] * 3)
        domega = np.array([omega_.derivative() for omega_ in omega])
        
        # ---- Inputs ---- #
        "F"
        F = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.omega_min, upper_bound=self.omega_max,
                                name=["F"] * 3)
        "tau"
        tau = self.define_MX_spline(degree=3, knot_intervals=self.knot_intervals, n_spl=3,
                                lower_bound=self.omega_min, upper_bound=self.omega_max,
                                name=["tau"] * 3)
        
        "Dynamics"
        tmp = self.quaternion_product(q, F*(1/self.m))
        r_dotdot = self.quaternion_product(tmp, self.quaternion_conjugate(q))[1:] + self.gravity
        q_dot = 0.5 * self.quaternion_product(q, omega)
        tmp = (tau - self.cross_product(omega, np.dot(self.I, omega)  ))
        omega_dot = np.dot(np.linalg.inv(self.I), tmp)
        
        "Dynamics constraints"
        slack = 0.01
        self.define_constraint(ddr - r_dotdot,
                                lower_bound = [0 - slack] * 3,
                                upper_bound = [0 + slack] * 3,
                                constraint_type='overall',
                                name=["ddr_constraint"] * self.n_dimensions)
        self.define_constraint(dq - q_dot,
                                lower_bound = [0 - slack] * 4,
                                upper_bound = [0 + slack] * 4,
                                constraint_type='overall',
                                name=["dq_constraint"] * self.n_dimensions)
        self.define_constraint(domega - omega_dot,
                                lower_bound = [0 - slack] * 3,
                                upper_bound = [0 + slack] * 3,
                                constraint_type='overall',
                                name=["domega_constraint"] * self.n_dimensions)
        
        "Other constraints"
        # self.define_constraint([q[1]],
        #                         lower_bound = [0],
        #                         upper_bound = [0],
        #                         constraint_type='overall',
        #                         name=[""] * self.n_dimensions)
        # self.define_constraint([q[3]],
        #                         lower_bound = [0],
        #                         upper_bound = [0],
        #                         constraint_type='overall',
        #                         name=[""] * self.n_dimensions)
        # self.define_constraint([q[0]**2 + q[2]**2],
        #                         lower_bound = [1],
        #                         upper_bound = [1],
        #                         constraint_type='overall',
        #                         name=[""] * self.n_dimensions)
        
        
        # Cost function
        cost = 0
        for i in range(len(tau)):
            cost += tau[i]**2
        self.J += 1 * definite_integral(cost, 0, 1)
        cost = 0
        for i in range(len(F)):
            cost += F[i]**2
        self.J += 1 * definite_integral(cost, 0, 1)
        
        "Starting time constraint"
        
        
        
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
        
        #         # Initial position constraint on y
        # self.define_constraint(y,
        #                         self.x0[:self.n_dimensions],
        #                         self.x0[:self.n_dimensions],
        #                         constraint_type='initial',
        #                         name=["y0"] * self.n_dimensions)


        # # Final position constraint on y
        # self.define_constraint(y,
        #                         self.xf[:self.n_dimensions],
        #                         self.xf[:self.n_dimensions],
        #                         constraint_type='final',
        #                         name=["yf"] * self.n_dimensions)




        #             # Formation constraint.
        #     for t in np.linspace(0, 1, self.t_resolution_length):
        #         self.define_constraint([cross(t)],
        #                                 [-self.slack * 1],
        #                                 [self.slack * 1],
        #                                 constraint_type='time',
        #                                 name=["formation_vehicle_" + str(i)] * self.n_dimensions)
        
        # Input constraint
        # self.define_constraint(dy,
        #                        self.v_min,
        #                        self.v_max,
        #                        constraint_type='overall',
        #                        name=["velocity_constraint"] * self.n_dimensions)
        # self.define_constraint(u,
        #                        self.u_min,
        #                        self.u_max,
        #                        constraint_type='overall',
        #                        name=["acceleration_constraint"] * self.n_dimensions)


    def cross_product(self, spline1, spline2):
        a1, a2, a3 = spline1[0], spline1[1], spline1[2]
        b1, b2, b3 = spline2[0], spline2[1], spline2[2]
        
        c1 = a2 * b3 - b2 * a3
        c2 = -(a1 * b3 - b1 * a3)
        c3 = a1 * b2 - b1 * a2
        
        return np.array([c1, c2, c3])
        
    def quaternion_conjugate(self, q):
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
        
        x0 = q0 * w0 - q1 * w1 + q2 * w2 + q3 * w3
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
    
        else:
            raise NotImplementedError()
            
flipper = Flipper()
flipper.setup_solver()