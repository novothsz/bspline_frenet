import numpy as np
import math
import matplotlib.pyplot as plt
# from numpy import trapz
from math import hypot
from numpy import interp
from scipy.integrate import quad
from .frenet_spline import SplineFitter

from casadi import MX, SX, Function, vertcat, cos, sin, nlpsol
from .spline import BSpline, BSplineBasis

from .spline_extra import definite_integral, shift_spline, shift_knot1_fwd, shift_knot1_bwd, shift_over_knot, extrapolate




class FrenetPath(object):
    def __init__(self, tau_0 : float = float(-2 * math.pi) - math.pi/2, tau_f : float = float(2 * math.pi), N : int = 100):

        self.tau_0 = tau_0
        self.tau_f = tau_f
        self.N = N

        self.s0 = {}
        self.sf = {}
        
        
        # Temporary containers
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []
        self.options = {'print_time': False, 'ipopt': {'print_level' : 0, 'max_iter': 1000, 'max_cpu_time': 100}}
        self.knot_intervals = 10
        self.state_degree = 3
        

        pass
        
        self.fit_all()
        # Ezt is csináljuk meg, hamár :)
        
        # [equation_min_p, equation_max_p] = self.equation_min_max('p')
        # [equation_min_q, equation_max_q] = self.equation_min_max('q')
        # self.equation_min_p = equation_min_p
        # self.equation_max_p = equation_max_p
        # self.equation_min_q = equation_min_q
        # self.equation_max_q = equation_max_q



    def fit_all(self):

        "Fitted splines"
        try:
            # TODO: Maybe we shouldn't load the coeffs blindly, but check somehow, that the values we load are still valid.
            # Because before there was a problem, that I have changed the horizon of the frenet path and because of that, 
            # the loaded coefficients were a bit off...
            # 1 / 0
            fitter = SplineFitter()
            basis = fitter.define_knots(degree = 3, knot_intervals = fitter.knot_intervals)
            import pickle
            pickle_in = open("coeffs.pickle", "rb")
            coeffs = pickle.load(pickle_in)

            self.fx_spline = BSpline(basis, coeffs["fx_spline"])
            self.fx_d_spline = BSpline(basis, coeffs["fx_d_spline"])
            self.fx_c_spline = BSpline(basis, coeffs["fx_c_spline"])

            self.fy_spline = BSpline(basis, coeffs["fy_spline"])
            self.fy_d_spline = BSpline(basis, coeffs["fy_d_spline"])
            self.fy_c_spline = BSpline(basis, coeffs["fy_c_spline"])

            self.fz_spline = BSpline(basis, coeffs["fz_spline"])
            self.fz_d_spline = BSpline(basis, coeffs["fz_d_spline"])
            self.fz_c_spline = BSpline(basis, coeffs["fz_c_spline"])

            self.cos_f_theta_spline = BSpline(basis, coeffs["cos_f_theta_spline"])
            self.sin_f_theta_spline = BSpline(basis, coeffs["sin_f_theta_spline"])
                        
            if "equation_min_p" in coeffs:
                self.equation_min_p = BSpline(basis, coeffs["equation_min_p"])
                self.equation_max_p = BSpline(basis, coeffs["equation_max_p"])
                self.equation_min_q = BSpline(basis, coeffs["equation_min_q"])
                self.equation_max_q = BSpline(basis, coeffs["equation_max_q"])
            else:
                self.equation_min_p = None
                self.equation_max_p = None
                self.equation_min_q = None
                self.equation_max_q = None
            
            return self

        except:
            self.fx_spline = []
            self.fx_d_spline = []
            self.fx_c_spline = []

            self.fy_spline = []
            self.fy_d_spline = []
            self.fy_c_spline = []

            self.fz_spline = []
            self.fz_d_spline = []
            self.fz_c_spline = []

            self.cos_f_theta_spline = []
            self.sin_f_theta_spline = []
            
            print("Starting create spline (frenet path/fit_all")
            self.create_splines()
            print("Finished create spline (frenet path/fit_all")
            
            
            
            self.equation_min_p = None
            self.equation_max_p = None
            self.equation_min_q = None
            self.equation_max_q = None

            coeffs = {
                "fx_spline" : self.fx_spline.coeffs,
                "fx_d_spline" : self.fx_d_spline.coeffs,
                "fx_c_spline" : self.fx_c_spline.coeffs,
                "fy_spline" : self.fy_spline.coeffs,
                "fy_d_spline" : self.fy_d_spline.coeffs,
                "fy_c_spline" : self.fy_c_spline.coeffs,
                "fz_spline" : self.fz_spline.coeffs,
                "fz_d_spline" : self.fz_d_spline.coeffs,
                "fz_c_spline" : self.fz_c_spline.coeffs,
                "cos_f_theta_spline" : self.cos_f_theta_spline.coeffs,
                "sin_f_theta_spline" : self.sin_f_theta_spline.coeffs,
                }
            import pickle
            pickle_out = open("coeffs.pickle", "wb")
            pickle.dump(coeffs, pickle_out)
            pickle_out.close()

        return self

    "Creating spline versions of f, f_d, f_c"
    def create_splines(self):
        tau = np.linspace(self.tau_0, self.tau_f, 100).tolist()
        fitter = SplineFitter()

        # fx
        fx_ = np.array([self.fx(tau_) for tau_ in tau]).tolist()
        self.fx_spline = fitter.fitting_single([fx_])[0]

        # topickle = self.fx_spline
        # import pickle
        # pickle_out = open("fp.pickle", "wb")
        # pickle.dump(topickle, pickle_out)
        # pickle_out.close()

        # pickle_in = open("dict.pickle", "rb")
        # topickle_read = pickle.load(pickle_in)

        # fx_d
        fx_d_ = np.array([self.fx_d(tau_) for tau_ in tau]).tolist()
        self.fx_d_spline = fitter.fitting_single([fx_d_], degree = 3)[0]
        # fx_c
        fx_c_ = np.array([self.fx_c(tau_) for tau_ in tau]).tolist()
        self.fx_c_spline = fitter.fitting_single([fx_c_])[0]


        # fy
        fy_ = np.array([self.fy(tau_) for tau_ in tau]).tolist()
        self.fy_spline = fitter.fitting_single([fy_])[0]
        # fy_d
        fy_d_ = np.array([self.fy_d(tau_) for tau_ in tau]).tolist()
        self.fy_d_spline = fitter.fitting_single([fy_d_], degree = 3)[0]
        # fy_c
        fy_c_ = np.array([self.fy_c(tau_) for tau_ in tau]).tolist()
        self.fy_c_spline = fitter.fitting_single([fy_c_], degree = 3)[0]


        # fz
        fz_ = np.array([self.fz(tau_) for tau_ in tau]).tolist()
        self.fz_spline = fitter.fitting_single([fz_])[0]
        # fz_d
        fz_d_ = np.array([self.fz_d(tau_) for tau_ in tau]).tolist()
        self.fz_d_spline = fitter.fitting_single([fz_d_], degree = 3)[0]
        # fz_c
        fz_c_ = np.array([self.fz_c(tau_) for tau_ in tau]).tolist()
        self.fz_c_spline = fitter.fitting_single([fz_c_], degree = 3)[0]

        # cos(theta)
        cos_f_theta_ = np.array([np.cos(self.f_theta(tau_)) for tau_ in tau]).tolist()
        self.cos_f_theta_spline = fitter.fitting_single([cos_f_theta_], degree =3)[0]
        # sin(theta)
        sin_f_theta_ = np.array([np.sin(self.f_theta(tau_)) for tau_ in tau]).tolist()
        self.sin_f_theta_spline = fitter.fitting_single([sin_f_theta_], degree =3)[0]

        return self


    def t_to_tau(self, t):
        return interp(t,[0,1],[self.tau_0,self.tau_f])
    def tau_to_t(self, tau):
        return interp(tau,[self.tau_0,self.tau_f],[0,1])


    # The path function, the derivatives and the curvature
    "Functions"
    def fx(self, tau):
        return tau
    def fy(self, tau):
        return 1 * np.sin(tau / (math.pi / 2.0))
    def fz(self, tau):
        return 0.1 * tau

    def fx_d(self, tau):
        return 1.0
    def fy_d(self, tau):
        return np.cos(tau / (math.pi / 2.0)) / (math.pi / 2.0)
    def fz_d(self, tau):
        return 0.1

    def fx_dd(self, tau):
        return 0.0
    def fy_dd(self, tau):
        return -np.sin(tau / (math.pi / 2.0)) / (math.pi / 2.0)**2
    def fz_dd(self, tau):
        return 0.0

    def f_theta(self, tau):
        return math.atan2(self.fy_d(tau)/self.fx_d(tau), 1)

    def fx_c(self, tau):
        return abs(self.fx_dd(tau)) * (1 + self.fx_d(tau)**2)**-1.5
    def fy_c(self, tau):
        return abs(self.fy_dd(tau)) * (1 + self.fy_d(tau)**2)**-1.5
    def fz_c(self, tau):
        return abs(self.fz_dd(tau)) * (1 + self.fz_d(tau)**2)**-1.5




    def f_v(self, T, t, axes):
        """
        The frenet coodinate system mooves along the path at constant speed.
        """
        # ??
        # np.sqrt(definite_integral(group.fp.fx_spline.derivative()**2 + group.fp.fy_spline.derivative()**2,0, 1))
        # np.sqrt(definite_integral(self.fx_spline.derivative()**2 + self.fy_spline.derivative()**2,0, 1))
        if t < 1 and t >= 0:
            return self.sf[axes] / T
        else:
            # At the end the frenet frame stops mooving.
            return 0


    def inertial_to_frenet(self, x, y, t):
        """
        Takes a position in the inertial frame and transforms it to
        pq coordinates in the frenet frame, given the frenet frame is at a
        position, defined by t.
        """
        x_frenet, y_frenet = self.fx(self.t_to_tau(t)), self.fy(self.t_to_tau(t))
        theta = self.f_theta(self.t_to_tau(t))
        x_rel = x - x_frenet
        y_rel = y - y_frenet
        
        p = x_rel * cos(theta) + y_rel * sin(theta)
        q =-x_rel * sin(theta) + y_rel * cos(theta)
        
        return p, q
    
    def frenet_to_inertial(self, p, q, t):
        """
        Takes a position in the frenet frame and transforms it to
        xy coordinates, given the frenet frame is at a position, defined by t.
        """
        x, y = self.fx(self.t_to_tau(t)), self.fy(self.t_to_tau(t))
        theta = self.f_theta(self.t_to_tau(t))
        
        x_ = p * cos(theta) - q * sin(theta)
        y_ = p * sin(theta) + q * cos(theta)
        
        return x + x_, y + y_
    
    
        
        
    """    
    def min_max_p_dot(self):
        
        try:
            1/0
            import pickle
            # my_list = [equation_min.basis.degree, len(equation_min.basis), equation_min.coeffs,
            #            equation_max.basis.degree, len(equation_max.basis), equation_max.coeffs]
            with open('min_max_p_dot.pickle', 'rb') as f:
                my_list = pickle.load(f)
                
            degree = my_list[0]
            length = my_list[1]
            
            knot_intervals = length - degree
            knots = np.r_[np.zeros(degree),
                          np.linspace(0, 1, knot_intervals+1),
                          np.ones(degree)]
            basis = BSplineBasis(knots, degree)
            
            coeffs = my_list[2]
            equation_min = BSpline(basis, coeffs)
            coeffs = my_list[5]
            equation_max = BSpline(basis, coeffs)
            
            return [equation_min, equation_max]
        
        except:
            
            # lists
            w, lbw, ubw = [], [], []
            g, lbg, ubg = [], [], []
            J = 0
            
            # minimum & maximum values
            vx_min, vx_max = -10, 10
            vy_min, vy_max = -10, 10
            
            ax_min, ax_max = -100, 100
            ay_min, ay_max = -100, 100
            
            """"""
            # basis
            knot_intervals = 10
            degree = 3
            knots = np.r_[np.zeros(degree),
                          np.linspace(0, 1, knot_intervals+1),
                          np.ones(degree)]
            basis = BSplineBasis(knots, degree)
            
            # vx
            coeffs = MX.sym('vx', len(basis))
            w += [coeffs]
            for i in range(coeffs.shape[0]):
                lbw += [vx_min]
                ubw += [vx_max]
            vx = BSpline(basis, coeffs)
            
            # ax
            ax = vx.derivative()
            g += [ax.coeffs]
            for i in range(ax.coeffs.shape[0]):
                lbg += [ax_min]
                ubg += [ax_max]
            
            # vy
            coeffs = MX.sym('vy', len(basis))
            w += [coeffs]
            for i in range(coeffs.shape[0]):
                lbw += [vy_min]
                ubw += [vy_max]
            vy = BSpline(basis, coeffs)
            
            # ay
            ay = vy.derivative()
            g += [ay.coeffs]
            for i in range(ay.coeffs.shape[0]):
                lbg += [ay_min]
                ubg += [ay_max]
                
            
            # get some splines
            self.fit_all()
            sin_theta_c = self.sin_f_theta_spline
            cos_theta_c = self.cos_f_theta_spline
    
    
            "MINIMISE - part of p_dot equation"
            equation = vx * cos_theta_c + vy * sin_theta_c
            # equation = - vx * sin_theta_c + vy * cos_theta_c
            
            J += definite_integral(equation, 0, 1)
            
            # Creating solver class
            prob = {'f': J,
                    'x': vertcat(*w),
                    'g': vertcat(*g)
                    }
            
            solver = nlpsol('solver', 'ipopt', prob)
            
            # Assembling the argument dictionary
            arg = {'lbx': lbw,
                   'ubx': ubw,
                   'lbg': lbg,
                   'ubg': ubg}
            
            solution = solver.call(arg)
            solution = solution['x'].full()
            
            coeffs_len = vx.coeffs.shape[0]
            vx_min_sol = BSpline(basis, solution[:coeffs_len])
            vy_min_sol = BSpline(basis, solution[coeffs_len:coeffs_len*2])
            ax_min_sol = vx_min_sol.derivative()
            ay_min_sol = vy_min_sol.derivative()
            
            
            # sampling the solution
            t = np.linspace(0, 1, 100)
    
            vx_min_sol_ = np.array([vx_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            vy_min_sol_ = np.array([vy_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            ax_min_sol_ = np.array([ax_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            ay_min_sol_ = np.array([ay_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            
            
            # plotting
            fig, ax = plt.subplots(2)
            
            ax[0].plot(vx_min_sol_)
            ax[1].plot(vy_min_sol_)
            # ax[0].plot(ax_min_sol_)
            # ax[1].plot(ay_min_sol_)
    
            "MAXIMISE - part of p_dot equation"
            J = 0
            J += definite_integral(-equation, 0, 1)
            
            # Creating solver class
            prob = {'f': J,
                    'x': vertcat(*w),
                    'g': vertcat(*g)
                    }
            
            solver = nlpsol('solver', 'ipopt', prob)
            
            # Assembling the argument dictionary
            arg = {'lbx': lbw,
                   'ubx': ubw,
                   'lbg': lbg,
                   'ubg': ubg}
            
            solution = solver.call(arg)
            solution = solution['x'].full()
            
            coeffs_len = vx.coeffs.shape[0]
            vx_max_sol = BSpline(basis, solution[:coeffs_len])
            vy_max_sol = BSpline(basis, solution[coeffs_len:coeffs_len*2])
            ax_max_sol = vx_max_sol.derivative()
            ay_max_sol = vy_max_sol.derivative()
            
            
            # sampling the solution
            t = np.linspace(0, 1, 100)
    
            vx_max_sol_ = np.array([vx_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            vy_max_sol_ = np.array([vy_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            ax_max_sol_ = np.array([ax_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            ay_max_sol_ = np.array([ay_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            
            # plotting
            ax[0].plot(vx_max_sol_)
            ax[1].plot(vy_max_sol_)
            # ax[0].plot(ax_max_sol_)
            # ax[1].plot(ay_max_sol_)
            
            ax[0].set_title('vx to minimese (blue) and to  maximise(orange)')
            ax[1].set_title('vy to minimese (blue) and to  maximise(orange)')
            
            "Thus the boundaries:"
            
            # the equations
            equation_min = vx_min_sol * cos_theta_c + vy_min_sol * sin_theta_c
            equation_max = vx_max_sol * cos_theta_c + vy_max_sol * sin_theta_c
            
            
            # equation_min = - vx_min_sol * sin_theta_c + vy_min_sol * cos_theta_c
            # equation_max = - vx_max_sol * sin_theta_c + vy_max_sol * cos_theta_c
            
            # But we have a problem... equation_min and equation_max are splines with
            # a large number of coefficients. We need to reduce this, otherwise we will
            # have problems in the future.
            # Can we describe these splines with a spline, that has fewer coefficients?
            
            
            # sampling
            equation_min_ = np.array([equation_min(t_)[0] for t_ in t]).reshape(-1).tolist()
            equation_max_ = np.array([equation_max(t_)[0] for t_ in t]).reshape(-1).tolist()
            
            # plotting
            fig, ax = plt.subplots()
            ax.plot(equation_min_)
            ax.plot(equation_max_)
            ax.set_title('blue: minimum of the equation, orange: maximum of the equation')
            ax.grid()
            
            
            # Pickling
            import pickle
            my_list = [equation_min.basis.degree, len(equation_min.basis), equation_min.coeffs.reshape(-1,).tolist(),
                       equation_max.basis.degree, len(equation_max.basis), equation_max.coeffs.reshape(-1,).tolist()]
            with open('min_max_p_dot.pickle', 'wb') as f:
                pickle.dump(my_list, f)
            
            return [equation_min, equation_max]
    
    
    def min_max_q_dot(self):
        
        try:
            1/0
            import pickle
            # my_list = [equation_min.basis.degree, len(equation_min.basis), equation_min.coeffs,
            #            equation_max.basis.degree, len(equation_max.basis), equation_max.coeffs]
            with open('min_max_q_dot.pickle', 'rb') as f:
                my_list = pickle.load(f)
                
            degree = my_list[0]
            length = my_list[1]
            
            knot_intervals = length - degree
            knots = np.r_[np.zeros(degree),
                          np.linspace(0, 1, knot_intervals+1),
                          np.ones(degree)]
            basis = BSplineBasis(knots, degree)
            
            coeffs = my_list[2]
            equation_min = BSpline(basis, coeffs)
            coeffs = my_list[5]
            equation_max = BSpline(basis, coeffs)
            
            return [equation_min, equation_max]
        
        except:
            
            # lists
            w, lbw, ubw = [], [], []
            J = 0
            
            # minimum & maximum values
            vx_min, vx_max = -10, 10
            vy_min, vy_max = -10, 10
            
            # basis
            knot_intervals = 10
            degree = 3
            knots = np.r_[np.zeros(degree),
                          np.linspace(0, 1, knot_intervals+1),
                          np.ones(degree)]
            basis = BSplineBasis(knots, degree)
            
            # vx
            coeffs = MX.sym('vx', len(basis))
            w += [coeffs]
            for i in range(coeffs.shape[0]):
                lbw += [vx_min]
                ubw += [vx_max]
            vx = BSpline(basis, coeffs)
            
            # vy
            coeffs = MX.sym('vy', len(basis))
            w += [coeffs]
            for i in range(coeffs.shape[0]):
                lbw += [vy_min]
                ubw += [vy_max]
            vy = BSpline(basis, coeffs)
            
            # get some splines
            self.fit_all()
            sin_theta_c = self.sin_f_theta_spline
            cos_theta_c = self.cos_f_theta_spline
    
    
            "MINIMISE - part of p_dot equation"
            equation = - vx * sin_theta_c + vy * cos_theta_c
            
            J += definite_integral(equation, 0, 1)
            
            # Creating solver class
            prob = {'f': J,
                    'x': vertcat(*w)
                    }
            
            solver = nlpsol('solver', 'ipopt', prob)
            
            # Assembling the argument dictionary
            arg = {'lbx': lbw,
                   'ubx': ubw}
            
            solution = solver.call(arg)
            solution = solution['x'].full()
            
            coeffs_len = vx.coeffs.shape[0]
            vx_min_sol = BSpline(basis, solution[:coeffs_len])
            vy_min_sol = BSpline(basis, solution[coeffs_len:coeffs_len*2])
            
            # sampling the solution
            t = np.linspace(0, 1, 100)
    
            vx_min_sol_ = np.array([vx_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            vy_min_sol_ = np.array([vy_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            
            # plotting
            fig, ax = plt.subplots(2)
            
            ax[0].plot(vx_min_sol_)
            ax[1].plot(vy_min_sol_)
    
            "MAXIMISE - part of p_dot equation"
            J = 0
            J += definite_integral(-equation, 0, 1)
            
            # Creating solver class
            prob = {'f': J,
                    'x': vertcat(*w)
                    }
            
            solver = nlpsol('solver', 'ipopt', prob)
            
            # Assembling the argument dictionary
            arg = {'lbx': lbw,
                   'ubx': ubw}
            
            solution = solver.call(arg)
            solution = solution['x'].full()
            
            coeffs_len = vx.coeffs.shape[0]
            vx_max_sol = BSpline(basis, solution[:coeffs_len])
            vy_max_sol = BSpline(basis, solution[coeffs_len:coeffs_len*2])
            
            # sampling the solution
            t = np.linspace(0, 1, 100)
    
            vx_max_sol_ = np.array([vx_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            vy_max_sol_ = np.array([vy_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
            
            # plotting
            ax[0].plot(vx_max_sol_)
            ax[1].plot(vy_max_sol_)
            
            ax[0].set_title('vx to minimese (blue) and to  maximise(orange)')
            ax[1].set_title('vy to minimese (blue) and to  maximise(orange)')
            
            "Thus the boundaries:"
            
            # the equations
            equation_min = - vx_min_sol * sin_theta_c + vy_min_sol * cos_theta_c
            equation_max = - vx_max_sol * sin_theta_c + vy_max_sol * cos_theta_c
            
            # sampling
            equation_min_ = np.array([equation_min(t_)[0] for t_ in t]).reshape(-1).tolist()
            equation_max_ = np.array([equation_max(t_)[0] for t_ in t]).reshape(-1).tolist()
            
            # plotting
            fig, ax = plt.subplots()
            ax.plot(equation_min_)
            ax.plot(equation_max_)
            ax.set_title('blue: minimum of the equation, orange: maximum of the equation')
            ax.grid()
            
            
            # Pickling
            import pickle
            my_list = [equation_min.basis.degree, len(equation_min.basis), equation_min.coeffs.reshape(-1,).tolist(),
                       equation_max.basis.degree, len(equation_max.basis), equation_max.coeffs.reshape(-1,).tolist()]
            with open('min_max_q_dot.pickle', 'wb') as f:
                pickle.dump(my_list, f)
            
            
            return [equation_min, equation_max]
    """
       
       
    "Plot"
    def plot_rotation(self, x, y, theta):
        x_new = x * cos(theta) - y * sin(theta)
        y_new = x * sin(theta) + y * cos(theta)
        return x_new, y_new
    def plot_path(self, ax, t_start = 0):
        import numpy as np
        # Plottin entire path
        t = np.linspace(0, 1, 100)
        # x, y = [], []
        fx_ = [self.fx(self.t_to_tau(t_)) for t_ in t]
        fy_ = [self.fy(self.t_to_tau(t_)) for t_ in t]
        
        # path_plot = 
        ax.plot(fx_, fy_, label='Frenet path', linestyle = ':', color = 'gray', zorder = 2)
        # path_plot[0].set_label("Frenet path")

        # Plotting current position of the frenet coordinate system
        # x_current, y_current = self.t_to_xy(t)
        x_current, y_current = self.frenet_to_inertial(0, 0, t_start)
        # x_, _ = self.t_to_xy(t)
        # theta_c = math.atan2(self.f_d(float(x_)), 1
        theta_c = math.atan2(self.fy_d_spline(t_start)[0][0],
                                    self.fx_d_spline(t_start)[0][0])

        rot_x, rot_y = self.plot_rotation(0.1, 0, theta_c)
        ax.arrow(x_current, y_current, rot_x, rot_y, head_width=0.01, head_length=0.02, fc='r', ec='r', zorder = 3)
        rot_x, rot_y = self.plot_rotation(0.1, 0, theta_c + np.pi/2)
        ax.arrow(x_current, y_current, rot_x, rot_y, head_width=0.01, head_length=0.02, fc='r', ec='r', zorder = 3)


        return ax #, x_current, y_current


    def plot_curvature(self):
        x = np.linspace(self.x0, self.xf, self.N)
        y = [self.curvature(x_) for x_ in x]
        plt.plot(x, y)

    # def frenet_to_inertial(self, p, q, t):
    #     """
    #     Takes a position in the frenet frame and transforms it to
    #     xy coordinates, given the frenet frame is at a position, defined by t.
    #     """
    #     x, y = self.t_to_xy(t)
    #     theta_c = math.atan2(self.f_d(x), 1)
    #     x_ = p * cos(theta_c) - q * sin(theta_c)
    #     y_ = p * sin(theta_c) + q * cos(theta_c)

    #     return x + x_, y + y_


    # def inertial_to_frenet(self, x, y, t):
    #     """
    #     Takes a position in the inertial frame and transforms it to
    #     pq coordinates in the frenet frame, given the frenet frame is at a
    #     position, defined by t.
    #     """
    #     x_frenet, y_frenet = self.t_to_xy(t)
    #     theta_c = math.atan2(self.f_d(x_frenet), 1)
    #     x_rel = x - x_frenet
    #     y_rel = y - y_frenet

    #     p = x_rel * cos(theta_c) + y_rel * sin(theta_c)
    #     q =-x_rel * sin(theta_c) + y_rel * cos(theta_c)

    #     return p, q
    
    
    "Evaluation"

    def evaluate_x_fitting(self):
        # Creating frenet class
        fp = FrenetPath()
        fp.fit_all()
        plt.close('all')
        fig, ax = plt.subplots(3, 1)
        t = np.linspace(0, 1, 1000)

        # Spline fitting
        fx_ = [fp.fx(fp.t_to_tau(t_)) for t_ in t]
        fx_spline_ = [fp.fx_spline(t_)[0] for t_ in t]

        fx_d_ = [fp.fx_d(fp.t_to_tau(t_)) for t_ in t]
        fx_d_spline_ = [fp.fx_d_spline(t_)[0] for t_ in t]

        fx_c_ = [fp.fx_c(fp.t_to_tau(t_)) for t_ in t]
        fx_c_spline_ = [fp.fx_c_spline(t_)[0] for t_ in t]

        # Plotting and comparison
        ax[0].plot(t, fx_)
        ax[0].plot(t, fx_spline_)
        ax[0].set_title('fx')
        ax[1].plot(t, fx_d_, '.')
        ax[1].plot(t, fx_d_spline_)
        ax[1].set_title('fx_d_')
        ax[2].plot(t, fx_c_)
        ax[2].plot(t, fx_c_spline_)
        ax[2].set_title('fx_c')



    def evaluate_y_fitting(self):
        # Creating frenet class
        fp = FrenetPath()
        fp.fit_all()
        plt.close('all')
        fig, ax = plt.subplots(5, 1)
        t = np.linspace(0, 1, 1000)

        # Spline fitting
        fy_ = [fp.fy(fp.t_to_tau(t_)) for t_ in t]
        fy_spline_ = [fp.fy_spline(t_)[0] for t_ in t]

        fy_d_ = [fp.fy_d(fp.t_to_tau(t_)) for t_ in t]
        fy_d_spline_ = [fp.fy_d_spline(t_)[0] for t_ in t]

        fy_c_ = [fp.fy_c(fp.t_to_tau(t_)) for t_ in t]
        fy_c_spline_ = [fp.fy_c_spline(t_)[0] for t_ in t]

        cos_f_theta_ = [np.cos(fp.f_theta(fp.t_to_tau(t_))) for t_ in t]
        cos_f_theta_spline_ = [fp.cos_f_theta_spline(t_)[0] for t_ in t]

        sin_f_theta_ = [np.sin(fp.f_theta(fp.t_to_tau(t_))) for t_ in t]
        sin_f_theta_spline_ = [fp.sin_f_theta_spline(t_)[0] for t_ in t]

        # Plotting and comparison
        ax[0].plot(t, fy_)
        ax[0].plot(t, fy_spline_)
        ax[0].set_title('fy')
        ax[1].plot(t, fy_d_, '.')
        ax[1].plot(t, fy_d_spline_)
        ax[1].set_title('fy_d')
        ax[2].plot(t, fy_c_)
        ax[2].plot(t, fy_c_spline_)
        ax[2].set_title('fy_c')
        ax[3].plot(t, cos_f_theta_)
        ax[3].plot(t, cos_f_theta_spline_)
        ax[3].set_title('cos_f_theta')
        ax[4].plot(t, sin_f_theta_)
        ax[4].plot(t, sin_f_theta_spline_)
        ax[4].set_title('sin_f_theta')

    def evaluate_z_fitting(self):
        # Creating frenet class
        fp = FrenetPath()
        fp.fit_all()
        plt.close('all')
        fig, ax = plt.subplots(3, 1)
        t = np.linspace(0, 1, 1000)

        # Spline fitting
        fz_ = [fp.fz(fp.t_to_tau(t_)) for t_ in t]
        fz_spline_ = [fp.fz_spline(t_)[0] for t_ in t]

        fz_d_ = [fp.fz_d(fp.t_to_tau(t_)) for t_ in t]
        fz_d_spline_ = [fp.fz_d_spline(t_)[0] for t_ in t]

        fz_c_ = [fp.fz_c(fp.t_to_tau(t_)) for t_ in t]
        fz_c_spline_ = [fp.fz_c_spline(t_)[0] for t_ in t]

        # Plotting and comparison
        ax[0].plot(t, fz_)
        ax[0].plot(t, fz_spline_)
        ax[1].plot(t, fz_d_, '.')
        ax[1].plot(t, fz_d_spline_)
        ax[2].plot(t, fz_c_)
        ax[2].plot(t, fz_c_spline_)
        
        
        
    def equation_min_max(self, equation_name = ''):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []
        
        
        # minimum & maximum values
        vx_min, vx_max = -10, 10
        vy_min, vy_max = -10, 10
        
        ax_min, ax_max = -100, 100
        ay_min, ay_max = -100, 100
        
        
        v = self.define_MX_spline(degree=self.state_degree, knot_intervals=self.knot_intervals, n_spl=2,
                                lower_bound=[vx_min, vy_min], upper_bound=[vx_max, vy_max],
                                name=["spline"] * 2)
        # Centralised
        a = [v_.tolist().derivative() for v_ in v]
        
        
        self.define_constraint(a,
                               [ax_min, ay_min],
                               [ax_max, ay_max],
                               constraint_type='overall',
                               name=["input_constraint"] * 2)
        
        # get some splines
        # self.fit_all()
        sin_theta_c = self.sin_f_theta_spline
        cos_theta_c = self.cos_f_theta_spline
                
        # Choosing the equation
        if equation_name == 'p':
            equation = v[0] * cos_theta_c + v[1] * sin_theta_c
        elif equation_name == 'q':
            equation =-v[0] * sin_theta_c + v[1] * cos_theta_c
        elif equation_name == '':
            raise NotImplementedError()
            
        self.J += definite_integral(equation, 0, 1)
        
        

        # Creating solver class
        prob = {'f': self.J,
                'x': vertcat(*self.w),
                'g': vertcat(*self.g)
                }

        self.solver = nlpsol('solver', 'ipopt', prob, self.options)
        
        # Assembling the argument dictionary
        self.arg = {'x0' : self.w0,
                   'lbx': self.lbw,
                   'ubx': self.ubw,
                   'lbg': self.lbg,
                   'ubg': self.ubg}
        
        solution = self.solver.call(self.arg)
        solution = solution['x'].full()
        
        coeffs_len = v[0].tolist().coeffs.shape[0]
        vx_coeffs = solution[:coeffs_len]
        vy_coeffs = solution[coeffs_len:coeffs_len*2]
        
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        vx_min_sol = BSpline(basis, vx_coeffs)
        vy_min_sol = BSpline(basis, vy_coeffs)
        
        
        "---- Pretty much same, as in the old code ----"
        ax_min_sol = vx_min_sol.derivative()
        ay_min_sol = vy_min_sol.derivative()
        
        
        # sampling the solution
        t = np.linspace(0, 1, 100)

        vx_min_sol_ = np.array([vx_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        vy_min_sol_ = np.array([vy_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        ax_min_sol_ = np.array([ax_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        ay_min_sol_ = np.array([ay_min_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        
        
        # plotting
        # fig, ax = plt.subplots(2)
        
        # ax[0].plot(vx_min_sol_)
        # ax[1].plot(vy_min_sol_)
        
        "MAXIMISE - part of ..?.. equation"
        self.J = 0
        self.J += definite_integral(-equation, 0, 1)
        
        
        # Creating solver class
        prob = {'f': self.J,
                'x': vertcat(*self.w),
                'g': vertcat(*self.g)
                }

        self.solver = nlpsol('solver', 'ipopt', prob, self.options)
        
        # Assembling the argument dictionary
        self.arg = {'x0' : self.w0,
                   'lbx': self.lbw,
                   'ubx': self.ubw,
                   'lbg': self.lbg,
                   'ubg': self.ubg}
        
        solution = self.solver.call(self.arg)
        solution = solution['x'].full()
        
        
        
        coeffs_len = v[0].tolist().coeffs.shape[0]
        vx_coeffs = solution[:coeffs_len]
        vy_coeffs = solution[coeffs_len:coeffs_len*2]
        
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        vx_max_sol = BSpline(basis, vx_coeffs)
        vy_max_sol = BSpline(basis, vy_coeffs)
        ax_max_sol = vx_max_sol.derivative()
        ay_max_sol = vy_max_sol.derivative()
        
        "---- Pretty much same, as in the old code ----"
        # sampling the solution
        t = np.linspace(0, 1, 100)

        vx_max_sol_ = np.array([vx_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        vy_max_sol_ = np.array([vy_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        ax_max_sol_ = np.array([ax_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        ay_max_sol_ = np.array([ay_max_sol(t_)[0] for t_ in t]).reshape(-1).tolist()
        
        # plotting
        # ax[0].plot(vx_max_sol_)
        # ax[1].plot(vy_max_sol_)
        # ax[0].plot(ax_max_sol_)
        # ax[1].plot(ay_max_sol_)
        
        # ax[0].set_title('vx to minimese (blue) and to  maximise(orange)')
        # ax[1].set_title('vy to minimese (blue) and to  maximise(orange)')
        
        
        "Plotting the equation min/max"
        # Choosing the equation
        if equation_name == 'p':
            equation_min = vx_min_sol * cos_theta_c + vy_min_sol * sin_theta_c
            equation_max = vx_max_sol * cos_theta_c + vy_max_sol * sin_theta_c
        elif equation_name == 'q':
            equation_min = - vx_min_sol * sin_theta_c + vy_min_sol * cos_theta_c
            equation_max = - vx_max_sol * sin_theta_c + vy_max_sol * cos_theta_c
        elif equation_name == '':
            raise NotImplementedError()
            
        "---- Pretty much same, as in the old code ----"
        
        # sampling
        equation_min_ = np.array([equation_min(t_)[0] for t_ in t]).reshape(-1).tolist()
        equation_max_ = np.array([equation_max(t_)[0] for t_ in t]).reshape(-1).tolist()
        
        # plotting
        # fig, ax = plt.subplots()
        # ax.plot(equation_min_)
        # ax.plot(equation_max_)
        # ax.set_title('blue: minimum of the equation, orange: maximum of the equation')
        # ax.grid()
        
        # But we have a problem... equation_min and equation_max are splines with
        # a large number of coefficients. We need to reduce this, otherwise we will
        # have problems in the future.
        # Can we describe these splines with a spline, that has fewer coefficients?
        
        # Yes, we can!!!
        # What we will do, is we will fit a new spline on the equation spline, but with fewer
        # coefficients. # We will copy the fitter function over, from the SplineFitter
        # class, but with a slight modification: This version will have to have a minimum or
        # a maximum value constraint.
        # (Meaning: we are searching for the best fit, but we are not allowed to go lower
        # than the MAXIMUM of the equation_min spline or above the MIMIMUM of the 
        # equation_max spline)
        
        fitter = SplineFitter()
        
        # No, we can't do it like this.
        # We do need to change the fitting function, but the following way:
        # we add a cost for being far away from the original spline at every timestep
        # but additionally, we also say, that at every timestep we need to be either above
        # or below the original spline. (Depending on which equation we are fiting to.)
        
        
        # lbw_eq1 = max(equation_min.coeffs)
        # ubw_eq1 = math.inf
        # lbw_eq2 = -math.inf
        # ubw_
        equation_min_fitted, equation_max_fitted = fitter.min_max_fitting(equation_min, equation_max)
        # sampling
        equation_min_fitted_ = np.array([equation_min_fitted(t_)[0] for t_ in t]).reshape(-1).tolist()
        equation_max_fitted_ = np.array([equation_max_fitted(t_)[0] for t_ in t]).reshape(-1).tolist()
        # ax.plot(equation_min_fitted_, '*')
        # ax.plot(equation_max_fitted_, '*')
        
        return [equation_min_fitted, equation_max_fitted]
    
    ###########################################################################
    ###########################################################################
    
    # def shift_spline_khm(self, spline, shift_type = ''):
        
        
        
    def shift_spline_khm(spline, shift_type = ''):
        
        spline_original = BSpline(spline.basis, spline.coeffs)
        t = np.linspace(0, 1, 100)
        t_shift = np.linspace(0, 1, 100)
        # shifting
        if shift_type == 'extrapolate':
            shift = 0.15
            original_coeffs_len = len(spline.coeffs)
            tmp_coeffs = extrapolate(spline.coeffs, shift, spline.basis)
            extra_coeffs_len = len(tmp_coeffs) - original_coeffs_len
            spline.coeffs = tmp_coeffs[extra_coeffs_len:] # deleting the couple of coefficients
            
            t_shift = np.linspace(0+0.1, 1+0.1, 100)
            
        if shift_type == 'cut_beginning':
            shift = 0.15
            spline.coeffs = shift_knot1_fwd(spline.coeffs, spline.basis, shift)
            t_shift = np.linspace(0, 1, 100)
            
        if shift_type == 'cut_back':
            shift = 0.15
            spline.coeffs = np.flip(spline.coeffs)
            spline.coeffs = shift_knot1_fwd(spline.coeffs, spline.basis, shift)
            spline.coeffs = np.flip(spline.coeffs)
            t_shift = np.linspace(0, 1, 100)
            
        if shift_type == 'cut_beginning2':
            shift = 0.15
            spline.coeffs = shift_spline(spline.coeffs, shift, spline.basis)
            t_shift = np.linspace(0.15, 1, 100)
        
        # plotting
        spline_original_t = np.array([spline_original(t_)[0] for t_ in t]).reshape(-1).tolist()
        spline_t = np.array([spline(t_)[0] for t_ in t]).reshape(-1).tolist()
        
        fig, ax = plt.subplots()
        ax.plot(t, spline_original_t, 'go')
        ax.plot(t_shift, spline_t, 'k')
        ax.set_title("spline evaluation")
        fig, ax = plt.subplots()
        ax.plot(spline_original.coeffs, 'go')
        ax.plot(spline.coeffs, 'k*')
        ax.set_title("--coefficients--")
        plt.show()
                
    # # Choosing the equation
    # if equation_name == 'p':
    # 	equation_min = vx_min_sol * cos_theta_c + vy_min_sol * sin_theta_c
    # 	equation_max = vx_max_sol * cos_theta_c + vy_max_sol * sin_theta_c
    # elif equation_name == 'q':
    # 	equation_min = - vx_min_sol * sin_theta_c + vy_min_sol * cos_theta_c
    # 	equation_max = - vx_max_sol * sin_theta_c + vy_max_sol * cos_theta_c
    # elif equation_name == '':
    # 	raise NotImplementedError()
    
    # plt.close('all')
    # shift_spline_khm(equation_min, 'extrapolate')
        
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

        else:
            raise NotImplementedError()




# fp = FrenetPath()
# fp = fp.fit_all()
# fp.evaluate_y_fitting()
