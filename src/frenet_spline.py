from .spline import BSpline, BSplineBasis
# from .frenet_path import FrenetPath

from casadi import MX, SX, Function, vertcat, dot, nlpsol, cos, sin, norm_2

import random
import numpy as np
import matplotlib.pyplot as plt
from .spline_extra import definite_integral
import math


class SplineFitter():
    def __init__(self, knot_intervals : int = 10):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.w_list, self.g_list, self.P_list = [], [], []
        self.options = {'print_time': False, 'ipopt': {'print_level' : 0, 'max_iter': 1000, 'max_cpu_time': 100}}
        

        self.n_dimensions = 2
        self.knot_intervals = knot_intervals
        self.y_min = [-20, -20]
        self.y_max = [20, 20]

    def define_knots(self,degree=3, **kwargs):
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

    def define_splines(self, degree, knot_intervals, n_spl, lower_bound, upper_bound, name = ''):
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
            self.w += [coeffs]
            self.w_list += [name[k] for i in range(len(basis))]
            splines += [BSpline(basis, coeffs)]


        for i in range(len(splines)):
            for j in range(splines[i].coeffs.shape[0]):
                self.lbw += [lower_bound[i]]
                self.ubw += [upper_bound[i]]

        return splines

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
                self.g_list += [name[i] + "_0"]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self
        elif constraint_type == 'final':
            for i in range(len(constraint)):
                self.g += [constraint[i].coeffs[-1]] # we restrict the last coefficient
                self.g_list += [name[i] + "_f"]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self

        else:
            raise NotImplementedError()

    def fitting_single(self, f, y_min : list = [-20], y_max : list = [20], degree : int = 3):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.w_list, self.g_list, self.P_list = [], [], []
        
        n_dimensions = len(f)
        n_samples = len(f[0])
        y = self.define_splines(degree=degree, knot_intervals=self.knot_intervals, n_spl=n_dimensions,
                                lower_bound=y_min, upper_bound=y_max,
                                name=["y"] * n_dimensions)
        
        # Initial position constraint on y
        # self.define_constraint(y,
        #                        [f_[0] for f_ in f],
        #                        [f_[0] for f_ in f],
        #                        constraint_type='initial',
        #                        name=["y0"] * n_dimensions)
        # # Final position constraint on y
        # self.define_constraint(y,
        #                        [f_[-1] for f_ in f],
        #                        [f_[-1] for f_ in f],
        #                        constraint_type='final',
        #                        name=["yf"] * n_dimensions)
        
        for i, t in enumerate(np.linspace(0, 1, n_samples)):
            for j in range(n_dimensions):
                self.J += (y[j](t) - f[j][i])**4
            
        
        prob = {'f': self.J,
                'x': vertcat(*self.w),
                'g': vertcat(*self.g)
                }

        solver = nlpsol('solver', 'ipopt', prob, self.options)

        arg = {'lbx': self.lbw,
               'ubx': self.ubw,
               'lbg': self.lbg,
               'ubg': self.ubg
               }

        self.solution = solver.call(arg)
        
        
        basis = y[0].basis
        coeffs = self.solution['x'].full()
        fitted_spline = []
        for i in range(len(f)):
            fitted_spline += [BSpline(basis, coeffs[len(basis)*i : len(basis)*(i+1)])]
        
        
        return fitted_spline
