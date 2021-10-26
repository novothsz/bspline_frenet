from matplotlib.patches import Polygon
import numpy as np
from .frenet_path import FrenetPath
from .frenet_spline import SplineFitter
import os
from .spline import BSpline, BSplineBasis

# import pickle
# import dill


class Environment():
    def __init__(self):
        
        
        # self.start_position = [-0.8, 0]
        # self.goal_position = [0.8, 0]
        # 0 in the frenet frame :)
        # self.start_position = [0, 0]
        # self.goal_position = [0, 0]
        
        self.obstacle_area = { 'x_limits': [[-0.2, 0.2]], 'y_limits': [[-0.3, 0.3]] }
        self.fp = FrenetPath()
        self.fp = self.fp.fit_all()
        self.fitter = SplineFitter()
        
        
        
        self.border_x = [self.fp.tau_0, self.fp.tau_f]
        self.border_y = [-2, 2]
        self.cwd = os.getcwd()
        
        
        
        self.n_obstacle_cropped_degree = 3
        self.o_d = self.n_obstacle_cropped_degree
        self.n_obstacle_cropped_knot_intervals = 5
        
        self.obstacle_cropped_basis = self.define_knots(degree = self.n_obstacle_cropped_degree, 
                                                        knot_intervals = self.n_obstacle_cropped_knot_intervals)
        self.n_obstacle_cropped_coeffs = len(self.obstacle_cropped_basis)
        
        
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
            knots = kwargs['knots']
            knot_intervals = len(knots) - 2*degree - 1
            

        basis = BSplineBasis(knots, degree)

        return basis
        # try:
            # self.fp = dill.load('use_dill')
            # pickle_in = open("dict.pickle", "rb")
            # self.fp = pickle.load(pickle_in)
        # except:
        #     fp = FrenetPath()
            # pickle_out = open("fp.pickle", "wb")
            # pickle.dump(fp, pickle_out)
            # pickle_out.close()
            
            # dill.dump(fp, open('use_dill', 'wb'))
            # dill.dump_session('dill_session.pkl')
            
        # self.spline_fitter = SplineFitter(y_min = [self.border_x[0], self.border_y[0]],
        #                                   y_max = [self.border_x[1], self.border_y[1]])
        
        
    def plot_environment(self, ax, t_start):
        
        
        # Plotting border
        # corners = [[self.border_x[0], self.border_y[0]],
        #            [self.border_x[1], self.border_y[0]],
        #            [self.border_x[1], self.border_y[1]],
        #            [self.border_x[0], self.border_y[1]],
        #            [self.border_x[0], self.border_y[0]]
        #            ]
        # for obstacle in self.obstacles:
        #     corners = np.array(corners)
        #     polygon = Polygon(corners, closed=True, fill=False,
        #                       linestyle = '--',
        #                       fc=(0,0,0,0.1), ec=(0,0,0,1), lw=1, zorder = 1)
        #     ax.add_patch(polygon)
            
        self.fp.plot_path(ax, t_start)
            
        return ax
