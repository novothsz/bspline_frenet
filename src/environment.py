from matplotlib.patches import Polygon
import numpy as np
from .frenet_path import FrenetPath
from .frenet_spline import SplineFitter
import os

# import pickle
# import dill


class Environment():
    def __init__(self):
        
        self.border_x = [-6, 6]
        self.border_y = [-5, 2]
        self.cwd = os.getcwd()
        
        # self.start_position = [-0.8, 0]
        # self.goal_position = [0.8, 0]
        # 0 in the frenet frame :)
        # self.start_position = [0, 0]
        # self.goal_position = [0, 0]
        
        self.obstacle_area = { 'x_limits': [[-0.2, 0.2]], 'y_limits': [[-0.3, 0.3]] }
        self.fp = FrenetPath()
        self.fp = self.fp.fit_all()
        self.fitter = SplineFitter()
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
