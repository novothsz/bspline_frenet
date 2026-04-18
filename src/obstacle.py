from casadi import SX, MX, Function, vertcat, sin, cos
from .frenet_path import FrenetPath
from matplotlib.patches import Polygon
import numpy as np
from numpy import interp
from .environment import Environment
from .bspline import BSpline, BSplineBasis, crop_spline, extrapolate, shift_knot1_fwd, shift_spline


class Obstacle(Environment):
    def __init__(self, ID : int = 0, corners: list = [], x_limits: list = [-0.2, 0.2], y_limits: list = [-0.3, 0.3]):
        super().__init__()
        self.ID = ID
        if corners:
            self.corners = corners
        else:
            self.corners = self.random_placement(x_limits, y_limits)
            # Calculating center
        points = self.corners
        center = []
        center += [np.mean(np.array(points)[:, 0]).tolist()]
        center += [np.mean(np.array(points)[:, 1]).tolist()]
        # center += [0.1]
        self.center = center
        self.max_dist_from_center = max([np.linalg.norm(np.array([self.center]) - np.array([corn])) for corn in self.corners])
        self.scaled_corners = self.scaled_corners(size = 0.03)    
        self.corners_spline = [] # these will be splines defined in the frenet frame.
        self.corners_t = [] # sampling of the corner positions in the mooving frenet frame. They represent
        # the true value at time t and can be used measure if we have correctly fitted the spline.
        
        
        self.scaled_corners_spline = []
        self.scaled_corners_t = []
        self.center_spline = []

        self.spline_position_in_frenet() # This function creates the self.corners_spline values.
        self.gate_pair_ID = []
        
        
        
        
        
    def cropped_corner_trajectories(self, default_basis, t_start, t_end):
        """In this function we slice up the entire spline trajectory into pices
        """
        cropped_corners = []
        for corner in self.corners_spline:
            cropped_corner = []
            for xy in corner:
                xy = crop_spline(xy, t_start, t_end)
                xy = xy.scale(1, -t_start)
                xy = xy.scale(  1 * 1 / (t_end - t_start), 0  )
                # for i in range(len(xy.basis.knots)):
                eps = 1e-5
                xy.basis.knots[0:default_basis.degree] = 0.0
                xy.basis.knots[0] = 0.0 - eps
                # xy.basis.knots[4] = xy.basis.knots[4] + eps * 10
                xy.basis.knots[-default_basis.degree:] = 1.0
                xy.basis.knots[-1] = 1.0 + eps
                # xy.basis.knots[-4] = xy.basis.knots[-4] - eps
                # xy.basis.knots[-1] = xy.basis.knots[-1] + eps * 10
                    # if xy.basis.knots[i] >= -eps and xy.basis.knots[i] <= eps:
                    #     xy.basis.knots[i] = 0
                    # if xy.basis.knots[i] >= -eps + 1 and xy.basis.knots[i] <= eps + 1:
                    #     xy.basis.knots[i] = 1
                # Oky, but the basis has changed. Let us now convert it to the default basis.
                new_coeffs = default_basis.transform(xy.basis).dot(xy.coeffs)
                cropped_corner += [BSpline(default_basis, new_coeffs)]
            cropped_corners += [cropped_corner]
            
        return cropped_corners
            
    
    
    def spline_position_in_frenet(self):
        
        # try:
        #     basis = self.fitter.define_knots(degree = 3, knot_intervals = self.fitter.knot_intervals)
        #     import pickle
        #     pickle_in = open("obst_" + str(int(self.ID)) + "_coeffs.pickle", "rb")
        #     coeffs = pickle.load(pickle_in)
    
        #     for i in range(len(self.corners)):
        #         self.corners_spline += [  [BSpline(basis, coeffs_) for coeffs_ in coeffs["corners_spline_coeffs"][i]]  ]
        #         self.scaled_corners_spline += [  [BSpline(basis, coeffs_) for coeffs_ in coeffs["scaled_corners_spline_coeffs"][i]]  ]
        #     return self
        
        # except:
        #     pass
        
        
        # ---- Regular corners
        corners_spline_coeffs = []
        t = np.linspace(0, 1, 100)
        self.fitter.knot_intervals = 10
        for i in range(len(self.corners)):
            corner_ = np.array([self.fp.inertial_to_frenet(self.corners[i][0], self.corners[i][1], t_) for t_ in t])
            p_ = corner_[:, 0].tolist()
            q_ = corner_[:, 1].tolist()
            corner_ = [p_, q_]
            # fitted_splines = self.fitter.fitting(corner_,
            #                                      y_min = [self.border_x[0], self.border_y[0]],
            #                                      y_max = [self.border_x[1], self.border_y[1]]
            #                                      )
            fitted_splines = self.fitter.fitting_single(corner_,
                                                 y_min = [-20, -20],
                                                 y_max = [20, 20]
                                                 )
            self.corners_spline += [fitted_splines]
            corners_spline_coeffs += [ [sp.coeffs for sp in fitted_splines] ]
            self.corners_t += [corner_]
            
        # ---- Scaled corners
        scaled_corners_spline_coeffs = []
        t = np.linspace(0, 1, 100)
        self.fitter.knot_intervals = 10
        for i in range(len(self.scaled_corners)):
            corner_ = np.array([self.fp.inertial_to_frenet(self.scaled_corners[i][0], self.scaled_corners[i][1], t_) for t_ in t])
            p_ = corner_[:, 0].tolist()
            q_ = corner_[:, 1].tolist()
            corner_ = [p_, q_]
            # fitted_splines = self.fitter.fitting(corner_,
            #                                      y_min = [self.border_x[0], self.border_y[0]],
            #                                      y_max = [self.border_x[1], self.border_y[1]]
            #                                      )
            fitted_splines = self.fitter.fitting_single(corner_,
                                                 y_min = [-20, -20],
                                                 y_max = [20, 20]
                                                 )
            self.scaled_corners_spline += [fitted_splines]
            scaled_corners_spline_coeffs += [ [sp.coeffs for sp in fitted_splines] ]
            self.scaled_corners_t += [corner_]
        # self.plot_corners_spline()
        
        # ---- Virtual center "corner/circle" with virtual "radious"
        center_spline_coeffs = []
        t = np.linspace(0, 1, 100)
        self.fitter.knot_intervals = 10
        center = self.center
        corner_ = np.array([self.fp.inertial_to_frenet(center[0], center[1], t_) for t_ in t])
        p_ = corner_[:, 0].tolist()
        q_ = corner_[:, 1].tolist()
        corner_ = [p_, q_]
        fitted_splines = self.fitter.fitting_single(corner_,
                                             y_min = [-20, -20],
                                             y_max = [20, 20]
                                             )
        self.center_spline += [fitted_splines]
        center_spline_coeffs += [ [sp.coeffs for sp in fitted_splines] ]
        # self.scaled_corners_t += [corner_]
        
        
        # ---- Collecting all the coeffs and writing it to file
        coeffs = {
                "corners_spline_coeffs" : corners_spline_coeffs,
                "scaled_corners_spline_coeffs" : scaled_corners_spline_coeffs,
                "center_spline_coeffs" : center_spline_coeffs
                }
        import pickle
        pickle_out = open("obst_" + str(int(self.ID)) + "_coeffs.pickle", "wb")
        pickle.dump(coeffs, pickle_out)
        pickle_out.close()
        
        
            
            
        return self
    
    
    def scaled_corners(self, proportion = 1, size = 0.08 * 2):
        """We can scale by proportion or by size.
        For example: scaling by 2 will increase the distance of the
        corners from the center twofold.
        By increasing by size, the sides of the obstacle will increase
        their distance from the center by 'size'."""
        import math
        center = np.array(self.center)
        scaled_corners = []
        # Scaling by proportion
        for i, corner in enumerate(self.corners):
            center_to_corner = corner - center
            new_center_to_corner = center_to_corner * proportion
            scaled_corners += [(center + new_center_to_corner).reshape(1, -1).tolist()[0]]
        
        scaled_corners = tuple(scaled_corners)
        size_scaled_corners = []
        # Scaling by size
        for i, corner in enumerate(scaled_corners):
            if i == 0:
                previous_corner = scaled_corners[-1]
            else:
                previous_corner = scaled_corners[i - 1]

            vector_1 = np.array(corner) - center
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            # self.corners[i] = (   center + (vector_1 + unit_vector_1 * math.sqrt(size**2 + size**2))   ).reshape(1, -1).tolist()[0]
            size_scaled_corners += [(   center + (vector_1 + unit_vector_1 * math.sqrt(size**2 + size**2))   ).reshape(1, -1).tolist()[0]]
            
        return size_scaled_corners
    
    
    
    def random_placement(self, x_limits, y_limits):
        import random
        
        middle_x = random.uniform(*x_limits)
        middle_y = random.uniform(*y_limits)
        
        size = 0.1
        delta = size/2
        corner1 = [middle_x - delta, middle_y - delta]
        corner2 = [middle_x + delta, middle_y - delta]
        corner3 = [middle_x + delta, middle_y + delta]
        corner4 = [middle_x - delta, middle_y + delta]
        
        corners = [corner1, corner2, corner3, corner4]
        
        return corners
        
    
    def plot_obstacle(self, ax):
        
        # Plotting of obstacle
        corners = np.array(self.corners)
        corners = np.vstack((corners, corners[0, :]))
        polygon = Polygon(corners, closed=True, fill=True, fc=(0,0,0,0.1), ec=(0,0,0,1), lw=1, zorder = 1)
        ax.add_patch(polygon)
        
        """
        "Real"
        # Plotting spline obstacle
        t = np.linspace(0, 1, 10)
        x, y = [], []
        for corner in self.corners_spline:
            for t_ in t:
                p_, q_ = corner[0](t_)[0], corner[1](t_)[0]
                x_, y_ = self.fp.frenet_to_inertial(p_, q_, t_)
                x += [x_]
                y += [y_]
            ax.plot(x, y, 'y*')
        "Scaled"
        t = np.linspace(0, 1, 10)
        x, y = [], []
        for corner in self.scaled_corners_spline:
            for t_ in t:
                p_, q_ = corner[0](t_)[0], corner[1](t_)[0]
                x_, y_ = self.fp.frenet_to_inertial(p_, q_, t_)
                x += [x_]
                y += [y_]
            ax.plot(x, y, 'b.')
            """
            # p = np.array([corner[0](t_)[0] for t_ in t]).reshape(-1)
            # q = np.array([corner[1](t_)[0] for t_ in t]).reshape(-1)
            # xy = [self.fp.frenet_to_inertial(p_, q_, t_) for p_, q_, t_ in zip(p, q, t)]
            
            
            
            
            
            
            
            
            
            
            
            
            
