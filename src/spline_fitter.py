from casadi import MX, vertcat, nlpsol
import numpy as np

from .bspline import BSpline, BSplineBasis


class SplineFitter:
    def __init__(self, knot_intervals: int = 10):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.w_list, self.g_list, self.P_list = [], [], []
        self.options = {
            "print_time": False,
            "ipopt": {"print_level": 0, "max_iter": 1000, "max_cpu_time": 100},
        }

        self.n_dimensions = 2
        self.knot_intervals = knot_intervals
        self.y_min = [-20, -20]
        self.y_max = [20, 20]

    def define_knots(self, degree=3, **kwargs):
        if "knot_intervals" in kwargs:
            knot_intervals = kwargs["knot_intervals"]
            knots = np.r_[np.zeros(degree), np.linspace(0, 1, knot_intervals + 1), np.ones(degree)]
        if "knots" in kwargs:
            knot_intervals = len(knots) - 2 * degree - 1
            knots = kwargs["knots"]

        return BSplineBasis(knots, degree)

    def define_splines(self, degree, knot_intervals, n_spl, lower_bound, upper_bound, name=""):
        basis = self.define_knots(degree=degree, knot_intervals=knot_intervals)

        splines = []
        for k in range(n_spl):
            coeffs = MX.sym(name[k], len(basis))
            self.w += [coeffs]
            self.w_list += [name[k] for _ in range(len(basis))]
            splines += [BSpline(basis, coeffs)]

        for i in range(len(splines)):
            for _ in range(splines[i].coeffs.shape[0]):
                self.lbw += [lower_bound[i]]
                self.ubw += [upper_bound[i]]

        return splines

    def define_constraint(self, constraint, lower_bound, upper_bound, constraint_type="overall", name=""):
        if constraint_type == "overall":
            for i in range(len(constraint)):
                for j in range(constraint[i].coeffs.shape[0]):
                    self.g += [constraint[i].coeffs[j]]
                    self.g_list += [name]
                    self.lbg += [lower_bound[i]]
                    self.ubg += [upper_bound[i]]
            return self

        if constraint_type == "time":
            for i in range(len(constraint)):
                self.g += [constraint[i]]
                for _ in range(constraint[i].shape[0]):
                    self.g_list += [name]
                    self.lbg += [lower_bound[i]]
                    self.ubg += [upper_bound[i]]
            return self

        if constraint_type == "initial":
            for i in range(len(constraint)):
                self.g += [constraint[i].coeffs[0]]
                self.g_list += [name[i] + "_0"]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self

        if constraint_type == "final":
            for i in range(len(constraint)):
                self.g += [constraint[i].coeffs[-1]]
                self.g_list += [name[i] + "_f"]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self

        raise NotImplementedError()

    def fitting_single(self, f, y_min=[-20], y_max=[20], degree: int = 3):
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.w_list, self.g_list, self.P_list = [], [], []

        n_dimensions = len(f)
        n_samples = len(f[0])
        y = self.define_splines(
            degree=degree,
            knot_intervals=self.knot_intervals,
            n_spl=n_dimensions,
            lower_bound=y_min,
            upper_bound=y_max,
            name=["y"] * n_dimensions,
        )

        for i, t in enumerate(np.linspace(0, 1, n_samples)):
            for j in range(n_dimensions):
                self.J += (y[j](t) - f[j][i]) ** 4

        prob = {"f": self.J, "x": vertcat(*self.w), "g": vertcat(*self.g)}
        solver = nlpsol("solver", "ipopt", prob, self.options)

        arg = {"lbx": self.lbw, "ubx": self.ubw, "lbg": self.lbg, "ubg": self.ubg}
        self.solution = solver.call(arg)

        basis = y[0].basis
        coeffs = self.solution["x"].full()
        return [BSpline(basis, coeffs[len(basis) * i : len(basis) * (i + 1)]) for i in range(len(f))]
