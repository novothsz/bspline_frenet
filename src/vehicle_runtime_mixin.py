import math
import numpy as np

from casadi import MX

from .bspline import BSpline, definite_integral
from .spline_extra import extrapolate, shift_spline


class VehicleRuntimeMixin:
    """Lean runtime mixin extracted from legacy VehicleBasis."""

    def _init_runtime_state(self):
        self.x_intermediate_list = []
        self.t_intermediate_list = []
        self.a_intermediate_list = []
        self.a_intermediate_ID_list = []
        self.t_real_intermediate_list = []
        self.t_real_activation_list = []

        self.stage = []
        self.n_dimensions = 3
        self.state_len = self.n_dimensions * 2

        self.n_dimensions_old = 2
        self.state_len_old = self.n_dimensions_old * 2

        self.state_degree = 3
        self.radious = 0.08
        self.T = 5

        self.x0 = []
        self.xf = []
        self.x_reference = []
        self.neighbours = []
        self.obstacles = []
        self.ID = -1
        self.iteration_count = 0

        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.J = 0
        self.P = []
        self.P0 = []
        self.w0 = []
        self.w_list, self.g_list, self.P_list = [], [], []

        self.P0_assemble = []
        self.P0_z_assemble = []

        self.slack = 0.00001
        self.feasibility_slack = 0.001
        self.rho = 50
        self.rho_input = 0.1 * 10 * 2 * 100
        self.rho_final_value = 0.1 * 10 * 100
        self.rho_intermediate = 10

        self.epsilon = 0.05
        self.safety_weight = 100

        self.knot_intervals = 5
        self.t_resolution_length = self.knot_intervals + 1

        self.y_min = [-2, -2, -math.pi * 2]
        self.y_max = [2, 2, math.pi * 2]
        assert len(self.y_min) == self.n_dimensions

        u__ = 1
        self.u_min = [-u__, -u__]
        self.u_max = [u__, u__]

        self.options = {
            'print_time': False,
            'ipopt': {'print_level': 0, 'max_iter': 10000, 'max_cpu_time': 100},
        }
        self.options_z = {
            'print_time': False,
            'ipopt': {'print_level': 0, 'max_iter': 10000, 'max_cpu_time': 100},
        }

        self.initial_values = {}
        self.variable_history = {
            'y': [],
            'y_j': [],
            't_start': [],
            't_end': [],
            'xf': [],
            'x_intermediate_list': [],
            'a_intermediate_list': [],
            't_intermediate_list': [],
            't_real_intermediate_list': [],
            't_real_activation_list': [],
            'current_configuration_position': [],
            'x_update_time': [],
            'z_update_time': [],
            'a': [],
            'b': [],
            'solution': [],
            'solver_stats': [],
            'feasibility_dict': [],
            'first_time_success': [],
            'DvX_prior': [],
            'DvX_posterior': [],
            'PvX': [],
        }

        self.n_intermediate_ADMM = 1
        self.vehicle_positions_new = {'stage': [], 'vehicle_positions_new': []}
        self.MPC_version = []
        self.message_in = {}
        self.message_out = {}

    def set_position(self, position: list, position_type: str):
        assert len(position) == self.n_dimensions
        if position_type == 'initial':
            self.x0 = position + [0] * len(position)
            self.current_configuration_position = position
        elif position_type == 'final':
            self.xf = position + [0] * len(position)
            self.variable_history['xf'] += [self.xf]
        else:
            raise NotImplementedError()

        return self

    def shift_spline_v2(self, spline, t_shift):
        """Shift a spline by extrapolate+crop and transform to original basis."""
        default_basis = self.define_knots(degree=spline.basis.degree, knots=spline.basis.knots)
        spline.basis, spline.coeffs = extrapolate(spline.coeffs, t_shift, spline.basis)
        spline.basis, spline.coeffs = shift_spline(spline.coeffs, t_shift, spline.basis)
        spline = spline.scale(1, -t_shift)
        new_coeffs = default_basis.transform(spline.basis).dot(spline.coeffs)
        new_spline = BSpline(default_basis, new_coeffs)

        return new_spline

    def update_PvZ(self):
        """Update P0_z parameter values."""
        self.PvZ.y = self.DvX.y
        try:
            self.PvZ.y_j = self.message_in['y_j']
        except Exception:
            pass

        return self

    def update_PvX(self):
        if self.stage == 14 and self.ID == 0:
            kappa = True
        self.PvX.z_i = self.DvZ.z_i
        self.PvX.x0 = self.x0
        self.PvX.xf = self.xf

        t_evaluation = np.linspace(self.t_start, self.t_start + self.t_window_size, self.t_resolution_length)
        self.PvX.v_s = [
            self.fp.fx_d_spline(t_).tolist()[0][0] + self.fp.fy_d_spline(t_).tolist()[0][0]
            for t_ in t_evaluation
        ]
        self.PvX.curvature = [self.fp.fy_c_spline(t_).tolist()[0][0] for t_ in t_evaluation]
        if self.fp.equation_min_p is not None:
            self.PvX.equation_min_p = [self.fp.equation_min_p(t_).tolist()[0][0] for t_ in t_evaluation]
            self.PvX.equation_max_p = [self.fp.equation_max_p(t_).tolist()[0][0] for t_ in t_evaluation]
            self.PvX.equation_min_q = [self.fp.equation_min_q(t_).tolist()[0][0] for t_ in t_evaluation]
            self.PvX.equation_max_q = [self.fp.equation_max_q(t_).tolist()[0][0] for t_ in t_evaluation]
        self.PvX.obst = []

        if self.MPC_version == 'MPC_param':
            for obstacle in self.obstacles:
                val = t_evaluation[-1]
                val = val * (val <= 1) + 1 * (val > 1)
                cropped_corners = obstacle.cropped_corner_trajectories(
                    self.obstacle_cropped_basis, t_evaluation[0], val
                )
                for corner in cropped_corners:
                    self.PvX.obst += corner[0].coeffs.reshape(-1).tolist() + corner[1].coeffs.reshape(-1).tolist()
        else:
            for obstacle in self.obstacles:
                for t_ in t_evaluation:
                    for corner in obstacle.corners_spline:
                        self.PvX.obst += [corner[0](t_).tolist()[0][0], corner[1](t_).tolist()[0][0]]

        self.PvX.x_intermediate = self.x_intermediate_list
        self.PvX.a_intermediate = self.a_intermediate_list
        self.PvX.t_intermediate = self.t_intermediate_list

        try:
            self.PvX.z_ji = self.message_in['z_ji']
            self.PvX.lambda_ji = self.message_in['lambda_ji']
        except Exception:
            pass

        return self

    def shift_DvZ(self):
        basis = self.define_knots(degree=self.state_degree, knot_intervals=self.knot_intervals)

        z_i_coeffs_shifted = []
        for i in range(self.n_dimensions):
            idx = np.arange(len(basis) * i, len(basis) * i + len(basis))

            z_i = BSpline(basis, self.DvZ.z_i[idx[0]:idx[-1] + 1])
            z_i_shifted = self.shift_spline_v2(z_i, self.t_step)
            z_i_coeffs_shifted += z_i_shifted.coeffs.tolist()
        self.DvZ.z_i = z_i_coeffs_shifted

        z_ij_coeffs_shifted = []
        for i in range(len(self.neighbours) * self.n_dimensions):
            idx = np.arange(len(basis) * i, len(basis) * i + len(basis))

            z_ij = BSpline(basis, self.DvZ.z_ij[idx[0]:idx[-1] + 1])
            z_ij_shifted = self.shift_spline_v2(z_ij, self.t_step)
            z_ij_coeffs_shifted += z_ij_shifted.coeffs.tolist()
        self.DvZ.z_ij = z_ij_coeffs_shifted

        return self

    def shift_PvZ(self):
        basis = self.define_knots(degree=self.state_degree, knot_intervals=self.knot_intervals)

        self.PvZ.lambda_i = self.PvX.lambda_i

        lambda_ij_coeffs_shifted = []
        for i in range(len(self.neighbours) * self.n_dimensions):
            idx = np.arange(len(basis) * i, len(basis) * i + len(basis))

            lambda_ij = BSpline(basis, self.PvZ.lambda_ij[idx[0]:idx[-1] + 1])
            lambda_ij_shifted = self.shift_spline_v2(lambda_ij, self.t_step)
            lambda_ij_coeffs_shifted += lambda_ij_shifted.coeffs.tolist()
        self.PvZ.lambda_ij = lambda_ij_coeffs_shifted

        return self

    def shift_PvX(self):
        basis = self.define_knots(degree=self.state_degree, knot_intervals=self.knot_intervals)
        sol = self.solution['x'].full().reshape(-1).tolist()

        coeffs = [sol[len(basis) * i:len(basis) * (i + 1)] for i in range(self.n_dimensions)]
        y = [BSpline(basis, coeffs_) for coeffs_ in coeffs]
        y_dot = [y_.derivative() for y_ in y]
        y0 = [y_(self.t_step * 1 / self.t_window_size).tolist()[0] for y_ in y]
        y_dot0 = [y_dot_(self.t_step * 1 / self.t_window_size).tolist()[0] for y_dot_ in y_dot]
        self.x0 = y0 + y_dot0
        self.PvX.x0 = self.x0

        z_i_coeffs_shifted = []
        lambda_i_coeffs_shifted = []

        for i in range(self.n_dimensions):
            idx = np.arange(len(basis) * i, len(basis) * i + len(basis))

            z_i = BSpline(basis, self.PvX.z_i[idx[0]:idx[-1] + 1])
            z_i_shifted = self.shift_spline_v2(z_i, self.t_step)
            z_i_coeffs_shifted += z_i_shifted.coeffs.tolist()

            lambda_i = BSpline(basis, self.PvX.lambda_i[idx[0]:idx[-1] + 1])
            lambda_i_shifted = self.shift_spline_v2(lambda_i, self.t_step)
            lambda_i_coeffs_shifted += lambda_i_shifted.coeffs.tolist()

        self.PvX.z_i = z_i_coeffs_shifted
        self.PvX.lambda_i = lambda_i_coeffs_shifted

        z_ji_coeffs_shifted = []
        lambda_ji_coeffs_shifted = []
        for i in range(len(self.neighbours) * self.n_dimensions):
            idx = np.arange(len(basis) * i, len(basis) * i + len(basis))

            z_ji = BSpline(basis, self.PvX.z_ji[idx[0]:idx[-1] + 1])
            z_ji_shifted = self.shift_spline_v2(z_ji, self.t_step)
            z_ji_coeffs_shifted += z_ji_shifted.coeffs.tolist()

            lambda_ji = BSpline(basis, self.PvX.lambda_ji[idx[0]:idx[-1] + 1])
            lambda_ji_shifted = self.shift_spline_v2(lambda_ji, self.t_step)
            lambda_ji_coeffs_shifted += lambda_ji_shifted.coeffs.tolist()

        self.PvX.z_ji = z_ji_coeffs_shifted
        self.PvX.lambda_ji = lambda_ji_coeffs_shifted

    def shift_DvX(self):
        basis = self.define_knots(degree=self.state_degree, knot_intervals=self.knot_intervals)

        coeffs = [self.DvX.y[len(basis) * i:len(basis) * (i + 1)] for i in range(self.n_dimensions)]
        y = [BSpline(basis, coeffs_) for coeffs_ in coeffs]
        y_shifted = [self.shift_spline_v2(y_, self.t_step) for y_ in y]
        y_shifted_coeffs = [y_shifted_.coeffs for y_shifted_ in y_shifted]
        self.DvX.y = np.array(y_shifted_coeffs).reshape(-1).tolist()

        a_coeffs_shifted = []
        basis_a = basis
        for i in range(len(self.obstacles) * 2):
            idx = np.arange(len(basis_a) * i, len(basis_a) * i + len(basis_a))
            a = BSpline(basis_a, self.DvX.a[idx[0]:idx[-1] + 1])
            a_shifted = self.shift_spline_v2(a, self.t_step)
            a_coeffs_shifted += a_shifted.coeffs.tolist()
        self.DvX.a = a_coeffs_shifted

        b_coeffs_shifted = []
        d_tau_coeffs_shifted = []
        basis_a = basis
        for i in range(len(self.obstacles) * 1):
            idx = np.arange(len(basis_a) * i, len(basis_a) * i + len(basis_a))
            b = BSpline(basis_a, self.DvX.b[idx[0]:idx[-1] + 1])
            b_shifted = self.shift_spline_v2(b, self.t_step)
            b_coeffs_shifted += b_shifted.coeffs.tolist()

            d_tau = BSpline(basis_a, self.DvX.d_tau[idx[0]:idx[-1] + 1])
            d_tau_shifted = self.shift_spline_v2(d_tau, self.t_step)
            d_tau_coeffs_shifted += d_tau_shifted.coeffs.tolist()

        self.DvX.b = b_coeffs_shifted
        self.DvX.d_tau = d_tau_coeffs_shifted

    def simulation_step(self):
        self.t_start = self.t_start + self.t_step
        self.t_end = self.t_start + self.t_window_size

        if self.t_end > 1:
            self.t_end = 1
        if self.t_start > 1:
            self.t_start = 1

        self.shift_enabled = True

        return self

    def data_exchange_z_send(self):
        lambda_ij = np.array(self.PvZ.lambda_ij).reshape((len(self.neighbours), -1))
        z_ij = np.array(self.DvZ.z_ij).reshape((len(self.neighbours), -1))

        message = []
        for i in range(len(self.neighbours)):
            message += [{
                'sender': self.ID,
                'receiver': self.neighbours[i].ID,
                'lambda_ij': lambda_ij[i, :].tolist(),
                'z_ij': z_ij[i, :].tolist(),
            }]

        return message

    def data_exchange_z_receive(self, message):
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
        message = [{'sender': self.ID, 'receiver': -1, 'y_i': self.DvX.y}]

        return message

    def data_exchange_x_receive(self, message):
        y_j = []
        for neighbour_ in self.neighbours:
            for dict_ in message:
                if neighbour_.ID == dict_['sender']:
                    y_j += dict_['y_i']

        self.message_in['y_j'] = y_j
        self.variable_history['y_j'] += [y_j]
        return self

    def lambda_update(self):
        lambda_i_old = np.array(self.PvZ.lambda_i)
        y = np.array(self.DvX.y)
        z_i = np.array(self.DvZ.z_i)

        lambda_ij_old = np.array(self.PvZ.lambda_ij)
        y_j = np.array(self.message_in['y_j'])
        z_ij = np.array(self.DvZ.z_ij)

        lambda_i_new = lambda_i_old + self.rho * (y - z_i)
        lambda_ij_new = lambda_ij_old + self.rho * (y_j - z_ij)

        self.PvX.lambda_i = lambda_i_new.tolist()
        self.PvZ.lambda_i = lambda_i_new.tolist()
        self.PvZ.lambda_ij = lambda_ij_new.tolist()

    def prepare0(self):
        "TODO"

    def prepare1(self):
        self.setup_x_update()
        self.setup_z_update()

    def prepare2(self):
        "TODO"

    def define_MX_spline(
        self,
        degree,
        knot_intervals,
        n_spl,
        lower_bound,
        upper_bound,
        initial_value=[],
        name='',
        category='variable',
    ):
        basis = self.define_knots(degree=degree, knot_intervals=knot_intervals)

        splines = []
        for k in range(n_spl):
            coeffs = MX.sym(name[k], len(basis))
            if category == 'variable':
                self.w += [coeffs]
                self.w_list += [name[k] for i in range(len(basis))]
                if initial_value:
                    if len(initial_value[k]) == len(basis):
                        self.w0 += initial_value[k]
                    elif initial_value[k][0] is not None or initial_value[k][1] is not None:
                        w0_noise_added = np.linspace(initial_value[k][0], initial_value[k][1], len(basis))
                        import random
                        for i in range(len(w0_noise_added)):
                            w0_noise_added[i] = w0_noise_added[i] + w0_noise_added[i] * 0.05 * random.uniform(-0.5, 0.5)

                        self.w0 += w0_noise_added.tolist()
                    else:
                        import random
                        self.w0 += [random.uniform(-0.5, 0.5) for i in range(len(basis))]

                else:
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

    def check_constraint(self, constraint, lower_bound, upper_bound, constraint_type='overall', name=''):
        feasibility = []
        if constraint_type == 'overall':
            for i in range(len(constraint)):
                for j in range(constraint[i].coeffs.shape[0]):
                    feasibility += [
                        lower_bound[i] - self.feasibility_slack <= constraint[i].coeffs[j]
                        and constraint[i].coeffs[j] <= upper_bound[i] + self.feasibility_slack
                    ]
            return name, feasibility

        if constraint_type == 'time':
            for i in range(len(constraint)):
                for j in range(constraint[i].shape[0]):
                    feasibility += [
                        lower_bound[i] - self.feasibility_slack <= constraint[i].reshape(-1).tolist()[0]
                        and constraint[i].reshape(-1).tolist()[0] <= upper_bound[i] + self.feasibility_slack
                    ]
            return name, feasibility

        if constraint_type == 'initial':
            for i in range(len(constraint)):
                feasibility += [
                    lower_bound[i] - self.feasibility_slack <= constraint[i].coeffs[0]
                    and constraint[i].coeffs[0] <= upper_bound[i] + self.feasibility_slack
                ]
            return name, feasibility

        if constraint_type == 'final':
            for i in range(len(constraint)):
                feasibility += [
                    lower_bound[i] - self.feasibility_slack <= constraint[i].coeffs[-1]
                    and constraint[i].coeffs[-1] <= upper_bound[i] + self.feasibility_slack
                ]
            return name, feasibility

        if constraint_type == 'initial_param':
            raise NotImplementedError()
        if constraint_type == 'final_param':
            raise NotImplementedError()

    def define_constraint(self, constraint, lower_bound, upper_bound, constraint_type='overall', name=''):
        if constraint_type == 'overall':
            for i in range(len(constraint)):
                for j in range(constraint[i].coeffs.shape[0]):
                    self.g += [constraint[i].coeffs[j]]
                    self.g_list += [name]
                    self.lbg += [lower_bound[i]]
                    self.ubg += [upper_bound[i]]
            return self

        if constraint_type == 'time':
            for i in range(len(constraint)):
                self.g += [constraint[i]]
                for j in range(constraint[i].shape[0]):
                    self.g_list += [name]
                    self.lbg += [lower_bound[i]]
                    self.ubg += [upper_bound[i]]
            return self

        if constraint_type == 'initial':
            for i in range(len(constraint)):
                self.g += [constraint[i].coeffs[0]]
                self.g_list += [name[i]]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self

        if constraint_type == 'final':
            for i in range(len(constraint)):
                self.g += [constraint[i].coeffs[-1]]
                self.g_list += [name[i]]
                self.lbg += [lower_bound[i]]
                self.ubg += [upper_bound[i]]
            return self

        if constraint_type == 'initial_param':
            for i in range(lower_bound.shape[0]):
                self.g += [constraint[i].coeffs[0] - lower_bound[i]]
                self.g_list += [name[i] + '_part1']
                self.lbg += [0]
                self.ubg += [math.inf]
            for i in range(upper_bound.shape[0]):
                self.g += [constraint[i].coeffs[0] - upper_bound[i]]
                self.g_list += [name[i] + '_part2']
                self.lbg += [-math.inf]
                self.ubg += [0]
            return self

        if constraint_type == 'final_param':
            for i in range(lower_bound.shape[0]):
                self.g += [constraint[i].coeffs[-1] - lower_bound[i]]
                self.g_list += [name[i] + '_part1']
                self.lbg += [0]
                self.ubg += [math.inf]
            for i in range(upper_bound.shape[0]):
                self.g += [constraint[i].coeffs[-1] - upper_bound[i]]
                self.g_list += [name[i] + '_part2']
                self.lbg += [-math.inf]
                self.ubg += [0]
            return self

        raise NotImplementedError()

    def check_collision_avoidance_hyperplane(self, splines, points, hyperplane, radious, name, constraint_type='obstacle', n_samples=10):
        obst_feasibility_dict = {}
        any_ = [False]

        a, b, d_tau = hyperplane
        const1 = a[0] * splines[0] + a[1] * splines[1] - b[0]
        key, value = self.check_constraint([const1], lower_bound=[-math.inf], upper_bound=[-radious], name='eq1')
        obst_feasibility_dict[key] = value
        if any(value):
            any_[0] = True
            any_ += ['eq1']

        const2 = []
        if constraint_type == 'obstacle':
            for point in points:
                const2 += [a[0] * point[0] + a[1] * point[1] - b[0] - d_tau[0]]
            for i in range(len(points)):
                key, value = self.check_constraint(
                    [const2[i]],
                    lower_bound=[0],
                    upper_bound=[math.inf],
                    name='eq2' + '_corn_' + str(i),
                )
                obst_feasibility_dict[key] = value
                if any(value):
                    any_[0] = True
                    any_ += ['eq2' + '_corn_' + str(i)]
        else:
            raise NotImplementedError()

        const3 = a[0] * a[0] + a[1] * a[1]
        key, value = self.check_constraint([const3], lower_bound=[0.0], upper_bound=[1.0], name='eq3')
        obst_feasibility_dict[key] = value
        if any(value):
            any_[0] = True
            any_ += ['eq3']

        obst_feasibility_dict['any'] = any_
        return name, obst_feasibility_dict

    def collision_avoidance_hyperplane(self, splines, points, radious, name, constraint_type='obstacle', n_samples=10):
        a = self.define_MX_spline(
            degree=3,
            knot_intervals=self.knot_intervals,
            n_spl=len(splines),
            lower_bound=[-math.inf] * len(splines),
            upper_bound=[math.inf] * len(splines),
            initial_value=[[1, -1], [0, 0]],
            name=['a'] * self.n_dimensions_old,
        )

        b = self.define_MX_spline(
            degree=3,
            knot_intervals=self.knot_intervals,
            n_spl=1,
            lower_bound=[-math.inf],
            upper_bound=[math.inf],
            name=['b'],
        )

        d_tau = self.define_MX_spline(
            degree=3,
            knot_intervals=self.knot_intervals,
            n_spl=1,
            lower_bound=[0],
            upper_bound=[math.inf],
            name=['d_tau'],
        )

        const1 = a[0] * splines[0] + a[1] * splines[1] - b[0]
        self.define_constraint([const1], lower_bound=[-math.inf], upper_bound=[-radious], name='eq1')

        const2 = []
        if constraint_type == 'obstacle':
            for point in points:
                const2 += [a[0] * point[0] + a[1] * point[1] - b[0] - d_tau[0]]
            for i in range(len(points)):
                self.define_constraint([const2[i]], lower_bound=[0], upper_bound=[math.inf], name='eq2' + '_corn_' + str(i))

        elif constraint_type == 'spline_obstacle_t':
            t = np.linspace(0, 1, n_samples)
            for t_ in t:
                for point in points:
                    const2 += [a[0](t) * point[0] + a[1](t) * point[1] - b[0](t) - d_tau[0](t)]

            for i in range(len(const2)):
                self.define_constraint(
                    [const2[i]],
                    lower_bound=[0],
                    upper_bound=[math.inf],
                    constraint_type='time',
                    name='eq2' + '_corn_' + str(i),
                )

        elif constraint_type == 'spline_obstacle_spline_t':
            t = np.linspace(0, 1, n_samples)
            for t_ in t:
                for point in points:
                    const2 += [a[0](t) * point[0](t) + a[1](t) * point[1](t) - b[0](t) - d_tau[0](t)]

            for i in range(len(const2)):
                self.define_constraint(
                    [const2[i]],
                    lower_bound=[0],
                    upper_bound=[math.inf],
                    constraint_type='time',
                    name='eq2' + '_corn_' + str(i),
                )

        elif constraint_type == 'spline_obstacle_param':
            t = np.linspace(0, 1, n_samples)
            for t_, points_t in zip(t, points):
                for point in points_t:
                    const2 += [a[0](t_) * point[0] + a[1](t_) * point[1] - b[0](t_) - d_tau[0](t_)]

            for i in range(len(const2)):
                self.define_constraint(
                    [const2[i]],
                    lower_bound=[0],
                    upper_bound=[math.inf],
                    constraint_type='time',
                    name='eq2' + '_corn_' + str(i),
                )

        elif constraint_type == 'inter_vehicle':
            for i in range(points[0].coeffs.shape[0]):
                const2 += [a[0] * points[0].coeffs[i] + a[1] * points[1].coeffs[i] - b[0] - d_tau[0]]

            for i in range(points[0].coeffs.shape[0]):
                self.define_constraint([const2[i]], lower_bound=[0], upper_bound=[math.inf], name='eq2' + '_' + str(i))

        elif constraint_type == 'inter_vehicle_t':
            t = np.linspace(0, 1, n_samples)
            for t_ in t:
                const2 += [a[0](t) * points[0](t) + a[1](t) * points[1](t) - b[0](t) - d_tau[0](t)]

            for i in range(len(const2)):
                self.define_constraint(
                    [const2[i]],
                    lower_bound=[0],
                    upper_bound=[math.inf],
                    constraint_type='time',
                    name='eq2' + '_' + str(i),
                )
        else:
            raise NotImplementedError()

        const3 = a[0] * a[0] + a[1] * a[1]
        self.define_constraint([const3], lower_bound=[0.0], upper_bound=[1.0], name='eq3')

        self.J += self.safety_weight * definite_integral((self.epsilon - d_tau[0]) ** 2, 0, 1)

        return a

    def initialize_values(self):
        self.initial_values['y'] = self.DvX.y
        self.initial_values['z_i'] = self.DvX.y
        self.initial_values['z_ji'] = self.DvX.y * len(self.neighbours)
        self.initial_values['y_j'] = self.message_in['y_j']
        self.initial_values['z_j'] = self.message_in['y_j']
        self.initial_values['z_ij'] = self.message_in['y_j']
        self.initial_values['lambda_i'] = [1] * len(self.DvX.y)
        self.initial_values['lambda_ij'] = [1] * len(self.DvX.y) * len(self.neighbours)
        self.initial_values['lambda_ji'] = [1] * len(self.DvX.y) * len(self.neighbours)

        self.DvX = []
        self.message_in = {}
        self.variable_history = {
            'y': [],
            'y_j': [],
            't_start': [],
            't_end': [],
            'xf': [],
            'x_intermediate_list': [],
            'a_intermediate_list': [],
            't_intermediate_list': [],
            't_real_intermediate_list': [],
            't_real_activation_list': [],
            'current_configuration_position': [],
            'x_update_time': [],
            'z_update_time': [],
            'a': [],
            'b': [],
            'solution': [],
            'solver_stats': [],
            'feasibility_dict': [],
            'first_time_success': [],
            'DvX_prior': [],
            'DvX_posterior': [],
            'PvX': [],
        }

    def initialize_x(self):
        tmp_neighbours = self.neighbours
        self.neighbours = []
        rho = self.rho
        self.rho = 0
        self.setup_x_update()
        self.setup_z_update()
        self.rho = rho
        self.x_update()
        self.initial_values['w0_initial'] = self.solution['x']
        self.initial_values['DvX'] = self.DvX
        self.neighbours = tmp_neighbours
        print('Should be working, but please implement this method properly')
