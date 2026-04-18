from collections import defaultdict


def _build_index_map(labels):
    index_map = defaultdict(list)
    for idx, label in enumerate(labels):
        index_map[label].append(idx)
    return index_map


def _flatten_solution(solution):
    try:
        return solution["x"].full().reshape(-1).tolist()
    except Exception:
        return list(solution)


def _normalized_values(values, target_len):
    values = list(values)
    if len(values) < target_len:
        values += [0.0] * (target_len - len(values))
    return values[:target_len]


class ParamValX:
    _FIELDS = (
        "T",
        "x0",
        "xf",
        "x_intermediate",
        "a_intermediate",
        "t_intermediate",
        "v_s",
        "curvature",
        "equation_min_p",
        "equation_max_p",
        "equation_min_q",
        "equation_max_q",
        "obst",
        "z_i",
        "z_ji",
        "lambda_i",
        "lambda_ji",
    )

    def __init__(self, labels, _unused_defaults=None):
        self.P0_list = labels
        self._indices = _build_index_map(labels)
        for field in self._FIELDS:
            setattr(self, field, [0.0] * len(self._indices.get(field, [])))

    def assemble(self):
        assembled = [0.0] * len(self.P0_list)
        for field in self._FIELDS:
            idxs = self._indices.get(field, [])
            vals = _normalized_values(getattr(self, field, []), len(idxs))
            for local_idx, global_idx in enumerate(idxs):
                assembled[global_idx] = vals[local_idx]
        return assembled


class ParamValZ:
    _FIELDS = ("y", "y_j", "lambda_i", "lambda_ij")

    def __init__(self, labels, _unused_defaults=None):
        self.P0_z_list = labels
        self._indices = _build_index_map(labels)
        for field in self._FIELDS:
            setattr(self, field, [0.0] * len(self._indices.get(field, [])))

    def assemble(self):
        assembled = [0.0] * len(self.P0_z_list)
        for field in self._FIELDS:
            idxs = self._indices.get(field, [])
            vals = _normalized_values(getattr(self, field, []), len(idxs))
            for local_idx, global_idx in enumerate(idxs):
                assembled[global_idx] = vals[local_idx]
        return assembled


class DecisionVarZ:
    def __init__(self, labels, g_list, lbg, ubg):
        self.g_list = g_list
        self.lbg = lbg
        self.ubg = ubg

        self.w_z_list = labels
        self._indices = _build_index_map(labels)
        self.w0_z = []
        self.z_i = []
        self.z_ij = []

    def extract(self, solution):
        values = _flatten_solution(solution)
        self.w0_z = values
        self.z_i = [values[i] for i in self._indices.get("z_i", [])]
        self.z_ij = [values[i] for i in self._indices.get("z_ij", [])]
        return self

    def assemble(self):
        if not self.z_i:
            return self.w0_z
        assembled = [0.0] * len(self.w_z_list)
        z_i_vals = _normalized_values(self.z_i, len(self._indices.get("z_i", [])))
        z_ij_vals = _normalized_values(self.z_ij, len(self._indices.get("z_ij", [])))
        for local_idx, global_idx in enumerate(self._indices.get("z_i", [])):
            assembled[global_idx] = z_i_vals[local_idx]
        for local_idx, global_idx in enumerate(self._indices.get("z_ij", [])):
            assembled[global_idx] = z_ij_vals[local_idx]
        return assembled


class DecisionVarX:
    def __init__(self, labels, g_list, lbg, ubg):
        self.g_list = g_list
        self.lbg = lbg
        self.ubg = ubg

        self.w_list = labels
        self._indices = _build_index_map(labels)
        self.w0 = []
        self.T = []
        self.y = []
        self.a = []
        self.b = []
        self.d_tau = []

    def extract(self, solution):
        values = _flatten_solution(solution)
        self.w0 = values
        self.T = [values[i] for i in self._indices.get("T", [])]
        self.y = [values[i] for i in self._indices.get("y", [])]
        self.a = [values[i] for i in self._indices.get("a", [])]
        self.b = [values[i] for i in self._indices.get("b", [])]
        self.d_tau = [values[i] for i in self._indices.get("d_tau", [])]
        return self

    def assemble(self):
        if not self.y:
            return self.w0
        assembled = [0.0] * len(self.w_list)
        field_values = {
            "T": self.T,
            "y": self.y,
            "a": self.a,
            "b": self.b,
            "d_tau": self.d_tau,
        }
        for field, values in field_values.items():
            idxs = self._indices.get(field, [])
            vals = _normalized_values(values, len(idxs))
            for local_idx, global_idx in enumerate(idxs):
                assembled[global_idx] = vals[local_idx]
        return assembled
