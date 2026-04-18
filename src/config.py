from dataclasses import dataclass


_VALID_DFG_MODES = {"legacy", "analytic"}


@dataclass
class GroupRuntimeConfig:
    n_intermediate_ADMM: int = 1
    t_step: float = 0.01
    t_window_size: float = 0.2
    t_end: float = 0.2
    knot_intervals: int = 5
    t_resolution_length: int = 10
    rho: float = 50
    rho_input: float = 200
    rho_final_value: float = 5000
    mpc_version: str = "MPC_param"
    n_of_saved_waypoints: int = 5
    back_scaling_factor: float = 0.3
    back_rotation_factor: float = 0.4
    dfg_mode: str = "legacy"
    dfm_lookahead_ratio: float = 0.2
    dfm_lookback_ratio: float = 0.3

    def apply(self, group):
        dfg_mode = self.dfg_mode.strip().lower()
        if dfg_mode not in _VALID_DFG_MODES:
            raise ValueError(f"Invalid dfg_mode '{self.dfg_mode}'. Expected one of {_VALID_DFG_MODES}.")

        group.set_var({"n_intermediate_ADMM": self.n_intermediate_ADMM})
        group.set_var({"t_step": self.t_step})
        group.set_var({"t_window_size": self.t_window_size})
        group.set_var({"t_end": self.t_end})
        group.set_var({"knot_intervals": self.knot_intervals})
        group.set_var({"t_resolution_length": self.t_resolution_length})
        group.set_var({"rho": self.rho})
        group.set_var({"rho_input": self.rho_input})
        group.set_var({"rho_final_value": self.rho_final_value})
        group.set_var({"MPC_version": self.mpc_version})
        group.set_var({"n_of_saved_waypoints": self.n_of_saved_waypoints})
        group.set_var({"dfg_mode": dfg_mode})

        group.back_scaling_factor = self.back_scaling_factor
        group.back_rotation_factor = self.back_rotation_factor
        group.DFM_lookahead = group.vehicles[0].t_window_size * self.dfm_lookahead_ratio
        group.DFM_lookback = group.vehicles[0].t_window_size * self.dfm_lookback_ratio
        return group
