def initialize_values(group):
    """Initialize ADMM decision variables and shared messages for all vehicles."""

    # Step 1: initial trajectory optimization per vehicle.
    for vehicle in group.vehicles:
        vehicle.initialize_x()

    # Step 2: exchange x-update messages.
    message_container = []
    for vehicle in group.vehicles:
        message_container += vehicle.data_exchange_x_send()

    for vehicle in group.vehicles:
        vehicle.data_exchange_x_receive(message_container)

    # Step 3: initialize vehicle-local decision variables and parameters.
    for vehicle in group.vehicles:
        vehicle.initialize_values()


def prepare(group):
    """Build all ADMM update structures and solvers for each vehicle."""

    for vehicle in group.vehicles:
        vehicle.prepare0()
        vehicle.prepare1()
        vehicle.prepare2()

    return group


def simulation_step(group):
    for vehicle in group.vehicles:
        vehicle.simulation_step()


def set_var(group, var):
    for idx, vehicle in enumerate(group.vehicles):
        if 'n_intermediate_ADMM' in var:
            vehicle.n_intermediate_ADMM = var['n_intermediate_ADMM']
            group.n_intermediate_ADMM = var['n_intermediate_ADMM']
        if 'stage' in var:
            vehicle.stage = var['stage']
            group.stage = var['stage']

        if 'new_positions' in var:
            vehicle.vehicle_positions_new['stage'] += [var['new_positions']['stage']]
            if var['new_positions']['vehicle_positions_new'] != []:
                vehicle.vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new'][idx]]
            else:
                vehicle.vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new']]

        if 'new_times' in var:
            vehicle.vehicle_positions_new['stage'] += [var['new_times']['stage']]
            if var['new_times']['vehicle_times_new'] != []:
                vehicle.vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new'][idx]]
            else:
                vehicle.vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new']]

        if 't_step' in var:
            vehicle.t_step = var['t_step']
        if 't_window_size' in var:
            vehicle.t_window_size = var['t_window_size']
        if 't_end' in var:
            vehicle.t_end = var['t_end']
        if 'knot_intervals' in var:
            vehicle.knot_intervals = var['knot_intervals']
        if 't_resolution_length' in var:
            vehicle.t_resolution_length = var['t_resolution_length']
        if 'rho' in var:
            vehicle.rho = var['rho']
        if 'rho_input' in var:
            vehicle.rho_input = var['rho_input']
        if 'rho_final_value' in var:
            vehicle.rho_final_value = var['rho_final_value']

        if 'MPC_version' in var:
            vehicle.MPC_version = var['MPC_version']
            group.MPC_version = var['MPC_version']
        if 'n_of_saved_waypoints' in var:
            vehicle.n_of_saved_waypoints = var['n_of_saved_waypoints']


def set_simulation(group, simulation=False):
    for vehicle in group.vehicles:
        vehicle.simulation = simulation
        vehicle.shift_enabled = simulation