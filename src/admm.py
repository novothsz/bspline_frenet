def data_exchange_x(group):
    message_container = []
    for vehicle in group.vehicles:
        message_container += vehicle.data_exchange_x_send()

    for vehicle in group.vehicles:
        vehicle.data_exchange_x_receive(message_container)

    return group


def lambda_update_data_exchange_z(group):
    for vehicle in group.vehicles:
        vehicle.lambda_update()

    message_container = []
    for vehicle in group.vehicles:
        message_container += vehicle.data_exchange_z_send()

    for vehicle in group.vehicles:
        vehicle.data_exchange_z_receive(message_container)

    return group


def solve(group):
    for vehicle in group.vehicles:
        vehicle.x_update_prior()
    for vehicle in group.vehicles:
        vehicle.x_update()
    for vehicle in group.vehicles:
        vehicle.x_update_posterior()
    data_exchange_x(group)

    for vehicle in group.vehicles:
        vehicle.z_update_prior()
    for vehicle in group.vehicles:
        vehicle.z_update()
    for vehicle in group.vehicles:
        vehicle.z_update_posterior()

    lambda_update_data_exchange_z(group)

    return group
