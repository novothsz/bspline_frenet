import time
from concurrent.futures import ProcessPoolExecutor, as_completed


class Vehicle():
    def __init__(self):
        print("Vehicle created")
        
    def traj_generator(self, starting_point):
        # Lengthy process
        # print("generating..."); time.sleep(0.1)
        next_point = starting_point + 1
        return next_point
    
    def traj_generator_3D(self, starting_point_3D):
        end_point_3D = [self.traj_generator(starting_point) for starting_point in starting_point_3D]
        # print(end_point_3D)
        return end_point_3D
    
    def distributed_traj_generator_3D(self, starting_point_3D):
        return [self.traj_generator(starting_point) for starting_point in starting_point_3D]
    

if __name__ == '__main__':
    vehicle = Vehicle()
    with ProcessPoolExecutor(max_workers=3) as pool:
        start = [0, 0, 0]
        for i in range(10000):
            futures = pool.submit(vehicle.distributed_traj_generator_3D, start)
            print(futures.result())
            # print("bit of waiting"); time.sleep(1)
            start = futures.result()