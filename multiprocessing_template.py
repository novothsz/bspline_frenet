import time
from concurrent.futures import ProcessPoolExecutor, as_completed


class Vehicle():
    def __init__(self):
        print("Vehicle created")
        
    def traj_generator(self, list_):
        starting_point, idx = list_
        # Lengthy process
        print("generating..."); time.sleep(1)
        next_point = starting_point + 1
        return {idx : next_point}
    
    def traj_generator_3D(self, starting_point_3D):
        end_point_3D = [self.traj_generator(starting_point) for starting_point in starting_point_3D]
        # print(end_point_3D)
        return end_point_3D
    
    def distributed_traj_generator_3D(self, starting_point_3D):
        return [self.traj_generator(starting_point) for starting_point in starting_point_3D]
    

if __name__ == '__main__':
    vehicle = Vehicle()
    
    # target_function = lambda input_: {input_[0] : vehicle.traj_generator(input_[1])} 
    target_function = vehicle.traj_generator
    global_variable = 0
    with ProcessPoolExecutor(max_workers=3) as pool:
        start = [0, 1, 2]
        for i in range(10):
            global_variable += 1
            futures = [pool.submit(target_function, [start_, j]) for j, start_ in enumerate(start)] 
            print('start_eval')
            res = [f.result() for f in as_completed(futures)]
            print('eval_finished')
            
            # Flattening the dictionary, combining and gathering the results by index
            res_flatten = {}
            for d_ in res:
                res_flatten.update(d_)
            start = [res_flatten[j] for j in range(3)]
            print("Solution: " + str(start))
            
    print(global_variable)