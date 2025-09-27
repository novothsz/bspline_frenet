from .vehicle import Vehicle
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .environment import Environment
import random
import os
import functools
import matplotlib.path as mpltPath
from matplotlib.pyplot import cm
import copy

from .obstacle import Obstacle

class Group(Environment):
    def __init__(self, n_vehicles : int, start_position = [-0.8, 0, 0], goal_position = [0.8, 0, 0], stage = 0, seed = 0):
        self.vehicles = []
        for i in range(n_vehicles):
            vehicle = Vehicle()
            vehicle.ID = i
            vehicle.stage = stage
            self.vehicles += [vehicle]
        self.start_position = start_position
        self.goal_position = goal_position
        self.stage = stage
        self.seed = seed

        random.seed(seed)

        self.cwd = os.getcwd()
        os.system("mkdir " + str(self.cwd) + '/video/' + str(self.seed))
        os.system("mkdir " + str(self.cwd) + '/figures/' + str(self.seed))

        super().__init__()
        
        self.plot_stage = 5
        self.seed = []

        # Create figures for plotting
        self.figures = {}
        fig, ax = plt.subplots()
        self.figures["figures"] = [fig, ax]
        fig, ax = plt.subplots()
        self.figures["videos"] = [fig, ax]
        
        self.rotation_angle = 0
        self.scaling_factor = 1
        self.og_final_positions = []
        self.back_scaling_factor = 0.2 * 0
        self.back_rotation_factor = 0.2 * 0
        
        self.DFM_division = 12
        self.DFM_lookback = 1 / self.DFM_division / 2
        self.DFM_lookahead = 1 / self.DFM_division / 2
        self.MPC_version = []
        
        self.cum_rotation = 0
        self.cum_scaling = 1
        
        self.ACC_MPC_t_queue = []
        self.ACC_MPC_pos_queue = []
        
    @functools.lru_cache(maxsize=None)
    def get_obstacle_corners(self, t):
        corners = []
        for obstacle in self.vehicles[0].obstacles:
            corners_tmp = [ [corner[0](t).tolist()[0][0], corner[1](t).tolist()[0][0]] for corner in obstacle.corners_spline]
            corners += [corners_tmp]
        return corners
    
    @functools.lru_cache(maxsize=None)
    def get_scaled_obstacle_corners(self, t):
        corners = []
        for obstacle in self.vehicles[0].obstacles:
            corners_tmp = [ [corner[0](t).tolist()[0][0], corner[1](t).tolist()[0][0]] for corner in obstacle.scaled_corners_spline]
            corners += [corners_tmp]
        return corners
    
    def ACC_MPC_t_param(self):
        "DFG-MPC algorithm"

        t_sweep_start = self.vehicles[0].t_start
        t_sweep_end = self.vehicles[0].t_end
        
        # Step 1: Clear the intermediate lists
        # TODO: Can we moove this to the consutrctor of the vehicle class?
        if self.stage == 0:
            for vehicle in self.vehicles:
                vehicle.history['x_intermediate'] = []
                vehicle.history['a_intermediate'] = []
                vehicle.history['t_intermediate'] = []
                vehicle.history['t_real_intermediate'] = []
                
        for i, vehicle in enumerate(self.vehicles):
            vehicle.x_intermediate = []
            vehicle.a_intermediate = []
            vehicle.t_intermediate = []
            vehicle.t_real_intermediate = []
            vehicle.t_real_activation = []
        
        # Step 2: Sweep
        max_len_x = self.vehicles[0].n_waypoints * 3
        max_len_a = self.vehicles[0].n_waypoints * len(self.vehicles[0].obstacles) * 2 
        max_len_t = self.vehicles[0].n_waypoints
        
        # the cummulative values hold the relative formation rotation  scaling w.r.t. the original formation configuration
        cum_rotation_old = self.cum_rotation
        cum_scaling_old = self.cum_scaling
        self.sweep_ACC(t_sweep_start = t_sweep_start, t_sweep_end = t_sweep_end)
        # Setting back the cummulative values.
        self.cum_rotation = cum_rotation_old
        self.cum_scaling = cum_scaling_old
        
        
        # Step 3: Update the "current_configuration_position"
        for vehicle in self.vehicles:
            greater_ = False
            index_ = 0
            change_of_current_configuration_needed = False
            if len(vehicle.history['t_real_intermediate']) > 0: # Meaning: not the first iteration...
                # If in the next iteration we enter the active zone of an intermediate formation, then the next iteration of the DFG should assume,
                # that at the beginning of its iteration we will start from that specific formation.
                # This is reasonable, because we can assume, that the previously generated trajectories have already brought
                # the vehicles in a formation that is close-enough to it.
                for i, t in enumerate(vehicle.history['t_real_activation'][-1]):
                    greater_new = (t[0] <= t_sweep_start + vehicle.t_step)
                    # If before we were in the activation zone of an intermediate formation, but now we are not anymore. We have stepped out of it.
                    # TODO: is it correct like this?
                    if (greater_ == True) and (greater_new == False):
                        index_ = i-1 # We have already stepped out if it. So the previous i gives us the index of the formation configuration we are searching for.
                        change_of_current_configuration_needed = True
                        break
                    greater_ = greater_new
                # What happens when we step into an activation zone, but the list has only a single element, 
                # therefore greater_new will never be false again, hence we will not use the configuration?
                # Then we take the last index.
                if greater_ == True and greater_new == True:
                    index_ = len(vehicle.history['t_real_activation'][-1]) - 1
                    change_of_current_configuration_needed = True
                
                
            # We only change the "current_configuration_position" value, if 1) it needs to be changed 3) we don't get error when indexing
            if change_of_current_configuration_needed ==  True and index_ >= 0 and len(vehicle.history['t_real_intermediate']) > 0:
                idx = np.arange(int(vehicle.state_len/2)*index_,int(vehicle.state_len/2)*index_+int(vehicle.state_len/2))
                idx = np.arange(3*index_,3*index_+3)
                vehicle.current_configuration_position = np.array(vehicle.history['x_intermediate'][-1]).reshape(-1)[idx].tolist()[:3]
                self.cum_rotation = np.array(vehicle.history['x_intermediate'][-1]).reshape(-1)[idx].tolist()[2]
                self.cum_scaling = cum_scaling_old # TODO: well, what to do with this?
            vehicle.history['current_configuration_position'] += [vehicle.current_configuration_position]
                
            
        # Step 3.5: 
        # For each obsacle, that is isn't doing any problems for us, put zeros
        
        # Step 4: Replication/truncation    
        for i, vehicle in enumerate(self.vehicles):
            x_intermediate = vehicle.x_intermediate
            a_intermediate = vehicle.a_intermediate
            t_intermediate = vehicle.t_intermediate
            
            current_len_x = len(x_intermediate)
            current_len_a = len(a_intermediate)
            current_len_t = len(t_intermediate)
            
            single_len_a = len(self.vehicles[0].obstacles) * 2 
            
            # Truncation
            if len(x_intermediate) > max_len_x:
                vehicle.x_intermediate = x_intermediate[:int((current_len_x-max_len_x)/3)]
                vehicle.a_intermediate = a_intermediate[:int((current_len_a-max_len_a)/2)]
                vehicle.t_intermediate = vehicle.t_intermediate[:(current_len_t-max_len_t)]
                vehicle.t_real_intermediate = vehicle.t_real_intermediate[:(current_len_t-max_len_t)]
                
            # Replication
            if len(x_intermediate) < max_len_x:
                diff_x = max_len_x - current_len_x
                diff_a = max_len_a - current_len_a
                diff_t = max_len_t - current_len_t
                vehicle.x_intermediate = vehicle.x_intermediate + vehicle.x_intermediate[-3:] * int(diff_x/3)
                vehicle.a_intermediate = vehicle.a_intermediate + vehicle.a_intermediate[-single_len_a:] * int(diff_a/single_len_a)
                
                if max_len_a != len(vehicle.a_intermediate):
                    print('Baj van főnök!')
                    vehicle.a_intermediate = np.zeros(max_len_a).tolist()
                
                vehicle.t_intermediate = vehicle.t_intermediate + [vehicle.t_intermediate[-1]] * diff_t
                vehicle.t_real_intermediate = vehicle.t_real_intermediate + [vehicle.t_real_intermediate[-1]] * diff_t
            
            # Updating the intermediate lists with value, that have the correct length.
            vehicle.history['x_intermediate'][-1] = vehicle.x_intermediate
            vehicle.history['a_intermediate'][-1] = vehicle.a_intermediate
            vehicle.history['t_intermediate'][-1] = vehicle.t_intermediate
            vehicle.history['t_real_intermediate'][-1] = vehicle.t_real_intermediate
            
        return self


    
    # Check whether the obstacle has entered the danger zone.
    def sweep_ACC(self, t_sweep_start = 0, t_sweep_end = 1):
        """This is the function, described as Dynamic Formaiton Generator(DFG), that was described in the ACC paper.
        It works as follows:
            sweeps the time interval t \in [t_sweep_start, t_sweep_in]
            searches for collision
            if collision found, h... let's rather implemment it in a new function....
        """

        plt.close('all') # Close all figures

        def get_axes(corners):
            # Getting edge vectors
            edges = corners[:, 1:, :] - corners[:, :-1, :]
            edges = np.concatenate((edges, corners[:, :1, :] - corners[:, -1:, :]), axis=1)

            # Compiute the norms and prevent division by zero
            norms = np.linalg.norm(edges, axis=-1) + 1e-10
            axes = edges / norms[..., np.newaxis]

            return axes
        
        def project(corners, axes):
            # Project corners onto the axes
            return np.einsum('bij,bkj->bik', axes, corners)
        
        def sat_overlap(corners1, corners2):
            """
            corners1: (1, 4, 2) - corners of the first rectangle
            corners2: (n, 4, 2) - corners of the second rectangle
            """


            
            ## Collision detection using SAT ##
            axes1 = get_axes(corners1)
            axes2 = get_axes(corners2)

            # Because we are using rectangles, we only need to check 2 axes per box.
            axes1 = axes1[:, [0, 1]]
            axes2 = axes2[:, [0, 1]]

            # Combine axes for SAT test
            axes = np.concatenate((axes1, axes2), axis=1) # (n, 4, 2)

            # Project corners onto all axes
            projections1 = project(corners1, axes)
            projections2 = project(corners2, axes)

            # Finding the min and max projections
            min1 = np.min(projections1, axis=-1)
            max1 = np.max(projections1, axis=-1)
            min2 = np.min(projections2, axis=-1)
            max2 = np.max(projections2, axis=-1)

            # Check for separating axes
            amount_per_axes = np.min(np.concatenate(((max1 - min2)[..., np.newaxis], (max2 - min1)[..., np.newaxis]), axis=-1), axis=-1)
            collision_cases = ~np.any(amount_per_axes <= 0, axis=-1) # (n, 4)

            return {"sat_overlap": {
                    "axes": axes,
                    "amount": amount_per_axes[..., np.newaxis],
                    "cases": collision_cases
                    }}
        
        def inclusion_of_agent_by_ego(corners1, corners2):
            ## Inclusion of agent by ego ##
            # Inclusion check for corners2 inside corners1
            axes1_full = get_axes(corners1)         # Shape (1, 4, 2)
            proj1 = project(corners1, axes1_full)   # Shape (1, 4, 4)
            proj2 = project(corners2, axes1_full)   # Shape (n, 4, 4)

            # Min/max projections for both shapes
            min1_inc = np.min(proj1, axis=-1)  # Shape (1, 4)
            max1_inc = np.max(proj1, axis=-1)  # Shape (1, 4)
            min2_inc = np.min(proj2, axis=-1)  # Shape (n, 4)
            max2_inc = np.max(proj2, axis=-1)  # Shape (n, 4)

            # Check containment and compute shrinkage amount
            containment_mask = (min2_inc >= min1_inc) & (max2_inc <= max1_inc)  # Shape (n, 4)
            all_contained = np.all(containment_mask, axis=-1)  # Shape (n,)

            # Calculate shrinkage ratios for each axis (when contained)
            current_lengths = max1_inc - min1_inc + 1e-10  # Avoid division by zero
            required_lengths = max2_inc - min2_inc
            c2_c1_size_ratios = required_lengths / current_lengths  # Shape (n, 4), larger than 1 means corners2 is larger than corners1 on the given axis

            # Set shrink_ratios to 0 for axes where containment fails
            shrink_ratios = np.where(containment_mask, c2_c1_size_ratios, 0)
            shrinkage_amount = np.min(shrink_ratios, axis=-1)  # Shape (n,)
            # shrinkage_amount = np.where(all_contained, shrinkage_amount, 0)

            # Calculate growth factor for each axis (when not contained)
            # For each axis, compute how much corners1 needs to grow
            left_extension = np.maximum(0, min1_inc - min2_inc)  # How much to extend left
            right_extension = np.maximum(0, max2_inc - max1_inc)  # How much to extend right
            extension_needed = np.max(np.concatenate((left_extension[..., None], right_extension[..., None]), axis=-1), axis=-1)

            # Calculate growth ratio: extension / current_length
            growth_ratios = extension_needed / current_lengths  # Shape (n, 4)
            growth_amount = np.max(growth_ratios, axis=-1)  # Shape (n,) # Find maximum growth ratio across all axes (minimum required growth)
            growth_amount = np.where(all_contained, 0, growth_amount) # Set growth_amount to 0 if already contained


            # print(f"Collision cases: {collision_cases} & Inclusion cases: {inclusion_cases}")
            return {"inclusion_of_agent_by_ego": {
                    "shrinkage": shrinkage_amount,
                    "growth": growth_amount,
                    "cases": all_contained
                    }}
        
        def batched_rotate_corners(
            corners: np.ndarray,
            angle_degrees: np.ndarray
        ) -> np.ndarray:
            """
            Rotates each set of 4 corners in a batch by corresponding angles.

            Args:
                corners (np.ndarray): Input array of shape (batch_size, 4, 2)
                angle_degrees (np.ndarray): Rotation angles in degrees, shape (batch_size, 1)

            Returns:
                np.ndarray: Rotated corners with same shape as input (batch_size, 4, 2)
            """
            angles = np.radians(angle_degrees.squeeze(-1))  # Convert to (B,)
            
            # Batch-wise trigonometric computations
            cos_theta = np.cos(angles)
            sin_theta = np.sin(angles)
            
            # Construct batched rotation matrices (B, 2, 2)
            rotation_matrices = np.array([
                [cos_theta, -sin_theta],
                [sin_theta,  cos_theta]
            ]).transpose(2, 0, 1)  # Reshape to (B, 2, 2)
            
            # Batched matrix multiplication using einsum
            return np.einsum('bij,bjk->bik', corners, rotation_matrices)

        # The timestep of the DFG algorithm.
        t_step = 0.01
        
        # The initial configuration with which DFG calculates.
        # This value is sometimes being changed in "ACC_MPC_t_param".
        # t_sweep = np.arange(t_sweep_start, t_sweep_end + t_step, t_step)
        t_sweep = np.arange(t_sweep_start, t_sweep_start + t_step, t_step)
        # t_sweep_corners = np.array([self.get_obstacle_corners(t) for t in t_sweep]) # (t, n_obst, n_corners, 2)
        t_sweep_corners = np.expand_dims(np.array([self.get_obstacle_corners(t)[0] for t in t_sweep]), axis=1) # (t, n_obst, n_corners, 2) with only 1 obstacle
        vehicle_positions = np.array([vehicle.current_configuration_position[:2] for vehicle in self.vehicles])

        t_sweep_corners *= 0
        t_sweep_corners += vehicle_positions
        # t_sweep_corners += 0.28 * 2
        # t_sweep_corners *= 1.1
        # print(t_sweep_corners)


        # Tiling corners1 to include various angles
        corners1 = vehicle_positions[np.newaxis, ...]
        rotation_angles = np.arange(0, 1, 1) #  np.arange(0, 360, 10)
        # rotation_angles = np.arange(0, 360, 10)
        batched_rotation_angles = np.repeat(rotation_angles, corners1.shape[0], axis=0)[:, np.newaxis]
        corners1 = np.tile(corners1, (len(rotation_angles), 1, 1))
        corners1 = batched_rotate_corners(corners1, batched_rotation_angles)

        # Tiling corners1 to include various time steps
        corners1 = np.tile(corners1, (len(t_sweep), 1, 1)) # (1*n_angle*n_t, 4, 2) --- [t_0:[angle0, angle1, angle2, ...], t_1:[angle0, angle1, angle2, ...], ...]

        # Tiling corners1 to have the same shape as the future corners2
        corners1 = np.tile(corners1, (t_sweep_corners.shape[1], 1, 1))    # [obst_0:[t_0:[angle0, angle1, angle2, ...], t_1:[angle0, angle1, angle2, ...], ...], obst_1:[....], ...]

        # Tiling corners2 to include varoius angles and time steps
        # corners2 = t_sweep_corners.reshape(-1, 4, 2) # (t*n_obst, 4, 2) --- [t_0:[obst_0, obst_1, obst_2, ...], t_1:[obst_0, obst_1, obst_2, ...], ...]
        corners2 = t_sweep_corners # (t, n_obst, 4, 2)
        corners2 = np.repeat(corners2, len(rotation_angles), axis=1) # (t, n_obst*n_angle, 4, 2)
        corners2 = corners2.reshape(-1, 4, 2) # (t*n_obst*n_angle, 4, 2)

        corners1 = np.tile(np.array([[0, 0], [0, 2], [2, 2], [2, 0]])[np.newaxis, ...], (corners1.shape[0], 1, 1)) 
        # corners2 = np.tile(np.array([[0, 1], [0, 3], [2, 3], [2, 1]])[np.newaxis, ...], (corners2.shape[0], 1, 1)) # collision, but no inclusion
        # corners2 = np.array([[0, 2.5], [0, 3], [2, 3], [2, 2.5]])[np.newaxis, ...] # no collision, no inclusion
        corners2 = np.array([[0.1, 0.1], [0.1, 1.9], [1.9, 1.9], [1.9, 0.1]])[np.newaxis, ...] # collision, but with inclusion



        overlap_dict =  sat_overlap(corners1, corners2)
        agent_in_ego = inclusion_of_agent_by_ego(corners1, corners2)

        all_dict = {}
        all_dict.update(overlap_dict)
        all_dict.update(agent_in_ego)


        reshaped_all_dict = {}
        for key, value in all_dict.items():
            reshaped_all_dict[key] = {}
            for key2, value2 in value.items():
                    if key2 == "axes":
                        reshaped_all_dict[key].update({key2 : value2.reshape(len(t_sweep), t_sweep_corners.shape[1], len(rotation_angles), -1, 2)})
                    else:
                        reshaped_all_dict[key].update({key2 : value2.reshape(len(t_sweep), t_sweep_corners.shape[1], len(rotation_angles), -1)})

        for key, value in reshaped_all_dict.items():
            for key2, value2 in value.items():
                print(f"{key}: \t {key2}_shape: {value2.shape}, \t {key2}_value: {value2}")

        def plot_rectangles(corners1, corners2, results_dict):
            """
            Plots two rectangles with collision/inclusion annotations.
            
            Args:
                corners1: (1, 4, 2) - Ego rectangle corners
                corners2: (n, 4, 2) - Agent rectangle corners
                results_dict: Dictionary containing collision and inclusion results
            """
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot ego rectangle (corners1)
            ego_corners = corners1[0]
            ego_rect = patches.Polygon(ego_corners, closed=True, 
                                    edgecolor='blue', facecolor='lightblue', 
                                    alpha=0.7, label='Ego')
            ax.add_patch(ego_rect)
            
            # Plot agent rectangles (corners2)
            for i, agent_corners in enumerate(corners2):
                agent_rect = patches.Polygon(agent_corners, closed=True, 
                                            edgecolor='red', facecolor='salmon', 
                                            alpha=0.5, label=f'Agent {i+1}' if i==0 else "")
                ax.add_patch(agent_rect)
            
            # Add annotations for collision results
            collision_text = "Collision: " + ("Yes" if results_dict['sat_overlap']['cases'].any() else "No")
            plt.text(0.05, 0.95, collision_text, transform=ax.transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            # Add annotations for inclusion results
            agent_in_ego = results_dict['inclusion_of_agent_by_ego']['cases'].any()
            # ego_in_agent = results_dict['inclusion_of_ego_by_agent']['cases'].any()
            
            inclusion_text = (
                f"Agent in Ego: {'Yes' if agent_in_ego else 'No'}\n"
                # f"Ego in Agent: {'Yes' if ego_in_agent else 'No'}"
            )
            plt.text(0.05, 0.85, inclusion_text, transform=ax.transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            # Configure plot
            ax.set_aspect('equal')
            ax.autoscale_view()
            ax.set_title('Rectangle Collision/Containment Analysis')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add corner coordinates as text
            for i, (x, y) in enumerate(ego_corners):
                plt.text(x, y, f'E{i+1}({x:.1f},{y:.1f})', fontsize=9, ha='right')
            
            for agent_idx, agent in enumerate(corners2):
                for i, (x, y) in enumerate(agent):
                    plt.text(x, y, f'A{agent_idx+1}-{i+1}({x:.1f},{y:.1f})', 
                            fontsize=9, ha='left')
            
            plt.tight_layout()
            plt.show()
        plot_rectangles(corners1, corners2, all_dict)




        # reshaped_overlap_dict["collision"]["amount"][0, 0, 0, ...] 

        # Check, if current formation has collision
        def check_any_collision(coll_dict):
            """Check if any collision occurs in the current formation."""
            return np.any(coll_dict["collision"]["cases"][:, :, 0, :])
        # Check, which rotation angle has no collision.
        def get_safe_angles(coll_dict):
            """Check, which rotation angle has collision."""
            return np.where(~np.any(coll_dict["collision"]["cases"], axis=(0, 1, 3)))
        # Check for the smallest amount of shrinkage or growth needed to avoid collision (for each angle).
        def get_safe_size_delta(coll_dict):
            """Get the smallest amount of shrinkage or growth needed to avoid collision."""
            # Get the minimum shrinkage amount for each angle
            min_shrinkage = np.min(coll_dict["inclusion"]["shrinkage"], axis=(0, 1, 3))
            # Get the maximum growth amount for each angle
            max_growth = np.max(coll_dict["inclusion"]["growth"], axis=(0, 1, 3))
            # Combine them into a single array
            return min_shrinkage, max_growth
        # Compare the resulting rotation amount and scaling amount to the original formation and to the previous. 
        # Select the one, that has the least amount of change compared to the previous formation.
        get_safe_size_delta(reshaped_overlap_dict)
        axes = axes.reshape(len(t_sweep), -1, 4, 2) # (t, n_obst*n_angle, 4, 2)
        amount_per_axes = amount_per_axes.reshape(len(t_sweep), -1, 4, 1) # (t, n_obst*n_angle, 4, 2)
        collision_cases = collision_cases.reshape(len(t_sweep), -1) # (t, n_obst*n_angle)
        # inclusion_cases = inclusion_cases.reshape(len(t_sweep), -1) # (t, n_obst*n_angle)

        axes = axes.reshape(len(t_sweep), t_sweep_corners.shape[1], -1, 4, 2) # (t, n_obst, n_angle, 4, 2)
        amount_per_axes = amount_per_axes.reshape(len(t_sweep), t_sweep_corners.shape[1], -1, 4) # (t, n_obst, n_angle, 4,)
        collision_cases = collision_cases.reshape(len(t_sweep), t_sweep_corners.shape[1], -1) # (t, n_obst, n_angle)
        # inclusion_cases = inclusion_cases.reshape(len(t_sweep), t_sweep_corners.shape[1], -1) # (t, n_obst, n_angle)
        # allowed_configuration = np.where(inclusion_cases == True, inclusion_cases, ~collision_cases) # (t, n_obst, n_angle)

        if np.any(collision_cases[:, :, 0, :]):
            # Find the first time step where collision occurs
            problematic_indicies = np.where(~allowed_configuration[:, :, 0, :])
            collision_indices = np.where(collision_cases[:, :, 0, :]) # tuple of arrays (t, n_obst, n_angle, 0)
            collision_corners = t_sweep_corners[collision_indices[0], collision_indices[1], :, :]
            collision_time = t_sweep[collision_indices[0]]
            vehicle_positions = vehicle_positions[collision_indices[1], :]
            axes = axes[collision_indices[0], collision_indices[1], :, :]
            amount_per_axes = amount_per_axes[collision_indices[0], collision_indices[1], :, :]








        axes = axes.reshape(t_sweep_corners.shape[0], -1, 4, 2)
        amount_per_axes = amount_per_axes.reshape(t_sweep_corners.shape[0], -1, 4, 1)
        amount_per_axes = np.clip(amount_per_axes, a_min=0, a_max=None)



        # The DFG iteration!
        t_sweep_current = t_sweep_start
        while t_sweep_current <= t_sweep_end + self.TOL:
            # --> TODO: simplify this part
            # Step 1: Check if inside danger zone at time t
            all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions, self.get_scaled_obstacle_corners(t_sweep_current))
            
            # Let's check, with which obstacle we have collision.
            obstacle_idx = []
            for i, obstacle in enumerate(self.vehicles[0].obstacles):
                is_collision = all_collisions[i]
                if is_collision:
                    all_collisions2, collision2 = self.check_collision_with_obstacles(vehicle_positions, np.array(self.get_obstacle_corners(t_sweep_current))[ [i] ].tolist()) 
                    is_collision2 = collision2
                    if is_collision2:
                        obstacle_idx += [i]
                        
            # <-- TODO: simplify this part
            # Step 2: If collision has been found, find t_danger_end time.
            if obstacle_idx != []:
                t_danger_start = t_sweep_current
                t_danger_end = t_danger_start
                t_danger_step = t_step / 10 # if collision is detected, we step this 'smoothly' until no danger is detected 
                # TODO: We only look ahead for some time horizon anyway... Maybe we should simplyfy and look ahead until that time horizon and not fool around here?
                while t_danger_end <= 1:
                    t_danger_end += t_danger_step
                    all_collisions, collision = self.check_danger_zone_with_obstacles(vehicle_positions, np.array(self.get_scaled_obstacle_corners(t_danger_end))[obstacle_idx].tolist())
                    if collision == False:
                        # this is the t_danger_end we were looking for
                        self.check_danger_zone_with_obstacles(vehicle_positions, np.array(self.get_scaled_obstacle_corners(t_danger_end))[obstacle_idx].tolist())
                        break
                    
                # TODO: This is probably bullshit.
                # Step 3: Find the right formation configuration for t \in [t_danger_start, t_danger_end]
                # Form t_zizz
                t = (t_danger_start + t_danger_end) / 2
                lookback = (t_danger_end - t_danger_start) / 2 * 1.0
                lookahead = (t_danger_end - t_danger_start) / 2 * 1.0
                t_zizz = np.linspace( (t-lookback >= 0) * (t-lookback) + (t-lookback > 0) * 0,
                                          (t+lookahead <= 1) * (t+lookahead) + (t+lookahead > 1) * 1,
                                          10)
                
                # TODO: Why do we save this? We will overwrite anyway.
                # Get the least cost formation & ACTION_TAKEN
                cum_rotation = self.cum_rotation
                cum_scaling = self.cum_scaling
                
                # Find optimal formation configuration
                vehicle_positions, cum_rotation, cum_scaling, ACTION_TAKEN = self.intermediate_position_generator_SZILARD(vehicle_positions, cum_rotation, cum_scaling, t_zizz)
            
                self.cum_rotation = cum_rotation
                self.cum_scaling = cum_scaling
                
                
                # Step 4: Handle actions taken (back_transformation, yes, no_action, no_solution_found)
                # Create a t_intermediate & x_intermediate from this
                if ACTION_TAKEN == "back_transformation" or ACTION_TAKEN == "yes":
                    
                    if self.MPC_version == 'MPC_param':
                        # We need to account for the fact, that t_global_horizon != t_local_horizon
                        t_local = np.interp(t,[t_sweep_start,t_sweep_end],[0,1])
                        for i, vehicle in enumerate(self.vehicles):
                            vehicle.x_intermediate += [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]
                            vehicle.t_intermediate += [t_local]
                            t_cropped = t * (t <= t_sweep_end) + t_sweep_end * ( t > t_sweep_end )
                            vehicle.t_real_intermediate += [t_cropped]
                            vehicle.t_real_activation += [[t_danger_start, t_danger_end]]

                            if ACTION_TAKEN == "back_transformation":
                                pass
                            
                            if ACTION_TAKEN == "yes":
                                # assume, that only a single obstacle is causing trouble... (for simplicity)
                                obstacle_idx[0]
                                obst_ID = self.vehicles[0].obstacles[obstacle_idx[0]].ID
                                
                                # - outside or inside the formation?
                                a = [0, 0]
                                
                                # If this is a gate, then it has a pair. Let's get its ID as well
                                gate_pair_exists = False
                                gate_pair_ID = []
                                for obst in self.vehicles[0].obstacles:
                                    if obst.ID == obst_ID:
                                        if obst.gate_pair_ID != []:
                                            gate_pair_exists = True
                                            gate_pair_ID = obst.gate_pair_ID
                                        else:
                                            gate_pair_exists = False
                                            gate_pair_ID = obst.gate_pair_ID
                                            
                                # Okay. So for this specific obstacle, we add a, and [0, 0] for the others.
                                for obst in self.vehicles[0].obstacles:
                                    if obst.ID == obst_ID:
                                        vehicle.a_intermediate += a
                                        vehicle.a_intermediate_ID_list += [ int(obst_ID * (obst.ID == obst_ID)) + int(obst_ID * (obst.gate_pair_ID == obst_ID))]
                                    elif (gate_pair_exists and gate_pair_ID == obst.ID):
                                        vehicle.a_intermediate += [-a[0], -a[1]]
                                        vehicle.a_intermediate_ID_list += [ int(obst_ID * (obst.ID == obst_ID)) + int(obst_ID * (obst.gate_pair_ID == obst_ID))]
                                    else:
                                        vehicle.a_intermediate += [0, 0]
                                        vehicle.a_intermediate_ID_list += ["[0, 0]"]
                        
                    elif self.MPC_version == False or self.MPC_version == True:
                        for i, vehicle in enumerate(self.vehicles):
                            vehicle.x_intermediate += [vehicle_positions[i][0], vehicle_positions[i][1], cum_rotation]
                            vehicle.t_intermediate += [t]
                        
                elif ACTION_TAKEN == 'no_action':
                    pass
                elif ACTION_TAKEN == 'no_solution_found':
                    print("No solution has ben found. We need to halve the time. This should be implemented later :)")
                    assert 0
                
                t_sweep_current = t_danger_end + t_step
                
            elif obstacle_idx == []:
                t_sweep_current += t_step
           
        tmp_len = len(self.vehicles[0].t_intermediate)

        if tmp_len == 0:
            for i, vehicle in enumerate(self.vehicles):
                vehicle.x_intermediate += [vehicle_positions_original[i][0], vehicle_positions_original[i][1], self.cum_rotation]
                vehicle.a_intermediate += [0, 0] * len(vehicle.obstacles)
                t_local = np.interp(t_sweep_end,[t_sweep_start,t_sweep_end],[0,1])
                vehicle.t_intermediate += [t_local]
                vehicle.t_real_intermediate += [t_sweep_end]

        # Saving stuff to the history
        for i, vehicle in enumerate(self.vehicles):
            vehicle.history['x_intermediate'] += [vehicle.x_intermediate]
            vehicle.history['a_intermediate'] += [vehicle.a_intermediate]
            vehicle.history['t_intermediate'] += [vehicle.t_intermediate]
            vehicle.history['t_real_intermediate'] += [vehicle.t_real_intermediate]
            vehicle.history['t_real_activation'] += [vehicle.t_real_activation]
            
        return self
        

    
    def intermediate_position_generator_SZILARD(self, vehicle_positions, cum_rotation, cum_scaling, t):
        """ Checks for collision. 
        If no collision:
            - rotate/scale back -> save it to t&x_intermediate
            - do nothing
        If collision:
            - save it to t&x_intermediate
        """        
        
        ACTION_TAKEN = []
        """The goal of this function is to receive a set of vehicle positions and calculate a rotated-scaled frame, that does not collide with 
        obstaacles at the given time-point.
        If t is a list of time values, then each of these time values will be checked for collision"""
        if type(t) != list and type(t) != type(np.array([])):
            if t >= 1 - self.TOL and t <= 1 + self.TOL:
                self.back_rotation_factor = 1
                self.back_scaling_factor = 1
        elif type(t) == list or type(t) == type(np.array([])):
            if any([(t_ >= 1 - self.TOL and t_ <= 1 + self.TOL) for t_ in t]):
                self.back_rotation_factor = 1
                self.back_scaling_factor = 1
                
        # Step 0: first always try to turn&scale it back... :)
        #Backturning
        rotation_angle = -1 * self.back_rotation_factor * cum_rotation
        # --minimum rotation value--
        if -5/360 * 2 * math.pi <= rotation_angle < 0.0 or 0.0 < rotation_angle <= -5/360 * 2 * math.pi:
            rotation_angle = -1 * cum_rotation
        
        # Backscaling
        deviance = abs(1 - 1 / cum_scaling)
        deviance *= self.back_scaling_factor
        
        # if the formation is larger, than the reference size, then we want to shrink it
        if cum_scaling >= 1:
            scaling_factor = 1 - deviance
        # if the formation is smaller, than the reference size, then we want to expand it
        else:
            scaling_factor = 1 + deviance
            
        # --minimum scaling value--
        # if the back-scaling factor changes the size less then 10%, then we scale back completely
        if 0.9 <= scaling_factor < 1.0 or 1.0 < scaling_factor <= 1.1:
            deviance = abs(1 - 1 / cum_scaling)
            if cum_scaling >= 1:
                scaling_factor = 1 - deviance
            else:
                scaling_factor = 1 + deviance
        
        vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
        vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
        collision_saved = []
        for t_ in t:
            all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, self.get_scaled_obstacle_corners(t_))

            collision_saved += [False]
            if collision == True:
                collision_saved[-1] = True
                break
        
        if all(collision is False for collision in collision_saved):
            if rotation_angle == 0 and scaling_factor == 1:
                ACTION_TAKEN = "no_action"
            else:
                ACTION_TAKEN = "back_transformation"
            return vehicle_positions_scaled_rotated, cum_rotation + rotation_angle, cum_scaling * scaling_factor, ACTION_TAKEN
        
        # Otherwise, if we cannot rotate&scale back, find something else:
        # Step 1: generate possible rotation angles & scaling factors
        degree_step = 5
        radian_step = degree_step/360 * 2 * math.pi
        rotation_angles = [[0 + radian_step * i, 0 - radian_step * i] for i in range(1, int( (math.pi/2) / radian_step))]
        rotation_angles = np.array(rotation_angles).reshape(-1).tolist()
        
        scaling_step = 1.5
        scaling_factors_shrink = [  1 / (scaling_step ** i)  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
        scaling_factors_expand = [  1 * scaling_step ** i  for i in range(0, math.floor(abs(math.log(0.25) / math.log(scaling_step))))  ]
        scaling_factors = scaling_factors_shrink + scaling_factors_expand
        scaling_factors = [1, 0.9, 0.8, 0.6, 1.1, 1.2, 1.4, 1.8, 2.0, 2.5, 3.0, 4.0]
        
        # Step 2: We iterate through all possible rotation & scaling possibilities
        # Then, in b) we check, if that specific rotation & sacling results in collision between [t0, tf] or not.
        costs = []
        vehicle_positions_new_saved = []
        rotation_angle_new_saved = []
        scaling_factor_new_saved = []
        collision_saved = []
        for scaling_factor in scaling_factors:
            vehicle_positions_scaled = self.scale_formation(vehicle_positions, scaling_factor)
            for rotation_angle in rotation_angles:
                
                vehicle_positions_scaled_rotated = self.rotate_formation(vehicle_positions_scaled, rotation_angle)
                # Step 2b) check for each t_ in t if collision happens. If yes, do not check further, 
                # the given scaling factor & rotation angle is not good.
                
                # First let's save these :)
                vehicle_positions_new_saved += [vehicle_positions_scaled_rotated]
                rotation_angle_new_saved += [rotation_angle]
                scaling_factor_new_saved += [scaling_factor]
                collision_saved += [False]
                
                collision_saved_tmp = []
                for t_ in t:
                    all_collisions, collision = self.check_collision_with_obstacles(vehicle_positions_scaled_rotated, self.get_obstacle_corners(t_))
                    collision_saved_tmp += [collision]
                    if collision == True:
                        collision_saved[-1] = True
                        costs += [math.inf]
                        break
                # if for any t_ we did not break and put inf cost into costs, then calculate a proper cost   
                if all(collision is False for collision in collision_saved_tmp):
                    # Step 2c) if no collision happens, calculate a cost for the given rotation & scaling combo
                    cost = self.formation_change_cost_calculator(vehicle_positions, vehicle_positions_scaled_rotated, rotation_angle, scaling_factor)
                    costs += [cost]
                    
        max_distance = self.max_distance_during_turning(vehicle_positions, vehicle_positions_new_saved)
        # if we cannot find good solution, do nothing
        if all(collision_saved):
            vehicle_positions_new = vehicle_positions
            cum_rotation = 0
            cum_scaling = 1
            print("We have a problem boss! Every formation candidate collides :/")
            ACTION_TAKEN = "no_solution_found"
        else: 
            # Now we have the costs and everything in order.
            # Let's find the least cost value.
            cost_min = min(costs)
            cost_min_idx = costs.index(cost_min)
            # And the ideal position, rotation angle & scaling factor is:
            vehicle_positions_new = vehicle_positions_new_saved[cost_min_idx]
            rotation_angle_new = rotation_angle_new_saved[cost_min_idx]
            scaling_factor_new = scaling_factor_new_saved[cost_min_idx]
            
            cum_rotation += rotation_angle_new
            cum_scaling *= scaling_factor_new
            
            ACTION_TAKEN = "yes"
            
        return vehicle_positions_new, cum_rotation, cum_scaling, ACTION_TAKEN

     
    def max_distance_during_turning(self, vehicle_positions_original , vehicle_positions_new_saved):
        max_distance = -math.inf
        # for each formation candidate
        for vehicle_positions in vehicle_positions_new_saved:
            # for each vehicle
            for pos_new, pos_original in zip(vehicle_positions, vehicle_positions_original):
                distance = np.sqrt((pos_new[0] - pos_original[0])**2 + (pos_new[1] - pos_original[1])**2)
                if distance > max_distance:
                    max_distance = distance
                
        return max_distance
             
    def formation_change_cost_calculator(self, vehicle_positions_original, vehicle_positions_new, rotation_angle = 0, scaling_factor = 1):
        cost = 0
        alpha_distance = 0.1 * 1 * 0
        alpha_rotation = 0.1
        alpha_scaling_up = 1000
        alpha_scaling_down = 100
        for original, new in zip(vehicle_positions_original, vehicle_positions_new):
            cost += alpha_distance * ((original[0] - new[0])**2 + (original[1] - new[1])**2)
            
        cost += alpha_rotation * abs(rotation_angle)
        if scaling_factor > 1:
            cost += alpha_scaling_up * abs(1-scaling_factor)
        if scaling_factor < 1:
            cost += alpha_scaling_down * abs(1-scaling_factor)
                
        return cost 
    
    def rotate_formation(self, vehicle_positions, angle):
        vehicle_positions_new = []
        for position in vehicle_positions:
            vehicle_positions_new += [self.rotate_vector(position, angle)]
            
        return vehicle_positions_new
        
    def scale_formation(self, vehicle_positions, scaling_factor):
        vehicle_positions_new = []
        for position in vehicle_positions:
            vehicle_positions_new += [self.scale_vector(position, scaling_factor)]
            
        return vehicle_positions_new
        
    def rotate_vector(self, vector, angle):
        a1, a2 = vector[0], vector[1]
        
        return (a1 * math.cos(angle) - a2 * math.sin(angle), \
                        a1 * math.sin(angle) + a2 * math.cos(angle))
            
    def shift_vector(self, vector1, vector2):
        a1, a2 = vector1[0], vector1[1]
        b1, b2 = vector2[0], vector2[1]
        
        return [a1 + b1, a2 + b2]
        
    def scale_vector(self, vector, scaling_factor):
        scaled_vector = [vector[0] * scaling_factor, vector[1] * scaling_factor]
        
        return scaled_vector
    
    def check_danger_zone_with_obstacles(self, vehicle_positions, obstacle_corners):
        s_danger = -math.inf
        for position in vehicle_positions:
            vehicle_distance_from_origo = np.sqrt((0-position[0])**2 + (0-position[1])**2)
            if vehicle_distance_from_origo > s_danger:
                s_danger = vehicle_distance_from_origo
        # print(s_danger)
        s_danger = 0.6988905493709299
        # Check if any of the obstacles are in the danger zone
        corner_inside = []
        
        "New, highly professional code"
        for corners in obstacle_corners:
            # 1. Obtain obstacle center
            obst_center = self.get_obstacle_center(corners)
            # 2. Transform obstacle & circle
            corners = [ [corner[0] - obst_center[0], corner[1] - obst_center[1]] for corner in corners  ]
            d_zone_center = [-obst_center[0], -obst_center[1]]
            # 3. calculate angle of obstacle
            obst_angle = self.get_obstacle_angle(corners)
            # 4. rotate obstacle & circle
            new_corners = self.rotate_formation(corners, -obst_angle)
            new_d_zone_center = self.rotate_formation([d_zone_center], -obst_angle)
            # 5. check intersection
            is_intersection = self.intersects([list(new_d_zone_center[0]), s_danger], new_corners)
            corner_inside += [is_intersection]
            
        return corner_inside, any(corner_inside)
        
    def get_obstacle_center(self, corners):
        mean_x = np.array([])
        mean_y = np.array([])
        
        for corner in corners:
            mean_x = np.append(mean_x, corner[0])
            mean_y = np.append(mean_y, corner[1])
            
        return [np.mean(mean_x), np.mean(mean_y)]
    
    def get_obstacle_angle(self, corners):
        corners = corners + [corners[0]]
        edge_vectors = [ [p2[0] - p1[0], p2[1] - p1[1]]  for p1, p2 in zip(corners[:-1], corners[1:])]
        angles = np.array([])
        for vector in edge_vectors:
            vector = np.array(vector)
            x_axis = np.array([1, 0])
            alpha = math.acos( np.dot(vector, x_axis) / (np.linalg.norm(vector) * 1))
            angles = np.append(angles, alpha)
            
        min_rot_angle = max(angles)
        for angle in angles:
            min_rot_angle = min_rot_angle * (angle < 0 or angle > min_rot_angle) + angle * (not(angle < 0 or angle > min_rot_angle))
        
        return min_rot_angle
    
    def intersects(self, circle, rect):
        # https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
    
        circle_x = circle[0][0]
        circle_y = circle[0][1]
        circle_radious = circle[1]
        rect_x, rect_y = self.get_obstacle_center(rect)
        rect_width = abs(rect[0][0] * 2)
        rect_height = abs(rect[0][1] * 2)
        circleDistance_x = abs(circle_x - rect_x)
        circleDistance_y = abs(circle_y - rect_y)
    
        if (circleDistance_x > (rect_width/2 + circle_radious)):
            return False
        if (circleDistance_y > (rect_height/2 + circle_radious)):
            return False
        if (circleDistance_x <= (rect_width/2)):
            return True
        if (circleDistance_y <= (rect_height/2)):
            return True
    
        cornerDistance_sq = (circleDistance_x - rect_width/2)**2 + \
                             (circleDistance_y - rect_height/2)**2
    
        return cornerDistance_sq <= circle_radious**2
    
    def check_collision_with_obstacles(self, vehicle_positions, obstacle_corners):
        any_inside = []
        for corners in obstacle_corners:
            for position in vehicle_positions:
                any_inside += [self.check_collision_with_obstacle(position, corners)]
                        
        return any_inside, any(any_inside)
                    
    
    def check_collision_with_obstacle(self, vehicle_pos, obstacle_corners):
        path = mpltPath.Path(obstacle_corners)
        inside = path.contains_points(np.array([vehicle_pos]))[0]
        return inside
    
    
    def set_group_position(self, position : list, targetHeight : float = 0.8,  position_type : str = 'initial'):
        if position_type == 'initial':
            positions = self.ellipse_generator(centerpoint = self.start_position, n_positions = len(self.vehicles), a = self.vehicles[0].radious * 9, b = self.vehicles[0].radious * 5,
                                               ellipse_rotation = math.pi / 2)
            
        elif position_type == 'final':
            positions = self.ellipse_generator(centerpoint = self.goal_position, n_positions = len(self.vehicles), a = self.vehicles[0].radious * 9, b = self.vehicles[0].radious * 5,
                                               ellipse_rotation = math.pi / 2)
            self.og_final_positions = positions
        else:
            NotImplementedError()

        # Apply the positions for all vehicles.
        for i in range(len(self.vehicles)):
            self.vehicles[i].set_position(position = positions[i], position_type = position_type)

    def ellipse_generator(self, centerpoint : list, n_positions : int, a : float, b : float, 
                          ellipse_rotation : float = 0, vehicles_rotation : float = math.pi / 4,
                          ellipse_scale_x : float = 1, ellipse_scale_y : float = 1):

        # Rotating vehicles on the ellipse
        alpha = 0 + vehicles_rotation
        # Scaling
        a *= ellipse_scale_x
        b *= ellipse_scale_y
        positions = []
        for i in range(n_positions):
            positions += [ [centerpoint[0] + a * np.cos(alpha), centerpoint[1] + b * np.sin(alpha), centerpoint[2]] ] # [p, q, phi]
            alpha += np.pi * 2.0 / n_positions
        
        # Rotating the ellipse itself with the vehicles already in place
        if ellipse_rotation != 0 and centerpoint[:2] == [0.0, 0.0]:
            for i, pos in enumerate(positions):
                positions[i][:2] = self.rotate_vector(pos[:2], ellipse_rotation)
                
        if ellipse_rotation != 0 and centerpoint[:2] != [0.0, 0.0]:
            for i, pos in enumerate(positions):
                positions[i][:2] = self.rotate_vector(pos[:2], ellipse_rotation)
                positions[i][:2] = self.shift_vector(positions[i][:2], centerpoint[:2])
                
        return positions
    
    def add_obstacles(self, obstacles : list):
        for i in range(len(self.vehicles)):
            self.vehicles[i].obstacles = obstacles
        return self

    def organise_neighbours(self):
        """Sets up the l_neighbours for the vehicles in the group. Every agent
        n number of neighbours are assigned. The value n is hard-coded in
        the is_this_my_neighbour() function.

        Parameters
        ----------

        Returns
        -------
        self
        """
        for i in range(len(self.vehicles)):
            neighbours = []
            for j in range(len(self.vehicles)):
                my_id = self.vehicles[i].ID
                neighbour_id = self.vehicles[j].ID
                condition = self.is_this_my_neighbour(my_id = my_id, neighbour_id = neighbour_id)
                if condition == True:
                    neighbours += [self.vehicles[j]]
            self.vehicles[i].neighbours = neighbours
        return self

    def is_this_my_neighbour(self, my_id : int, neighbour_id : int):
        """Helper function, which tells wether neighbour_id is a
           neighbour of my_id or not
           (currently 2-distance neighbourhood is hard-coded -> max_on)

        Parameters
        ----------
        my_id : int
            ID of current vehicle
        neighbour_id : int
            ID of the other vehicle, whose neighbourhood is questioned

        Returns
        -------
        res : bool
            True if neighbour_id is a neighbour, False otherwise
        """
        # First we have to create a list of numbers
        old_list = np.linspace(0, len(self.vehicles) - 1, len(self.vehicles))
        # We cut the list where I am at. Put the first cut in to the front, the rest to the back
        # Example: [0, 1, 2!, 3, 4, 5, 6] -> [2!, 3, 4, 5, 6, 0, 1]
        new_list = np.append( old_list[my_id:] , old_list[0:my_id] )
        old_list = new_list
        max_on = 4 # max_observed_neighbours. How many neighbours we have on our right and on our left. 2-> 2+2=4 neighbours
        if(max_on + 1 < len(new_list) and 0 < len(new_list)):
            if any([neighbour_id == i for i in old_list[1:max_on + 1]]) or any( [neighbour_id == i for i in old_list[-max_on:]] ):
                res = True
            else:
                res = False
        elif max_on + 1 >= len(new_list):
            # No need to check anything further, because in the outer
            # loop we are only looking at potential neighbours anyway
            if my_id != neighbour_id:
                res = True
            else:
                res = False
        else:
            res = False
        return res


    def initialize_values(self):
        """ This function performs a single optimization step, whereas the
        decision variables and parameters that will be used later in the ADMM iteration
        are initialized.
        It lets the vehicles to generate a trajectory from the starting position
        to the goal position, without regarding each other, but avoiding the obstacles.
        The duplicate variables are set to equal their original counterparts,
        the a, b and d_tau values are initialized with the values found in this
        optimization stepd and all lambda values are set to 1.
        The data_exchange functions are reused here to exchange the data between
        the agents.
        
        Besides the below step the following steps have to be taken in the 
        vehicle group:
            - define initialize_x()
            - define initialize_values()
            - update DvX.w0, DvZ.w0, PvX, PvZ before creating the solver in setup_x_update() and setup_y_update()
        """

        "Step 1: trajectory optimization"
        for i in range(len(self.vehicles)):
            self.vehicles[i].initialize_x()

        "Step 2: exchanging solution"
        # Collecting messages
        message_container = []
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_x_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_x_receive(message_container)

        "Step 3: initializing decision variables & parameters"
        # Initializing decision variables and parameters.
        for i in range(len(self.vehicles)):
            self.vehicles[i].initialize_values()

    def prepare(self):
        """ Requests all vehicles to perform the preparation processes for
        creating the necessary variables and solver that are needed for the
        ADMM iteration.
        """

        for i in range(len(self.vehicles)):
            self.vehicles[i].prepare()

        return self
    
    def simulation_step(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].simulation_step()
            
    def set_var(self, var):
        for i in range(len(self.vehicles)):
            if 'n_intermediate_ADMM' in var:
                self.vehicles[i].n_intermediate_ADMM = var['n_intermediate_ADMM']
                self.n_intermediate_ADMM = var['n_intermediate_ADMM']
            if 'stage' in var:
                self.vehicles[i].stage = var['stage']
                self.stage = var['stage']
                
            if 'new_positions' in var:
                self.vehicles[i].vehicle_positions_new['stage'] += [var['new_positions']['stage']]
                if var['new_positions']['vehicle_positions_new'] != []:
                    self.vehicles[i].vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new'][i]]
                else:
                    self.vehicles[i].vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new']]
            if 'new_times' in var:
                self.vehicles[i].vehicle_positions_new['stage'] += [var['new_times']['stage']]
                if var['new_times']['vehicle_times_new'] != []:
                    self.vehicles[i].vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new'][i]]
                else:
                    self.vehicles[i].vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new']]
                    
            if 't_step' in var:
                self.vehicles[i].t_step = var['t_step']
            if 't_window_size' in var:
                self.vehicles[i].t_window_size = var['t_window_size']
            if 't_end' in var:
                self.vehicles[i].t_end = var['t_end']
            if 'knot_intervals' in var:
                self.vehicles[i].knot_intervals = var['knot_intervals']
            if 't_resolution_length' in var:
                self.vehicles[i].t_resolution_length = var['t_resolution_length']
            if 'rho' in var:
                self.vehicles[i].rho = var['rho']
            if 'rho_input' in var:
                self.vehicles[i].rho_input = var['rho_input']
            if 'rho_final_value' in var:
                self.vehicles[i].rho_final_value = var['rho_final_value']
                
            if 'MPC_version' in var:
                self.vehicles[i].MPC_version = var['MPC_version']
                self.MPC_version = var['MPC_version']
            if 'n_waypoints' in var:
                self.vehicles[i].n_waypoints = var['n_waypoints']
                
                
                
    def set_simulation(self, simulation = False):
        for i in range(len(self.vehicles)):
            self.vehicles[i].simulation = simulation
            self.vehicles[i].shift_enabled = simulation
        
    def data_exchange_x(self):
        # self.plot_frenet_view()
        """
        2) data_exchange_x(), where these values are shared between agents.
        """
        message_container = []
        # Collecting messages
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_x_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_x_receive(message_container)
            
        return self
    
    def lambda_update_data_exchange_z(self):
        """
        4) lambda_update(), which updates the lambda values.
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].lambda_update()

        """
        5) data_exchange_z, where z_i, z_ij, lambda_i, lambda_ij are shared.
        """
        message_container = []
        # Collecting messages
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_z_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_z_receive(message_container)
        
        return self
        
        

    def solve(self):
        
        # self.intermediate_position_generator()
        """
        1) x_update(), which optimizes the trajectory of the given vehicle.
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].x_update_prior()
        for i in range(len(self.vehicles)):
            self.vehicles[i].x_update()
        for i in range(len(self.vehicles)):
            self.vehicles[i].x_update_posterior()
            
        """
        2) data_exchange_x(), where these values are shared between agents.
        """
        message_container = []
        # Collecting messages
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_x_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_x_receive(message_container)

        """
        3) z_update(), optimizing the the duplicate variables z and z_ij.
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].z_update_prior()
        for i in range(len(self.vehicles)):
            self.vehicles[i].z_update()
        for i in range(len(self.vehicles)):
            self.vehicles[i].z_update_posterior()

        """
        4) lambda_update(), which updates the lambda values.
        """
        for i in range(len(self.vehicles)):
            self.vehicles[i].lambda_update()

        """
        5) data_exchange_z, where z_i, z_ij, lambda_i, lambda_ij are shared.
        """
        message_container = []
        # Collecting messages
        for i in range(len(self.vehicles)):
            message_container += self.vehicles[i].data_exchange_z_send()

        # Broadcasting messages
        for i in range(len(self.vehicles)):
            self.vehicles[i].data_exchange_z_receive(message_container)


        return self

    def generate_obstacles(self, seed : int = 0):
        # There are the following types of obstacles:
            # - obstacles on the path: these obstacles are created using the ellipse generator algorithm.
            # It's centerpoint, alpha, a, b is a random number in the Frenet frame (all of which in a defined bound).
            # Then, the corners are transformed from Frenet to Inertial.
            # 
            # - gates: two corners of each obstacle, that form a gate are generated with the 
            # ellipse generator algorithm. However alpha is always zero and b has a minimum value.
            # (both of these constarints ensure, that there is a tunnel, kinda parallel with the Frenet path so that 
            # the DFG algorithm will be able to find a solution.)
            # The corners are transformed from Frenet to Inertial and extended to the environment limits.
            #
            # - wall on one side: same as the gate, but drops one of the obstacle, that forms a gate.
        
        # Spacing of the obstacles:
            # randomly, but at least t_spacing between each obstacle.
            # no obstacle is allowed at the end
        
        # Deviation from the path
        centerpoint_x_bound = [-0.1, 0.1]
        centerpoint_y_bound = [-0.1, 0.1]
        # Ellipse bounds
        a_bound = [0.1, 0.7]
        b_bound = [0.1, 0.7] # 1.5]
        alpha_bound = [-math.pi/2, math.pi/2]
        

        # Generate obstacles along the way
        obstacles = []
        t_tmp = [0.35, 0.65]
        for i in range(len(t_tmp)):
            centerpoint = [random.uniform(centerpoint_x_bound[0], centerpoint_x_bound[1]), \
                           random.uniform(centerpoint_y_bound[0], centerpoint_y_bound[1]), 0 ]
                
            a = random.uniform(a_bound[0], a_bound[1])
            b = random.uniform(b_bound[0], b_bound[1])
            alpha = random.uniform(alpha_bound[0], alpha_bound[1])
            ellipse_corners = self.ellipse_generator( centerpoint = centerpoint, n_positions = 4, a = a, b = b, 
                                                      ellipse_rotation = alpha, vehicles_rotation = math.pi / 4)
            
            obstacle_corners = [corner[:2] for corner in ellipse_corners]
            t = random.uniform(a_bound[0], a_bound[1])
            # we need to place them at random location along the path
            # this is done by converting their frenet coordinates to the inertial frame at random times
            obstacle_corners = [self.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i]) for corner in obstacle_corners]
            obstacles += [Obstacle(ID = i, corners = obstacle_corners)]
        
        # Generate gates
        gate_gap_bound = [3.5 * self.vehicles[0].radious, 10 * self.vehicles[0].radious]
        gate_length_bound = 0.3
        
        t_tmp = [0.2, 0.5, 0.8]
        for i in range(len(t_tmp)):
            gate_points_tmp = random.uniform(gate_gap_bound[0], gate_gap_bound[1])
            # The lower part of the gate
            # corners = [top-right, top_left]
            gate1_inside_corners = [ [gate_length_bound / 2, -gate_points_tmp], \
                                     [-gate_length_bound / 2, -gate_points_tmp] ]
            # The upper part of the gate
            # corners = [bottom-left, bottom-right]
            gate2_inside_corners = [ [gate_length_bound / 2, gate_points_tmp], \
                                     [-gate_length_bound / 2, gate_points_tmp] ]
                
            # transforming the frenet coordinates to inertial frame at random times
            g1 = [list(self.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i])) for corner in gate1_inside_corners]
            g2 = [list(self.fp.frenet_to_inertial(corner[0], corner[1], t_tmp[i])) for corner in gate2_inside_corners]
            
            # Extending till the edge of the environment
            g1 = g1 + [[g1[-1][0], -6]] + [[g1[0][0], -6]]
            g2 = g2 + [[g2[-1][0],  6]] + [[g2[0][0],  6]]
            obstacles += [Obstacle(ID = 3+i*2, corners = g1)]
            obstacles += [Obstacle(ID = 3+i*2 + 1, corners = g2)]
            # Sharing ID-s between gate pairs
            obstacles[-2].gate_pair_ID = obstacles[-1].ID
            obstacles[-1].gate_pair_ID = obstacles[-2].ID


        return obstacles

    def plot_setup(self):
        """This function plots the environment and the group starting and final
        position"""

        fig, ax = plt.subplots()
        self.vehicles[0].plot_environment(ax, 0)
        for obstacle in self.vehicles[0].obstacles:
            obstacle.plot_obstacle(ax)
        for vehicle in self.vehicles:
            x, y = self.fp.frenet_to_inertial(vehicle.xf[0], vehicle.xf[1], 0)
            x0_plot = ax.plot(x, y, 'ko', markersize = 3)
            x, y = self.fp.frenet_to_inertial(vehicle.xf[0], vehicle.xf[1], 1)
            xf_plot =  ax.plot(x, y, 'go', markersize = 3)
        x0_plot[0].set_label("Starting positions")
        xf_plot[0].set_label("Final positions")
        ax.legend(fontsize = 'x-small')
            
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_aspect('equal', adjustable='box')

    def plotter(self, iternum : int = 0, seed = ''):
        """This function plots the trajectories calculated by each of the agent.
        It also plots the Frenet path.
        """
        # Plotting parameters for a single vehicle
        # self.l_groups[0].vehicles[0].plot_params(folder)
        # plt.close('all')

        # Plotting trajectories
        # https://stackoverflow.com/questions/34442791/pass-plot-to-function-matplotlib-python

        # Plotting frenet path
        # self.vehicles[0].fp.plot_path(ax, 1) # t = np.interp(i,[0,N-1],[0,1])

        # Plotting trajectory of the vehicles
        fig, ax = self.figures["figures"]
        for i in range(len(self.vehicles)):
            ax = self.vehicles[i].plot_vehicle_trajectories(ax)

        # Axis related stuff
        ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_aspect('equal', adjustable='box')
        # Saving figure to folder
        fig.savefig(self.cwd + '/figures/' +'{:0>1d}'.format(self.stage) + '{:0>2d}'.format(iternum) +'.png', dpi = 200)
        ax.clear()

        return self

    def plot_vehicle_trajectories_gradient(self):
        fig, ax = self.figures["figures"]
        for i in range(len(self.vehicles)):
            ax = self.vehicles[i].plot_vehicle_trajectories_gradient(ax)

        # Axis related stuff
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_title("Trajectory change over the iterations")
        ax.set_aspect('equal', adjustable='box')
        # Saving figure to folder
        fig.savefig(self.cwd + '/figures/' + 'stage_' + '{:0>1d}'.format(self.stage) + 'trajectory_gradients' +'.pdf')
        ax.clear()

    def calculate_formation_error(self):
        
        angle_errors_intermediate_all = []
        for intermediate_ADMM_idx in range(self.n_intermediate_ADMM):
            angle_errors = []
            for vehicle in self.vehicles:
                t, angle_errors_tmp = vehicle.calculate_formation_error(intermediate_ADMM_idx = intermediate_ADMM_idx)
                angle_errors += [angle_errors_tmp]
            
            angle_errors_mean = []
            for i in range(len(angle_errors_tmp)):
                angle_errors_sum_tmp = np.array(angle_errors[0][0]) * 0
                for j in range(len(self.vehicles)):
                    # angle_errors_sum_tmp += np.linalg.norm(angle_errors[j][i])
                    angle_errors_sum_tmp += np.array(angle_errors[j][i])
                angle_errors_mean += [angle_errors_sum_tmp / len(self.vehicles)]
            angle_errors_intermediate_all += [angle_errors_mean]
        
        color=cm.rainbow(np.linspace(0,1,self.n_intermediate_ADMM))
        
        plt.figure()
        for angle_errors_mean, c in zip(angle_errors_intermediate_all, color):
            for time, angle in zip(t, angle_errors_mean):
                plt.plot(time, angle, c = c)
        plt.show()

    def plot_frenet_view(self):
        fig, ax = self.figures["figures"]
        
        for i in range(self.stage):
            ax.clear()
            for vehicle in self.vehicles:
                ax = vehicle.visualize_x_problem(ax, i)
            fig.savefig(self.cwd + '/figures/' + str(self.seed) + '/frenet_view_' + '{:0>2d}'.format(i) +'.png', dpi = 200)
    
    def plot_moovie_frames(self, n_frames, iternum : int = 0, seed = ''):
        fig, ax = self.figures["figures"]
        ax.clear()
        frame_num = 0
        horizon_num = 0
        for t in np.linspace(0, 1, n_frames):

            # Then we plot the vehicles
            for i in range(len(self.vehicles)):
                t_start, ax = self.vehicles[i].plot_moovie_frames_mooving_horizon(ax, horizon_num)
            horizon_num += 1

            # Axis related stuff
            # ax.set_title("Trajectories of the vehicles after iteration {} with seed {}".format(iternum, seed))
            # Or setting the ax limits 
            ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
            ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
            ax.set_xlim(self.vehicles[0].fp.fx_spline(t_start)[0][0] - 2, self.vehicles[0].fp.fx_spline(t_start)[0][0] + 2*3)
            ax.set_ylim(self.vehicles[0].fp.fy_spline(t_start)[0][0] - 2.5, self.vehicles[0].fp.fy_spline(t_start)[0][0] + 2.5)

            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            # Saving figure to folder
            self.seed = 0
            fig.savefig(self.cwd + '/video/' + str(self.seed) + '/' + '{:0>2d}'.format(frame_num) +'.png', dpi = 200)
            ax.clear()
            frame_num += 1
            
        return self
    
    
    
    def plot_moovie_frames_old(self, iternum : int = 0, seed = ''):

        
        "Zoomed-in version"
        fig, ax = self.figures["figures"]

        frame_num = 0
        for t in np.linspace(0, 1, 100):
            for i in range(len(self.vehicles)):
                ax = self.vehicles[i].plot_moovie_frames(ax, t)
            ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
            ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
            # Or setting the ax limits 
            ax.set_xlim(self.vehicles[0].fp.fx_spline(t)[0][0] - 2*2, self.vehicles[0].fp.fx_spline(t)[0][0] + 2*2)
            ax.set_ylim(self.vehicles[0].fp.fy_spline(t)[0][0] - 1*2, self.vehicles[0].fp.fy_spline(t)[0][0] + 1*2)
            ax.set_xlabel("x axis")
            ax.set_ylabel("y axis")
            ax.set_aspect('equal', adjustable='box')
            
            plt.axis('off')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            
            
            # Saving figure to folder
            fig.savefig(self.cwd + '/video/' + '{:0>1d}'.format(self.stage) + '{:0>2d}'.format(frame_num) +'.png', dpi = 200)
            ax.clear()
            frame_num += 1
            
        return self

    def save_trajectory_to_csv(self, n_steps, t_desired = 1, t_hover = 0.1):
        for i in range(len(self.vehicles)):
            self.vehicles[i].save_trajectory_to_csv_SINGLE(n_steps, t_desired = t_desired, t_hover = t_hover)
        return self

    def frenet_plotter(self, iternum : int = 0, seed = ''):
        """This function plots the trajectories calculated by each of the agent.
        It also plots the Frenet path.
        """
        
        # Plotting trajectory of the vehicles
        fig, ax = self.figures["figures"] # plt.subplots()
        for i in range(len(self.vehicles)):
            ax = self.vehicles[i].plot_vehicle_frenet_trajectories(ax)
        
        try:
            if self.vehicle_positions_new != []:
                for position in self.vehicle_positions_new:
                    x, y = self.fp.frenet_to_inertial(position[0], position[1], self.vehicles[0].t_end + self.vehicles[0].t_step)
                    ax.plot(x, y, 'ro', markersize = 1)
            else:
                for vehicle in self.vehicles:
                    position = vehicle.xf[0:2]
                    x, y = self.fp.frenet_to_inertial(position[0], position[1], self.vehicles[0].t_end + self.vehicles[0].t_step)
                    ax.plot(x, y, 'go', markersize = 1)
        except:
            pass
            
        # Axis related stuff
        ax.set_title("Trajectories of the vehicles after iteration {} with seed {} in the frenet frame".format(iternum, seed))
        ax.set_xlim(self.border_x[0] * 1.2, self.border_x[1] * 1.2)
        ax.set_ylim(self.border_y[0] * 1.2, self.border_y[1] * 1.2)
        ax.set_xlabel("x axis")  
        ax.set_ylabel("y axis") 
        ax.set_aspect('equal', adjustable='box')
        fig.savefig(self.cwd + '/figures/' + '{:0>2d}'.format(iternum) +'.png', dpi = 200)
        ax.cla()
        return self