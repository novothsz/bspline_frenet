import numpy as np



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