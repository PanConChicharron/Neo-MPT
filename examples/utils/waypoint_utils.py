import numpy as np

def get_local_waypoints(global_waypoints, current_position, horizon=15):
    """
    Select a moving horizon of waypoints from the global path.
    
    Args:
        global_waypoints: Array of global waypoints [N, 2]
        current_position: Current vehicle position [x, y]
        horizon: Horizon parameter (not currently used)
        
    Returns:
        Array of local waypoints around the current position
    """
    # Find closest waypoint index
    dists = np.linalg.norm(global_waypoints - current_position, axis=1)
    closest_idx = np.argmin(dists)
    
    # Calculate chord lengths
    dx = np.diff(global_waypoints[:, 0])
    dy = np.diff(global_waypoints[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    s_values = np.concatenate(([0], np.cumsum(ds)))
    
    # Always use 7 waypoints (6 segments) to match initial setup
    n_waypoints = 7
    
    # Calculate how many points to take before and after
    n_before = min(n_waypoints // 2, closest_idx)
    n_after = min(n_waypoints - n_before - 1, len(global_waypoints) - closest_idx - 1)
    
    # Get waypoints around the closest point
    start_idx = closest_idx - n_before
    end_idx = closest_idx + n_after + 1
    local_waypoints = global_waypoints[start_idx:end_idx]
    local_s_values = s_values[start_idx:end_idx]
    
    # If we need more points, linearly extend the path
    if len(local_waypoints) < n_waypoints:
        # Get direction of last segment
        if len(local_waypoints) > 1:
            direction = local_waypoints[-1] - local_waypoints[-2]
            direction = direction / np.linalg.norm(direction)
        else:
            # If we only have one point, use a default direction
            direction = np.array([1.0, 0.0])
        
        # Create linearly spaced points
        n_extra = n_waypoints - len(local_waypoints)
        last_point = local_waypoints[-1]
        last_s = local_s_values[-1]
        
        # Generate points with increasing x-coordinates
        x_values = np.linspace(last_point[0] + 0.1, last_point[0] + n_extra + 0.1, n_extra)
        y_values = last_point[1] + direction[1] * (x_values - last_point[0])
        s_values_extra = np.linspace(last_s + 0.1, last_s + n_extra + 0.1, n_extra)
        
        extra_points = np.column_stack((x_values, y_values))
        local_waypoints = np.vstack([local_waypoints, extra_points])
        local_s_values = np.concatenate([local_s_values, s_values_extra])
    
    return local_waypoints 