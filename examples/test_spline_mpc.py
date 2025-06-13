import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc_controller.mpc_controller import MPCController
from mpc_controller.vehicle_model import VehicleModel
from mpc_controller.spline_path_dynamics import SplinePathDynamics
from spline_fit.curvilinear_coordinates import CurvilinearCoordinates
from spline_fit.spline_fitter import SplineFitter

def create_test_path():
    """Create a test path with waypoints."""
    # Create a simple path with clear progression
    waypoints = np.array([
        [0, 0],
        [5, 2],
        [10, 3],
        [15, 1],
        [20, -2],
        [25, 0],
        [30, 3]
    ])
    
    print("\nDEBUG: Test path created")
    print(f"DEBUG: Number of waypoints: {len(waypoints)}")
    print(f"DEBUG: First 5 waypoints:\n{waypoints[:5]}")
    print(f"DEBUG: Last 5 waypoints:\n{waypoints[-5:]}")
    return waypoints

def get_local_waypoints(global_waypoints, current_position, horizon=15):
    """Select a moving horizon of waypoints from the global path."""
    # Find closest waypoint index
    dists = np.linalg.norm(global_waypoints - current_position, axis=1)
    closest_idx = np.argmin(dists)
    
    print(f"\nDEBUG: Current position: {current_position}")
    print(f"DEBUG: Closest waypoint index: {closest_idx}")
    print(f"DEBUG: Closest waypoint: {global_waypoints[closest_idx]}")
    print(f"DEBUG: Distance to closest: {dists[closest_idx]}")
    
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
    
    print(f"DEBUG: Selected waypoints shape: {local_waypoints.shape}")
    print(f"DEBUG: Selected waypoints:\n{local_waypoints}")
    print(f"DEBUG: Selected s_values:\n{local_s_values}")
    
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
    
    print(f"DEBUG: Final waypoints shape: {local_waypoints.shape}")
    print(f"DEBUG: Final waypoints:\n{local_waypoints}")
    print(f"DEBUG: Final s_values:\n{local_s_values}")
    
    return local_waypoints

def plot_spline_debug(spline_fitter):
    """Plot spline and its derivatives for debugging."""
    # Create fine grid of s values
    s_fine = np.linspace(0, spline_fitter.path_length, 1000)
    
    # Evaluate spline and derivatives
    positions = np.array([spline_fitter.evaluate(s) for s in s_fine])
    tangents = np.array([spline_fitter.evaluate_derivatives(s) for s in s_fine])
    second_derivs = np.array([spline_fitter.evaluate_second_derivatives(s) for s in s_fine])
    curvatures = np.array([spline_fitter.compute_curvature(s) for s in s_fine])
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Spline path
    axs[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', label='Spline')
    # Add tangent vectors
    for i in range(0, len(s_fine), 50):
        pos = positions[i]
        tan = tangents[i]
        tan_norm = tan / np.linalg.norm(tan) * 0.5
        axs[0, 0].arrow(pos[0], pos[1], tan_norm[0], tan_norm[1], 
                       head_width=0.2, head_length=0.3, fc='red', ec='red')
    axs[0, 0].set_title('Spline Path with Tangent Vectors')
    axs[0, 0].set_xlabel('X [m]')
    axs[0, 0].set_ylabel('Y [m]')
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')
    
    # Plot 2: Tangent magnitude
    tangent_mags = np.linalg.norm(tangents, axis=1)
    axs[0, 1].plot(s_fine, tangent_mags, 'r-')
    axs[0, 1].set_title('Tangent Vector Magnitude |dx/ds|')
    axs[0, 1].set_xlabel('s [m]')
    axs[0, 1].set_ylabel('|dx/ds|')
    axs[0, 1].grid(True)
    
    # Plot 3: Second derivative magnitude
    second_deriv_mags = np.linalg.norm(second_derivs, axis=1)
    axs[1, 0].plot(s_fine, second_deriv_mags, 'g-')
    axs[1, 0].set_title('Second Derivative Magnitude |d²x/ds²|')
    axs[1, 0].set_xlabel('s [m]')
    axs[1, 0].set_ylabel('|d²x/ds²|')
    axs[1, 0].grid(True)
    
    # Plot 4: Curvature
    axs[1, 1].plot(s_fine, curvatures, 'purple')
    axs[1, 1].set_title('Curvature κ')
    axs[1, 1].set_xlabel('s [m]')
    axs[1, 1].set_ylabel('κ [1/m]')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def run_simulation():
    # Create test path
    waypoints = create_test_path()
    global_waypoints = waypoints  # Store global waypoints
    
    print("\nDEBUG: Global waypoints shape:", global_waypoints.shape)
    print("DEBUG: First few global waypoints:\n", global_waypoints[:5])
    
    # Create spline path with chord-length parameterization
    spline_fitter = SplineFitter()
    spline = spline_fitter.fit_spline(waypoints)
    
    # Debug plot spline and derivatives
    # plot_spline_debug(spline)
    
    spline_coords = CurvilinearCoordinates(waypoints)
    
    print("\nDEBUG: Path length:", spline.path_length)
    print("DEBUG: First few waypoints:\n", waypoints[:5])
    
    # Create vehicle model
    vehicle_model = VehicleModel(
        wheelbase_front=1.0,
        wheelbase_rear=1.0,
        max_steering=np.pi/4,
        min_steering=-np.pi/4,
        max_acceleration=2.0,
        min_acceleration=-2.0,
        max_velocity=20.0,
        min_velocity=0.0
    )
    
    # Create spline path dynamics
    spline_dynamics = SplinePathDynamics(vehicle_model, spline_coords)
    
    # Create MPC controller with more conservative settings
    mpc = MPCController(
        vehicle_model=vehicle_model,
        prediction_horizon=2.0,
        dt=0.1
    )
    
    # Set more conservative cost weights
    Q = np.diag([
        0.1,    # s (progress) - very small weight initially
        0.0,    # u (spline parameter) - not directly penalized
        10.0,   # e_y (cross-track error) - moderate weight
        1.0,    # e_ψ (heading error) - small weight
        0.1     # v (velocity) - very small weight
    ])
    
    R = np.diag([
        0.1,    # steering - very small weight initially
        0.01    # acceleration - very small weight initially
    ])
    
    Q_terminal = np.diag([
        1.0,    # s (progress)
        0.0,    # u (spline parameter)
        20.0,   # e_y (cross-track error)
        2.0,    # e_ψ (heading error)
        0.2     # v (velocity)
    ])
    
    mpc.set_weights(Q, R, Q_terminal)
    mpc.set_path(spline_dynamics)
    
    # Initial state [s, u, e_y, e_ψ, v]
    initial_state = np.array([
        0.1,    # s: start slightly ahead
        0.1,    # u: start slightly ahead
        0.0,    # e_y: on path
        0.0,    # e_ψ: aligned with path
        0.5     # v: very slow initial velocity
    ])
    
    print("\nDEBUG: Initial state:", initial_state)
    
    # Get initial position by evaluating spline at u=0.1 using numerical values
    initial_pos = spline_dynamics.spline_coords.curvilinear_to_cartesian(initial_state[1], 0.0)
    print("DEBUG: Initial position:", initial_pos)
    
    # Print initial state
    print("\n=== DEBUG: Initial State ===")
    print(f"Initial state: {initial_state}")
    print(f"Initial position: {spline_dynamics.spline_curvilinear_to_cartesian(initial_state)[:2]}")

    # Print constraints
    constraints = mpc.spline_dynamics.get_constraints()
    print("\n=== DEBUG: Constraints ===")
    for k, v in constraints.items():
        print(f"{k}: {v}")

    # Print cost weights
    print("\n=== DEBUG: Cost Weights ===")
    print(f"Q: {mpc.Q}")
    print(f"R: {mpc.R}")
    
    # Debug: Discrete dynamics from initial state with zero input
    print("\n=== DEBUG: Discrete dynamics from initial state with zero input ===")
    zero_input = np.zeros(2)
    params = np.concatenate([
        spline_dynamics.knots_np,
        spline_dynamics.coeffs_x_np.flatten(),
        spline_dynamics.coeffs_y_np.flatten()
    ])
    discrete_dyn = spline_dynamics.get_discrete_dynamics(mpc.dt)
    next_state = discrete_dyn(initial_state, zero_input, params)
    print("Next state:", np.array(next_state).flatten())

    # Debug: Trivial reference trajectory (all steps = initial state)
    trivial_reference_trajectory = {
        's_values': np.ones(mpc.N + 1) * initial_state[0],
        'velocities': np.ones(mpc.N + 1) * initial_state[4]
    }
    result = mpc.solve(initial_state, trivial_reference_trajectory)
    print("\n=== DEBUG: Trivial reference solve result ===")
    print("Solver status:", result['solver_status'])
    print("Optimal input:", result['optimal_input'])
    print("Predicted trajectory:", result['predicted_trajectory'])
    import sys; sys.exit(0)
    
    # Simulation parameters
    sim_time = 30.0  # seconds
    dt = mpc.dt
    n_steps = int(sim_time / dt)
    
    # Initialize arrays to store results
    states = np.zeros((n_steps + 1, 5))
    inputs = np.zeros((n_steps, 2))
    states[0] = initial_state
    
    # Set initial guess for the solver
    for i in range(mpc.N + 1):
        mpc.solver.set(i, "x", initial_state)
    for i in range(mpc.N):
        mpc.solver.set(i, "u", np.zeros(2))
    
    # Main simulation loop
    for step in range(n_steps):
        print(f"\nStep {step + 1}/{n_steps}")
        
        # Get current position
        current_pos = spline_dynamics.spline_curvilinear_to_cartesian(states[step])[:2]
        
        # Get local waypoints
        local_waypoints = get_local_waypoints(global_waypoints, current_pos)
        
        # Update spline parameters
        mpc.update_waypoints(local_waypoints)
        
        # Create reference trajectory
        s_values = np.linspace(states[step][0], states[step][0] + 4.0, mpc.N + 1)
        velocities = np.ones(mpc.N + 1) * 0.5  # Constant velocity reference
        
        reference_trajectory = {
            's_values': s_values,
            'velocities': velocities
        }
        
        # Solve MPC problem
        result = mpc.solve(states[step], reference_trajectory)
        
        if result['solver_status'] != 0:
            print(f"Warning: MPC solver failed at step {step}")
            print(f"Current state: {states[step]}")
            print(f"Reference s_values: {s_values}")
            print(f"Current position: {current_pos}")
            break
        
        # Apply first control input
        inputs[step] = result['optimal_input']
        
        # Simulate one step
        states[step + 1] = spline_dynamics.simulate_step(states[step], inputs[step], dt)
        
        # Update initial guess for next iteration
        for i in range(mpc.N):
            if i < mpc.N - 1:
                mpc.solver.set(i, "x", result['predicted_trajectory'][i + 1])
                mpc.solver.set(i, "u", result['optimal_sequence'][i + 1])
            else:
                # For the last step, use the terminal state
                mpc.solver.set(i, "x", result['predicted_trajectory'][-1])
                mpc.solver.set(i, "u", result['optimal_sequence'][-1])
        
        # Set terminal state guess
        mpc.solver.set(mpc.N, "x", result['predicted_trajectory'][-1])
    
    # Plot results
    plot_results(states, inputs, dt, spline_dynamics)
    
    return {
        'states': states,
        'inputs': inputs
    }

def plot_results(states, inputs, dt, spline_dynamics):
    """Plot simulation results."""
    # Create time vector
    t = np.arange(len(states)) * dt
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    
    # Plot path and vehicle trajectory
    axs[0, 0].set_title('Path and Vehicle Trajectory')
    # Plot reference path
    s_values = np.linspace(0, spline_dynamics.spline_coords.path_length, 1000)
    path_points = np.array([spline_dynamics.spline_coords.curvilinear_to_cartesian(s, 0) for s in s_values])
    axs[0, 0].plot(path_points[:, 0], path_points[:, 1], 'b-', label='Reference Path')
    
    # Plot vehicle trajectory
    vehicle_points = np.array([spline_dynamics.spline_curvilinear_to_cartesian(state)[:2] for state in states])
    axs[0, 0].plot(vehicle_points[:, 0], vehicle_points[:, 1], 'r-', label='Vehicle')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')
    
    # Plot states
    axs[0, 1].set_title('States')
    axs[0, 1].plot(t, states[:, 0], label='s (progress)')
    axs[0, 1].plot(t, states[:, 2], label='e_y (lateral error)')
    axs[0, 1].plot(t, states[:, 3], label='e_ψ (heading error)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot inputs
    axs[1, 0].set_title('Steering Angle')
    axs[1, 0].plot(t[:-1], np.rad2deg(inputs[:, 0]), 'b-')
    axs[1, 0].set_ylabel('Steering [deg]')
    axs[1, 0].grid(True)
    
    axs[1, 1].set_title('Acceleration')
    axs[1, 1].plot(t[:-1], inputs[:, 1], 'r-')
    axs[1, 1].set_ylabel('Acceleration [m/s²]')
    axs[1, 1].grid(True)
    
    # Plot errors
    axs[2, 0].set_title('Lateral Error')
    axs[2, 0].plot(t, states[:, 2], 'b-')
    axs[2, 0].set_ylabel('Lateral Error [m]')
    axs[2, 0].grid(True)
    
    axs[2, 1].set_title('Heading Error')
    axs[2, 1].plot(t, states[:, 3], 'r-')
    axs[2, 1].set_ylabel('Heading Error [rad]')
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_results_advanced(states, inputs, spline_coords, dt):
    """Plot simulation results using advanced visualization from basic_example.py."""
    # Create time vector
    t = np.arange(len(states)) * dt

    plt.figure(figsize=(15, 10))

    # Plot 1: Reference path and waypoints
    plt.subplot(2, 2, 1)
    # Generate trajectory for plotting
    trajectory = spline_coords.generate_trajectory(num_points=200)
    plt.plot(trajectory['positions'][:, 0], trajectory['positions'][:, 1], 'b-', linewidth=2, label='Spline')
    # Plot actual vehicle trajectory
    vehicle_points = np.array([spline_coords.curvilinear_to_cartesian(s, d) for s, d in zip(states[:, 0], states[:, 2])])
    plt.plot(vehicle_points[:, 0], vehicle_points[:, 1], 'r--', linewidth=2, label='Vehicle Trajectory', alpha=0.8)
    plt.plot(vehicle_points[0, 0], vehicle_points[0, 1], 'go', markersize=10, label='Start')
    plt.plot(vehicle_points[-1, 0], vehicle_points[-1, 1], 'rs', markersize=10, label='End')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Reference Path with Vehicle Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Plot 2: Curvature profile
    plt.subplot(2, 2, 2)
    s_fine = np.linspace(0, spline_coords.path_length, 200)
    curvatures = [spline_coords.get_curvature(s) for s in s_fine]
    plt.plot(s_fine, curvatures, 'g-', linewidth=2)
    plt.xlabel('Arc Length s [m]')
    plt.ylabel('Curvature κ [1/m]')
    plt.title('Path Curvature Profile')
    plt.grid(True, alpha=0.3)

    # Plot 3: State evolution
    plt.subplot(2, 2, 3)
    plt.plot(t, states[:, 0], 'g-', label='s (progress)')
    plt.plot(t, states[:, 2], 'b-', label='e_y (lateral error)')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Vehicle Position')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Control inputs
    plt.subplot(2, 2, 4)
    plt.plot(t[:-1], np.rad2deg(inputs[:, 0]), 'b-', label='Steering δ')
    plt.plot(t[:-1], inputs[:, 1], 'r-', label='Acceleration a')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input')
    plt.title('Control Inputs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation() 