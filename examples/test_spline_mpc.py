import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

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
    return waypoints

def create_straight_line_path():
    """Create a simple straight-line test path for debugging boundary behavior."""
    # Create a straight line path - this should have ds/du = 1.0 everywhere
    waypoints = np.array([
        [0, 0],
        [5, 0],
        [10, 0],
        [15, 0],
        [20, 0],
        [25, 0],
        [30, 0]
    ])
    return waypoints

def create_racetrack_path():
    #faulty
    """Create a race-track style pseudo-ellipse path with straight sections and curved ends."""
    # Race-track style pseudo-ellipse waypoints
    # Creates an oval with straight sections and curved ends
    waypoints = np.array([
        # Bottom straight section (start)
        [0, 0],
        [5, 0],
        [10, 0],
        [15, 0],
        [20, 0],
        
        # Right curved section (turn 1)
        [25, 2],
        [28, 5],
        [30, 8],
        [30, 12],
        
        # Top straight section
        [28, 15],
        [25, 16],
        [20, 16],
        [15, 16],
        [10, 16],
        [5, 16],
        [0, 16],
        
        # Left curved section (turn 2)
        [-3, 14],
        [-5, 10],
        [-5, 6],
        [-3, 2],
        
        # Return to start
        [0, 0]
    ])
    return waypoints

def create_challenging_track():
    """Create a longer, more challenging race track with multiple complex turns."""
    # Complex race track with chicanes, hairpins, and varying radius turns
    waypoints = np.array([
        # Start/finish straight
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
        
        # Turn 1: Fast right-hander
        [40, -3],
        [50, -8],
        [60, -10],
        [70, -8],
        
        # Short straight into chicane
        [80, -5],
        [90, -2],
        
        # Chicane section (S-curves)
        [95, 2],
        [100, 6],
        [105, 4],
        [110, 0],
        [115, -4],
        [120, -2],
        
        # Long straight
        [130, 0],
        [140, 2],
        [150, 4],
        [160, 6],
        
        # Hairpin turn (180-degree)
        [165, 10],
        [168, 15],
        [170, 20],
        [168, 25],
        [165, 30],
        [160, 32],
        [150, 30],
        [140, 28],
        [130, 26],
        
        # Back straight with slight curves
        [120, 24],
        [110, 22],
        [100, 20],
        [90, 18],
        [80, 16],
        
        # Complex turn sequence
        [70, 12],
        [60, 8],
        [50, 6],
        [40, 8],
        [30, 12],
        [20, 14],
        [10, 12],
        [5, 8],
        [2, 4],
        
        # # Return to start
        # [0, 0]
    ])
    
    return waypoints

def get_local_waypoints(global_waypoints, current_position, horizon=15):
    """Select a moving horizon of waypoints from the global path."""
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

def run_simulation(path_type="curved"):
    # Create test path based on type
    if path_type == "straight":
        waypoints = create_straight_line_path()
        print("\n=== RUNNING STRAIGHT-LINE TEST ===")
    elif path_type == "curved":
        waypoints = create_test_path()
        print("\n=== RUNNING CURVED PATH TEST ===")
    elif path_type == "racetrack":
        waypoints = create_racetrack_path()
        print("\n=== RUNNING RACE TRACK TEST ===")
    elif path_type == "challenging":
        waypoints = create_challenging_track()
        print("\n=== RUNNING CHALLENGING TRACK TEST ===")
    else:
        raise ValueError("Invalid path type. Use 'straight', 'curved', 'racetrack', 'challenging', or 'both'.")
        
    global_waypoints = waypoints  # Store global waypoints
    
    # Create spline path with chord-length parameterization
    spline_fitter = SplineFitter()
    spline = spline_fitter.fit_spline(waypoints)
    
    # Debug plot spline and derivatives
    # plot_spline_debug(spline)
    
    spline_coords = CurvilinearCoordinates(waypoints)
    
    # Create vehicle model
    vehicle_model = VehicleModel(
        wheelbase_front=1.0,
        wheelbase_rear=1.0,
        max_steering=np.pi/3,
        min_steering=-np.pi/3,
        max_acceleration=2.0,
        min_acceleration=-2.0,  # Increased braking capability for end-of-path approach
        max_velocity=20.0,
        min_velocity=-20.0        # Allow very slow speeds to prevent infeasibility
    )
    
    # Create spline path dynamics
    spline_dynamics = SplinePathDynamics(vehicle_model, spline_coords)
    
    # Create MPC controller with more conservative settings
    mpc = MPCController(
        vehicle_model=vehicle_model,
        prediction_horizon=2.0,
        dt=0.1
    )
    
    # Set cost weights (must be done before set_path)
    state_weights = np.array([1.0, 0.0, 5.0, 0.5, 1.0])  # [s, u, e_y, e_ψ, v] - velocity weight = 0.0
    input_weights = np.array([0.1, 0.125])  # [delta, a] - very low acceleration penalty
    terminal_weights = np.array([2.0, 0.0, 10.0, 1.0, 1.0])  # [s, u, e_y, e_ψ, v] - terminal velocity weight = 0.0
    
    mpc.set_weights(state_weights, input_weights, terminal_weights)
    mpc.set_path(spline_dynamics)
    
    # Initial state [s, u, e_y, e_ψ, v]
    # Start near the beginning of the path with proper s/u relationship
    initial_u = 0.0  # chord-length parameter
    initial_s = spline_coords.chord_to_arc_length(initial_u)  # convert to arc-length
    
    initial_state = np.array([
        initial_s,  # s: arc-length progress
        initial_u,  # u: chord-length parameter
        0.0,        # e_y: on path
        0.0,        # e_ψ: aligned with path
        0.0         # v: moderate initial velocity to 2 m/s
    ])
    
    # Get initial position by evaluating spline at u=0.1 using numerical values
    initial_pos = spline_dynamics.spline_coords.curvilinear_to_cartesian(initial_state[1], 0.0)
    
    # Print initial state
    print("\n=== DEBUG: Initial State ===")
    print(f"Initial state: {initial_state}")
    print(f"Initial position: {spline_dynamics.spline_curvilinear_to_cartesian(initial_state)[:2]}")

    # Print constraints
    constraints = mpc.spline_dynamics.get_constraints()
    print("\n=== DEBUG: Constraints ===")
    for k, v in constraints.items():
        print(f"{k}: {v}")
    
    # Simulation parameters
    sim_time = 250.0  # seconds
    dt = mpc.dt
    n_steps = int(sim_time / dt)
    
    # Initialize arrays to store results
    states = np.zeros((n_steps + 1, 5))
    inputs = np.zeros((n_steps, 2))
    solve_times = np.zeros(n_steps)  # Track MPC solve times
    states[0] = initial_state
    
    # Set initial guess for the solver
    for i in range(mpc.N + 1):
        mpc.solver.set(i, "x", initial_state)
    for i in range(mpc.N):
        mpc.solver.set(i, "u", np.zeros(2))
    
    # Main simulation loop
    for step in range(n_steps):
        print(f"\nStep {step + 1}/{n_steps}")
        
        # Check if we've reached the end of the path
        current_s = states[step][0]
        path_length = spline_dynamics.spline_coords.path_length
        remaining_path = path_length - current_s
        
        print(f"  Current s: {current_s:.2f} m, Path length: {path_length:.2f} m, Remaining: {remaining_path:.2f} m")
        
        # Stop simulation if we've reached the end of the path
        if remaining_path <= 0.1:  # Within 10cm of the end
            print(f"🏁 Reached end of path! Stopping simulation at step {step}")
            print(f"   Final position: s={current_s:.2f} m (path length: {path_length:.2f} m)")
            # Trim arrays to actual simulation length
            states = states[:step+1]
            inputs = inputs[:step]
            break
        
        # Get current position
        current_pos = spline_dynamics.spline_curvilinear_to_cartesian(states[step])[:2]
        
        # Check if coordinate transformation is failing (position stuck or invalid)
        if step > 0:
            prev_pos = spline_dynamics.spline_curvilinear_to_cartesian(states[step-1])[:2]
            pos_change = np.linalg.norm(current_pos - prev_pos)
            if pos_change < 1e-6 and states[step][4] > 0.1:  # Position not changing but velocity > 0
                print(f"⚠️  WARNING: Vehicle position stuck at {current_pos}, but velocity = {states[step][4]:.2f} m/s")
                print(f"   This indicates coordinate transformation failure. Stopping simulation.")
                states = states[:step+1]
                inputs = inputs[:step]
                break
        
        # Update spline parameters only for non-racetrack paths
        # For racetrack and challenging track, we use the full closed-loop path throughout
        if path_type not in ["racetrack", "challenging"]:
            # Get local waypoints
            local_waypoints = get_local_waypoints(global_waypoints, current_pos)
            
            # Update spline parameters
            mpc.update_waypoints(local_waypoints)
        
        # Create reference trajectory that respects path bounds
        current_s = states[step][0]
        path_length = spline_dynamics.spline_coords.path_length
        
        # Calculate how much path is remaining
        remaining_path = path_length - current_s
        
        # Set reference horizon based on remaining path
        if remaining_path > 4.0:
            # Normal case: look ahead 4.0 meters
            s_end = current_s + 4.0
        else:
            # Near end of path: just go to the end and maintain that position
            s_end = path_length
        
        s_values = np.linspace(current_s, s_end, mpc.N + 1)
        
        # Improved velocity planning for end-of-path approach
        current_velocity = states[step][4]
        
        if remaining_path < 5.0:
            # More aggressive braking when approaching the end
            # Calculate required deceleration to stop at the end
            # v² = v₀² + 2as  =>  a = (v² - v₀²) / (2s)
            # To reach 0.5 m/s at the end: a = (0.5² - v₀²) / (2 * remaining_path)
            target_final_velocity = 5  # m/s
            if remaining_path > 0.1:  # Avoid division by zero
                required_decel = (target_final_velocity**2 - current_velocity**2) / (2 * remaining_path)
                # Limit to maximum braking capability
                required_decel = max(required_decel, -2.0)  # Don't exceed -2.0 m/s² braking
                
                # Set target velocity based on physics-based deceleration
                # v = sqrt(v₀² + 2as) where s is the distance traveled in prediction horizon
                prediction_distance = min(2.0, remaining_path)  # Look ahead 2m or to end
                target_velocity = max(0.5, np.sqrt(max(0, current_velocity**2 + 2 * required_decel * prediction_distance)))
                
                print(f"  🛑 End approach: remaining={remaining_path:.2f}m, current_v={current_velocity:.2f}m/s")
                print(f"     Required decel={required_decel:.2f}m/s², target_v={target_velocity:.2f}m/s")
            else:
                target_velocity = 0.5
        elif remaining_path < 10.0:
            # Moderate slowdown when getting close
            target_velocity = 3.0
        else:
            # Normal velocity
            target_velocity = 5.0
            
        velocities = np.ones(mpc.N + 1) * target_velocity
        
        reference_trajectory = {
            's_values': s_values,
            'velocities': velocities
        }
        
        # Debug: Print reference vs current velocity every 10 steps
        if step % 10 == 0:
            print(f"  Reference velocity: {target_velocity:.2f} m/s, Current velocity: {states[step][4]:.2f} m/s")
            print(f"  Velocity error: {target_velocity - states[step][4]:.2f} m/s")
        
        # Solve MPC problem with timing
        solve_start_time = time.perf_counter()
        result = mpc.solve(states[step], reference_trajectory)
        solve_end_time = time.perf_counter()
        solve_times[step] = solve_end_time - solve_start_time
        
        if result['solver_status'] != 0:
            print(f"Warning: MPC solver failed at step {step}")
            print(f"Current state: {states[step]}")
            print(f"Reference s_values: {s_values}")
            print(f"Current position: {current_pos}")
            break
        
        # Apply first control input
        inputs[step] = result['optimal_input']
        
        # Debug: Print control inputs every 10 steps
        if step % 10 == 0 or step > n_steps - 20:  # Print first few and last 20 steps
            print(f"  Control input: steering={inputs[step][0]:.4f} rad ({np.rad2deg(inputs[step][0]):.2f}°), acceleration={inputs[step][1]:.4f} m/s²")
            print(f"  Current velocity: {states[step][4]:.2f} m/s, Current s: {states[step][0]:.2f} m")
        
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
    
    # Print timing statistics
    actual_steps = len([t for t in solve_times if t > 0])  # Count non-zero solve times
    if actual_steps > 0:
        valid_solve_times = solve_times[:actual_steps]
        avg_solve_time = np.mean(valid_solve_times)
        avg_frequency = 1.0 / avg_solve_time if avg_solve_time > 0 else 0
        median_frequency = 1.0 / np.median(valid_solve_times) if np.median(valid_solve_times) > 0 else 0
        max_frequency = 1.0 / np.min(valid_solve_times) if np.min(valid_solve_times) > 0 else 0
        min_frequency = 1.0 / np.max(valid_solve_times) if np.max(valid_solve_times) > 0 else 0
        
        print(f"\n=== MPC SOLVER TIMING STATISTICS ===")
        print(f"Total simulation steps: {actual_steps}")
        print(f"Average solve time: {avg_solve_time*1000:.2f} ms")
        print(f"Median solve time: {np.median(valid_solve_times)*1000:.2f} ms")
        print(f"Min solve time: {np.min(valid_solve_times)*1000:.2f} ms")
        print(f"Max solve time: {np.max(valid_solve_times)*1000:.2f} ms")
        print(f"Std dev solve time: {np.std(valid_solve_times)*1000:.2f} ms")
        print(f"")
        print(f"=== MPC SOLVER FREQUENCY STATISTICS ===")
        print(f"Average frequency: {avg_frequency:.2f} Hz")
        print(f"Median frequency: {median_frequency:.2f} Hz")
        print(f"Max frequency: {max_frequency:.2f} Hz")
        print(f"Min frequency: {min_frequency:.2f} Hz")
        print(f"Required frequency (1/dt): {1.0/dt:.2f} Hz")
        print(f"")
        print(f"=== REAL-TIME PERFORMANCE ===")
        print(f"Total solve time: {np.sum(valid_solve_times)*1000:.2f} ms")
        print(f"Total simulation time: {actual_steps * dt:.2f} s")
        print(f"Real-time factor: {(actual_steps * dt) / np.sum(valid_solve_times):.2f}x")
        if avg_frequency >= 1.0/dt:
            print(f"✅ MPC solver frequency ({avg_frequency:.1f} Hz) meets real-time requirement ({1.0/dt:.1f} Hz)")
        else:
            print(f"⚠️  MPC solver frequency ({avg_frequency:.1f} Hz) below real-time requirement ({1.0/dt:.1f} Hz)")
        if np.sum(valid_solve_times) > actual_steps * dt:
            print(f"⚠️  MPC solver is slower than real-time!")
        else:
            print(f"✅ MPC solver is faster than real-time")
    
    # Plot results with timing information
    plot_results_with_timing(states, inputs, solve_times, dt, spline_dynamics)
    
    return {
        'states': states,
        'inputs': inputs,
        'solve_times': solve_times
    }

def plot_results_with_timing(states, inputs, solve_times, dt, spline_dynamics):
    """Plot simulation results including MPC timing information."""
    # Create time vector
    t = np.arange(len(states)) * dt
    
    # Create figure with subplots - changed to 4x3 to add frequency plots
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    
    # Plot path and vehicle trajectory
    axs[0, 0].set_title('Path and Vehicle Trajectory')
    # Plot reference path
    s_values = np.linspace(0, spline_dynamics.spline_coords.path_length, 1000)
    path_points = np.array([spline_dynamics.spline_coords.curvilinear_to_cartesian(s, 0) for s in s_values])
    axs[0, 0].plot(path_points[:, 0], path_points[:, 1], 'b-', label='Reference Path')
    
    # Plot vehicle trajectory
    vehicle_points = np.array([spline_dynamics.spline_curvilinear_to_cartesian(state)[:2] for state in states])
    axs[0, 0].plot(vehicle_points[:, 0], vehicle_points[:, 1], 'r-', label='Vehicle')
    
    # Mark initial and final poses
    initial_state = states[0]
    initial_pos = vehicle_points[0]
    initial_heading = spline_dynamics.spline_coords.get_heading(initial_state[0])
    arrow_length = 2.0
    initial_dx = arrow_length * np.cos(initial_heading)
    initial_dy = arrow_length * np.sin(initial_heading)
    axs[0, 0].arrow(initial_pos[0], initial_pos[1], initial_dx, initial_dy, 
                   head_width=0.5, head_length=0.8, fc='green', ec='green', linewidth=2)
    axs[0, 0].plot(initial_pos[0], initial_pos[1], 'go', markersize=8, label='Start')
    
    final_state = states[-1]
    final_pos = vehicle_points[-1]
    final_heading = spline_dynamics.spline_coords.get_heading(final_state[0])
    final_dx = arrow_length * np.cos(final_heading)
    final_dy = arrow_length * np.sin(final_heading)
    axs[0, 0].arrow(final_pos[0], final_pos[1], final_dx, final_dy, 
                   head_width=0.5, head_length=0.8, fc='red', ec='red', linewidth=2)
    axs[0, 0].plot(final_pos[0], final_pos[1], 'rs', markersize=8, label='End')
    
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
    
    # Plot velocity
    axs[0, 2].set_title('Velocity Tracking')
    axs[0, 2].plot(t, states[:, 4], 'b-', linewidth=2, label='Actual Velocity')
    axs[0, 2].set_ylabel('Velocity [m/s]')
    axs[0, 2].set_xlabel('Time [s]')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # Plot inputs
    axs[1, 0].set_title('Control Inputs')
    ax_steer = axs[1, 0]
    ax_accel = ax_steer.twinx()
    
    line1 = ax_steer.plot(t[:-1], np.rad2deg(inputs[:, 0]), 'b-', label='Steering Angle')
    line2 = ax_accel.plot(t[:-1], inputs[:, 1], 'r-', label='Acceleration')
    
    ax_steer.set_ylabel('Steering [deg]', color='b')
    ax_accel.set_ylabel('Acceleration [m/s²]', color='r')
    ax_steer.set_xlabel('Time [s]')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_steer.legend(lines, labels, loc='upper right')
    ax_steer.grid(True)
    
    # Plot MPC solve times
    axs[1, 1].set_title('MPC Solve Times')
    actual_steps = len([t for t in solve_times if t > 0])
    if actual_steps > 0:
        valid_solve_times = solve_times[:actual_steps]
        axs[1, 1].plot(t[:actual_steps], valid_solve_times * 1000, 'g-', linewidth=2, label='Solve Time')
        axs[1, 1].axhline(y=dt * 1000, color='r', linestyle='--', label=f'Real-time limit ({dt*1000:.0f} ms)')
        axs[1, 1].set_ylabel('Solve Time [ms]')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    
    # Plot MPC solve frequencies
    axs[1, 2].set_title('MPC Solve Frequency')
    if actual_steps > 0:
        valid_solve_times = solve_times[:actual_steps]
        frequencies = 1.0 / valid_solve_times  # Convert solve times to frequencies
        axs[1, 2].plot(t[:actual_steps], frequencies, 'purple', linewidth=2, label='Solve Frequency')
        axs[1, 2].axhline(y=1.0/dt, color='r', linestyle='--', label=f'Required freq ({1.0/dt:.1f} Hz)')
        axs[1, 2].set_ylabel('Frequency [Hz]')
        axs[1, 2].set_xlabel('Time [s]')
        axs[1, 2].legend()
        axs[1, 2].grid(True)
    
    # Plot MPC solve time histogram
    axs[2, 0].set_title('Solve Time Distribution')
    if actual_steps > 0:
        valid_solve_times = solve_times[:actual_steps]
        axs[2, 0].hist(valid_solve_times * 1000, bins=20, alpha=0.7, color='green', edgecolor='black')
        axs[2, 0].axvline(x=np.mean(valid_solve_times) * 1000, color='red', linestyle='--', 
                         label=f'Mean: {np.mean(valid_solve_times)*1000:.2f} ms')
        axs[2, 0].axvline(x=dt * 1000, color='orange', linestyle='--', 
                         label=f'Real-time: {dt*1000:.0f} ms')
        axs[2, 0].set_xlabel('Solve Time [ms]')
        axs[2, 0].set_ylabel('Frequency')
        axs[2, 0].legend()
        axs[2, 0].grid(True, alpha=0.3)
    
    # Plot MPC frequency histogram
    axs[2, 1].set_title('Frequency Distribution')
    if actual_steps > 0:
        valid_solve_times = solve_times[:actual_steps]
        frequencies = 1.0 / valid_solve_times
        axs[2, 1].hist(frequencies, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axs[2, 1].axvline(x=np.mean(frequencies), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(frequencies):.1f} Hz')
        axs[2, 1].axvline(x=1.0/dt, color='orange', linestyle='--', 
                         label=f'Required: {1.0/dt:.1f} Hz')
        axs[2, 1].set_xlabel('Frequency [Hz]')
        axs[2, 1].set_ylabel('Count')
        axs[2, 1].legend()
        axs[2, 1].grid(True, alpha=0.3)
    
    # Plot real-time performance metric
    axs[2, 2].set_title('Real-time Performance')
    if actual_steps > 0:
        valid_solve_times = solve_times[:actual_steps]
        rt_performance = dt / valid_solve_times  # Real-time factor for each step
        axs[2, 2].plot(t[:actual_steps], rt_performance, 'orange', linewidth=2, label='RT Factor')
        axs[2, 2].axhline(y=1.0, color='red', linestyle='--', label='Real-time threshold')
        axs[2, 2].set_ylabel('Real-time Factor')
        axs[2, 2].set_xlabel('Time [s]')
        axs[2, 2].legend()
        axs[2, 2].grid(True)
        axs[2, 2].set_yscale('log')  # Log scale for better visualization
    
    # Plot errors
    axs[3, 0].set_title('Lateral Error')
    axs[3, 0].plot(t, states[:, 2], 'b-')
    axs[3, 0].set_ylabel('Lateral Error [m]')
    axs[3, 0].set_xlabel('Time [s]')
    axs[3, 0].grid(True)
    
    axs[3, 1].set_title('Heading Error')
    axs[3, 1].plot(t, states[:, 3], 'r-')
    axs[3, 1].set_ylabel('Heading Error [rad]')
    axs[3, 1].set_xlabel('Time [s]')
    axs[3, 1].grid(True)
    
    # Plot frequency vs time with moving average
    axs[3, 2].set_title('Frequency Analysis')
    if actual_steps > 0:
        valid_solve_times = solve_times[:actual_steps]
        frequencies = 1.0 / valid_solve_times
        
        # Plot instantaneous frequency
        axs[3, 2].plot(t[:actual_steps], frequencies, 'purple', alpha=0.6, linewidth=1, label='Instantaneous')
        
        # Plot moving average frequency (window of 10 samples)
        if len(frequencies) >= 10:
            window_size = min(10, len(frequencies))
            moving_avg_freq = np.convolve(frequencies, np.ones(window_size)/window_size, mode='valid')
            t_moving = t[window_size-1:actual_steps]
            axs[3, 2].plot(t_moving, moving_avg_freq, 'darkviolet', linewidth=2, label=f'Moving avg ({window_size})')
        
        axs[3, 2].axhline(y=1.0/dt, color='red', linestyle='--', label=f'Required ({1.0/dt:.1f} Hz)')
        axs[3, 2].axhline(y=np.mean(frequencies), color='green', linestyle=':', label=f'Mean ({np.mean(frequencies):.1f} Hz)')
        axs[3, 2].set_ylabel('Frequency [Hz]')
        axs[3, 2].set_xlabel('Time [s]')
        axs[3, 2].legend()
        axs[3, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_results(states, inputs, dt, spline_dynamics):
    """Plot simulation results - wrapper for backward compatibility."""
    # Create dummy solve times array for backward compatibility
    solve_times = np.zeros(len(inputs))
    plot_results_with_timing(states, inputs, solve_times, dt, spline_dynamics)

def run_both_tests():
    """Run both curved and straight-line tests for comparison."""
    print("="*60)
    print("RUNNING BOTH TEST CASES")
    print("="*60)
    
    # Run straight-line test first (simpler case)
    print("\n" + "="*40)
    print("1. STRAIGHT-LINE TEST")
    print("="*40)
    try:
        result_straight = run_simulation("straight")
        print("✅ Straight-line test completed successfully")
    except Exception as e:
        print(f"❌ Straight-line test failed: {e}")
        result_straight = None
    
    # Run curved path test
    print("\n" + "="*40)
    print("2. CURVED PATH TEST")
    print("="*40)
    try:
        result_curved = run_simulation("curved")
        print("✅ Curved path test completed successfully")
    except Exception as e:
        print(f"❌ Curved path test failed: {e}")
        result_curved = None
    
    return {
        'straight': result_straight,
        'curved': result_curved
    }

if __name__ == "__main__":
    import sys
    
    # Check command line arguments for test selection
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "straight":
            print("Running straight-line test only...")
            run_simulation("straight")
        elif test_type == "curved":
            print("Running curved path test only...")
            run_simulation("curved")
        elif test_type == "racetrack":
            print("Running race track test only...")
            run_simulation("racetrack")
        elif test_type == "challenging":
            print("Running challenging track test only...")
            run_simulation("challenging")
        elif test_type == "both":
            print("Running both tests...")
            run_both_tests()
        else:
            print("Usage: python test_spline_mpc.py [straight|curved|racetrack|challenging|both]")
            print("Running default curved path test...")
            run_simulation("curved")
    else:
        # Default: run straight-line test for debugging boundary behavior
        print("Running straight-line test (default for debugging)...")
        run_simulation("straight") 