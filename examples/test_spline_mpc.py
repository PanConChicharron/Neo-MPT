import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MPC.mpc_controller import MPCController
from Dynamics.vehicle_model import VehicleModel
from Dynamics.cubic_spline_path_dynamics import CubicSplinePathDynamics
from CoordinateSystem.spline_curvilinear_path import SplineCurvilinearPath
from paths import (
    create_test_path,
    create_straight_line_path,
    create_racetrack_path,
    create_challenging_track
)
from utils import get_local_waypoints

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
        
    # Debug plot spline and derivatives
    # plot_spline_debug(spline)
    
    spline_curvilinear_path = SplineCurvilinearPath(waypoints, closed_path=False)
    
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

    num_waypoints = 20# len(waypoints)
    
    # Create spline path dynamics
    spline_dynamics = CubicSplinePathDynamics(vehicle_model, num_waypoints)
    
    # Set cost weights (must be done before set_path)
    state_weights = np.array([1e0, 0.0, 1e2, 1e-1, 1e-1])  # [s, u, e_y, e_ψ, v] - velocity weight = 0.0
    input_weights = np.array([5, 10])  # [delta, a] - very low acceleration penalty
    terminal_state_weights = 2 * state_weights  # [s, u, e_y, e_ψ, v] - terminal velocity weight = 0.0

    # Create MPC controller with more conservative settings
    mpc = MPCController(
        state_weights=state_weights,
        input_weights=input_weights,
        terminal_state_weights=terminal_state_weights,
        vehicle_model=vehicle_model,
        spline_dynamics=spline_dynamics,
        prediction_horizon=5.0,
        dt=0.1,
    )
    
    # Initial state [s, u, e_y, e_ψ, v]
    # Start near the beginning of the path with proper s/u relationship
    initial_u = 0.0  # cubic spline parameter
    initial_s = spline_curvilinear_path.chord_to_arc_length(initial_u)  # convert to arc-length
    
    initial_state = np.array([
        initial_s,  # s: arc-length progress
        initial_u,  # u: chord-length parameter
        0.0,        # e_y: on path
        0.0,        # e_ψ: aligned with path
        2.0         # v: moderate initial velocity to 2 m/s
    ])
    
    # Simulation parameters
    sim_time = 250.0  # seconds
    dt = mpc.dt
    n_steps = int(sim_time / dt)
    
    # Initialize arrays to store results
    states = np.zeros((n_steps + 1, 5))
    inputs = np.zeros((n_steps, 2))
    cartesian_states = np.zeros((n_steps + 1, 2))
    solve_times = np.zeros(n_steps)  # Track MPC solve times
    states[0] = initial_state

    # Get initial spline parameters vector
    parameters = spline_curvilinear_path.get_parameters()
    parameters = np.concatenate((parameters, [spline_curvilinear_path.u_values[0], spline_curvilinear_path.u_values[-1], spline_curvilinear_path.path_length]))
    
    # Main simulation loop
    for step in range(n_steps):
        print(f"\nStep {step + 1}/{n_steps}")
        
        # Check if we've reached the end of the path
        current_s = states[step][0]
        path_length = spline_curvilinear_path.get_path_length()
        remaining_path = path_length - current_s
        
        print(f"  Current s: {current_s:.2f} m, Path length: {path_length:.2f} m, Remaining: {remaining_path:.2f} m")
        
        # Stop simulation if we've reached the end of the path
        if remaining_path <= 0.5:  # Within 10cm of the end
            print(f"🏁 Reached end of path! Stopping simulation at step {step}")
            print(f"   Final position: s={current_s:.2f} m (path length: {path_length:.2f} m)")
            # Trim arrays to actual simulation length
            states = states[:step+1]
            inputs = inputs[:step]
            break
        
        # Get current position
        current_pos = spline_curvilinear_path.curvilinear_to_cartesian(states[step][1], states[step][2])[:2]
        cartesian_states[step] = current_pos
        
        # Check if coordinate transformation is failing (position stuck or invalid)
        if step > 0:
            prev_pos = spline_curvilinear_path.curvilinear_to_cartesian(states[step-1][1], states[step-1][0])[:2]
            pos_change = np.linalg.norm(current_pos - prev_pos)
            if pos_change < 1e-6 and states[step][4] > 0.1:  # Position not changing but velocity > 0
                print(f"⚠️  WARNING: Vehicle position stuck at {current_pos}, but velocity = {states[step][4]:.2f} m/s")
                print(f"   This indicates coordinate transformation failure. Stopping simulation.")
                states = states[:step+1]
                inputs = inputs[:step]
                break
        
        # Update spline parameters only for non-racetrack paths
        # For racetrack and challenging track, we use the full closed-loop path throughout
        # Update parameters after waypoints change
        sub_spline_parameters = spline_curvilinear_path.get_sub_spline_parameters(current_pos, num_waypoints)
        parameters = np.concatenate((sub_spline_parameters, [spline_curvilinear_path.u_values[0], spline_curvilinear_path.u_values[-1], spline_curvilinear_path.path_length]))
    
        
        lookahead_distance = 25.0
        s_end = min(current_s + lookahead_distance, path_length)
        s_values = np.linspace(current_s, s_end, mpc.N + 1)
            
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
        result = mpc.solve(states[step], parameters, reference_trajectory)
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
        states[step + 1] = spline_dynamics.simulate_step(states[step], inputs[step], parameters[0:-3], dt)
        
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
    plot_results(states, inputs, solve_times, dt, spline_curvilinear_path, cartesian_states)
    
    return {
        'states': states,
        'inputs': inputs,
        'solve_times': solve_times
    }

def plot_results(states, inputs, solve_times, dt, spline_curvilinear_path, cartesian_states):
    """Plot simulation results including MPC timing information."""
    # Create time vector
    t = np.arange(len(states)) * dt
    
    # Create figure with subplots - changed to 4x3 to add frequency plots
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    
    # Plot path and vehicle trajectory
    axs[0, 0].set_title('Path and Vehicle Trajectory')
    # Plot reference path
    u_values = np.linspace(0, spline_curvilinear_path.chord_length, 1000)
    path_points = np.array([spline_curvilinear_path.curvilinear_to_cartesian(u, 0) for u in u_values])
    axs[0, 0].plot(path_points[:, 0], path_points[:, 1], 'b-', label='Reference Path')
    axs[0, 0].plot(cartesian_states[:, 0], cartesian_states[:, 1], 'r-', label='Vehicle Trajectory')
    
    
    # Plot vehicle trajectory
    vehicle_points = np.array([spline_curvilinear_path.curvilinear_to_cartesian(state[1], state[2])[:2] for state in states])
    axs[0, 0].plot(vehicle_points[:, 0], vehicle_points[:, 1], 'r-', label='Vehicle')

    # Plot vehicle trajectory
    vehicle_points = np.array([spline_curvilinear_path.curvilinear_to_cartesian(state[1], state[2])[:2] for state in states])
    axs[0, 0].plot(vehicle_points[:, 0], vehicle_points[:, 1], 'r-', label='Vehicle')
    
    # Mark initial and final poses
    initial_state = states[0]
    initial_pos = cartesian_states[0]
    initial_heading = spline_curvilinear_path.get_heading(initial_state[0])
    arrow_length = 2.0
    initial_dx = arrow_length * np.cos(initial_heading)
    initial_dy = arrow_length * np.sin(initial_heading)
    axs[0, 0].arrow(initial_pos[0], initial_pos[1], initial_dx, initial_dy, 
                   head_width=0.5, head_length=0.8, fc='green', ec='green', linewidth=2)
    axs[0, 0].plot(initial_pos[0], initial_pos[1], 'go', markersize=8, label='Start')
    
    final_state = states[-1]
    final_pos = cartesian_states[-1]
    final_heading = spline_curvilinear_path.get_heading(final_state[0])
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
    axs[0, 1].plot(t, states[:, 1], label='u (chord length)')
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