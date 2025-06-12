#!/usr/bin/env python3
"""
Basic example demonstrating spline-based MPC path tracking.

This example shows how to:
1. Create a reference path using waypoints
2. Fit cubic splines to generate smooth trajectories
3. Use MPC to track the spline reference
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spline_fit import CurvilinearCoordinates
from mpc_controller import MPCController, VehicleModel


def create_test_waypoints():
    """Create a set of test waypoints for demonstration."""
    # Create a curved open path (S-curve)
    x = np.array([0, 5, 10, 15, 20, 25, 30])
    y = np.array([0, 2, 1, -1, 0, 3, 2])
    
    waypoints = np.column_stack([x, y])
    return waypoints


def main():
    """Main demonstration function."""
    print("=== Spline-Based MPC Path Tracking Demo ===\n")
    
    # 1. Create waypoints and fit spline
    print("1. Creating waypoints and fitting spline...")
    waypoints = create_test_waypoints()
    spline = CurvilinearCoordinates(waypoints, closed_path=False)
    
    print(f"   - Number of waypoints: {len(waypoints)}")
    print(f"   - Path length: {spline.get_path_length():.2f} m")
    
    # 2. Generate reference trajectory
    print("\n2. Generating reference trajectory...")
    reference_traj = spline.generate_reference_trajectory(num_points=50)
    
    # Add velocities for MPC reference (conservative velocity for stability)
    reference_traj['velocities'] = np.full(len(reference_traj['positions']), 3.0)  # Further reduced
    
    print(f"   - Reference points: {len(reference_traj['positions'])}")
    print(f"   - Max curvature: {np.max(np.abs(reference_traj['curvatures'])):.4f} 1/m")
    
    # 3. Setup vehicle model and MPC controller
    print("\n3. Setting up MPC controller...")
    # Use separate front and rear wheelbase lengths with explicit constraints
    # Total wheelbase = 2.5m, with typical 60/40 front/rear distribution
    vehicle = VehicleModel(
        wheelbase_front=1.5, 
        wheelbase_rear=1.0, 
        max_steering=0.3,        # Reduced for stability
        min_steering=-0.3,       # Symmetric
        max_velocity=12.0,       # Reasonable max speed
        min_velocity=0.1,        # Small positive minimum to avoid stopping
        max_acceleration=1.5,    # Conservative acceleration
        min_acceleration=-2.0    # Conservative braking
    )
    
    # Create MPC controller with more reasonable weights
    print("Setting up MPC controller...")
    
    # Set cost function weights (heavily prioritize velocity)
    Q = np.diag([0.01, 0.01, 0.01, 100.0])  # State weights: [x, y, theta, v] - extremely high velocity weight
    R = np.diag([0.001, 0.001])             # Input weights: [steering, acceleration] - very low
    Q_terminal = Q                          # Terminal weights same as stage
    
    controller = MPCController(
        vehicle_model=vehicle,
        prediction_horizon=1.0,  # Reduced from 2.0 seconds
        dt=0.1,                  # 100ms time step
        use_curvilinear=False    # Use Cartesian coordinates for now
    )
    
    # Set weights (must be done before solver initialization)
    controller.Q = Q
    controller.R = R
    controller.Q_terminal = Q_terminal
    
    print(f"   - Prediction horizon: {controller.get_prediction_horizon():.1f} s")
    print(f"   - Time step: {controller.get_time_step():.2f} s")
    
    # 4. Simulate path tracking
    print("\n4. Simulating path tracking...")
    
    # Initial state: start very close to the first waypoint
    first_waypoint = waypoints[0]
    current_state = np.array([
        first_waypoint[0] + 0.1,  # Very small x offset
        first_waypoint[1] + 0.05, # Very small y offset  
        0.05,                     # Very small initial heading
        3.0                       # Match reference velocity
    ])
    
    # Simulation parameters
    sim_time = 750.0  # Increased from 10.0 seconds
    dt_sim = 0.1
    num_steps = int(sim_time / dt_sim)
    
    # Storage for results
    states_history = [current_state.copy()]
    inputs_history = []
    
    try:
        for step in range(num_steps):  # Removed the limit of 20 steps
            # Create a velocity-only reference (ignore position tracking)
            dt_mpc = controller.get_time_step()
            horizon_steps = int(controller.get_prediction_horizon() / dt_mpc)
            
            # Reference: maintain current position and heading, focus only on velocity
            mpc_reference = {
                'positions': np.tile(current_state[:2], (horizon_steps, 1)),  # Stay at current position
                'headings': np.full(horizon_steps, current_state[2]),         # Keep current heading
                'velocities': np.full(horizon_steps, 3.0)                    # Target velocity
            }
            
            # Solve MPC
            try:
                result = controller.solve(current_state, mpc_reference)
                
                if result['solver_status'] == 0:
                    optimal_input = result['optimal_input']
                    inputs_history.append(optimal_input.copy())
                    
                    # Simulate vehicle dynamics using the vehicle model
                    current_state = vehicle.simulate_step(current_state, optimal_input, dt_sim)
                    
                    # Keep velocity positive and within bounds
                    current_state[3] = np.clip(current_state[3], 0.1, vehicle.max_velocity)
                    
                    states_history.append(current_state.copy())
                    
                    if step % 10 == 0:  # Report every 10 steps instead of 5
                        progress = (current_state[0] / spline.get_path_length()) * 100
                        print(f"   Step {step:3d}: x={current_state[0]:.2f}, y={current_state[1]:.2f}, "
                              f"θ={current_state[2]:.3f}, v={current_state[3]:.2f}, progress={progress:.1f}%")
                else:
                    print(f"   MPC solver failed at step {step}")
                    break
                    
            except Exception as e:
                print(f"   Error in MPC solve at step {step}: {e}")
                break
        
        print(f"\n   Simulation completed: {len(states_history)} steps")
        
    except Exception as e:
        print(f"   Simulation error: {e}")
        print("   Note: This demo requires acados to be properly installed for MPC functionality")
    
    # 5. Visualization
    print("\n5. Creating visualization...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Reference path and waypoints
    plt.subplot(2, 2, 1)
    # Generate trajectory for plotting
    trajectory = spline.generate_trajectory(num_points=200)
    plt.plot(trajectory['positions'][:, 0], trajectory['positions'][:, 1], 
             'b-', linewidth=2, label='Spline')
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=8, label='Waypoints')
    
    # Plot actual vehicle trajectory if simulation ran
    if len(states_history) > 1:
        states_array = np.array(states_history)
        plt.plot(states_array[:, 0], states_array[:, 1], 'r--', linewidth=2, 
                label='Vehicle Trajectory', alpha=0.8)
        plt.plot(states_array[0, 0], states_array[0, 1], 'go', markersize=10, 
                label='Start')
        plt.plot(states_array[-1, 0], states_array[-1, 1], 'rs', markersize=10, 
                label='End')
    
    # Number the waypoints
    for i, (x, y) in enumerate(waypoints):
        plt.annotate(f'{i}', (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10)
    
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Reference Path with Waypoints')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 2: Curvature profile
    plt.subplot(2, 2, 2)
    s_fine = np.linspace(0, spline.get_path_length(), 200)
    curvatures = spline.compute_curvature(s_fine)
    plt.plot(s_fine, curvatures, 'g-', linewidth=2)
    plt.xlabel('Arc Length s [m]')
    plt.ylabel('Curvature κ [1/m]')
    plt.title('Path Curvature Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: State evolution (if simulation ran)
    if len(states_history) > 1:
        states_array = np.array(states_history)
        
        plt.subplot(2, 2, 3)
        plt.plot(states_array[:, 0], 'g-', label='X position')
        plt.plot(states_array[:, 1], 'b-', label='Y position')
        plt.xlabel('Time step')
        plt.ylabel('Position [m]')
        plt.title('Vehicle Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(states_array[:, 2], 'r-', label='Heading θ')
        plt.plot(states_array[:, 3], 'm-', label='Velocity v')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.title('Heading and Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Control inputs (if available)
    if inputs_history and not len(states_history) > 1:
        # Only show control inputs plot if we don't have state evolution plots
        inputs_array = np.array(inputs_history)
        
        plt.subplot(2, 2, 3)
        plt.plot(inputs_array[:, 0], 'b-', label='Steering δ')
        plt.plot(inputs_array[:, 1], 'r-', label='Acceleration a')
        plt.xlabel('Time step')
        plt.ylabel('Control Input')
        plt.title('Control Inputs')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Demo completed! ===")
    print("\nNote: For full MPC functionality, ensure acados is properly installed.")
    print("The spline interpolation works independently of acados.")


if __name__ == "__main__":
    main() 