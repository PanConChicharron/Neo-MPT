import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Spline.spline_fitter import SplineFitter
from MPC.spline_path_dynamics import SplinePathDynamics
from Dynamics.vehicle_model import VehicleModel
from CoordinateSystem.curvilinear_coordinates import CurvilinearCoordinates
from Dynamics.spline_path_dynamics import SplinePathDynamics


def create_test_path():
    """Create a test path with waypoints."""
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
    """Create a straight line test path."""
    waypoints = np.array([
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
        [40, 0]
    ])
    return waypoints


def compare_dynamics():
    """Compare spline dynamics with vehicle dynamics transformed through curvilinear coordinates."""
    
    # Test both curved and straight paths
    test_paths = [
        ("Curved Path", create_test_path()),
        ("Straight Line", create_straight_line_path())
    ]
    
    for path_name, waypoints in test_paths:
        print(f"\n{'='*60}")
        print(f"TESTING: {path_name}")
        print(f"{'='*60}")
        print("Waypoints:")
        print(waypoints)

        # Create spline path with chord-length parameterization
        spline_fitter = SplineFitter()
        spline = spline_fitter.fit_spline(waypoints)
        spline_coords = CurvilinearCoordinates(waypoints)

        # Create vehicle model and spline dynamics
        vehicle_model = VehicleModel()
        spline_dynamics = SplinePathDynamics(vehicle_model, spline_coords)

        # Test parameters
        dt = 0.1
        n_test_steps = 10

        # For straight line, use simpler test case
        if path_name == "Straight Line":
            test_cases = [{
                'name': 'Straight line test',
                'curvilinear_state': np.array([5.0, 5.0, 0.0, 0.0, 2.0]),  # [s, u, e_y, e_ψ, v]
                'input': np.array([0.0, 0.0])  # [delta, a] - no steering, no acceleration
            }]
        else:
            # Define chord-length parameters and compute corresponding arc-lengths
            test_u_values = [1.0, 2.0, 5.0]  # Chord-length parameters
            
            # Test different initial states and inputs
            test_cases = []
            for i, u_val in enumerate(test_u_values):
                # Compute corresponding arc-length using the actual function
                s_val = spline_coords.chord_to_arc_length(u_val)
                
                if i == 0:
                    test_cases.append({
                        'name': 'On path, aligned',
                        'curvilinear_state': np.array([s_val, u_val, 0.0, 0.0, 2.0]),  # [s, u, e_y, e_ψ, v]
                        'input': np.array([0.1, 0.5])  # [delta, a]
                    })
                    break  # Only test first case for curved path to keep output manageable

        for test_case in test_cases:
            print(f"\n{'-'*50}")
            print(f"Test Case: {test_case['name']}")
            print(f"{'-'*50}")
            
            curvilinear_state = test_case['curvilinear_state']
            input_vec = test_case['input']
            
            print(f"Initial curvilinear state: {curvilinear_state}")
            print(f"Input: {input_vec}")
            
            # Method 1: Direct spline dynamics
            print("\n--- Method 1: Direct Spline Dynamics ---")
            current_curvilinear = curvilinear_state.copy()
            spline_states = [current_curvilinear.copy()]
            spline_cartesian_states = []
            
            for i in range(n_test_steps):
                # Convert to Cartesian for storage
                cartesian_state = spline_dynamics.spline_curvilinear_to_cartesian(current_curvilinear)
                spline_cartesian_states.append(cartesian_state.copy())
                
                # Simulate spline dynamics directly
                next_curvilinear = spline_dynamics.simulate_step(current_curvilinear, input_vec, dt)
                spline_states.append(next_curvilinear.copy())
                current_curvilinear = next_curvilinear

            # Add final Cartesian state
            final_cartesian = spline_dynamics.spline_curvilinear_to_cartesian(current_curvilinear)
            spline_cartesian_states.append(final_cartesian.copy())

            # Method 2: Vehicle dynamics + coordinate transformation
            print("\n--- Method 2: Vehicle Dynamics + Coordinate Transformation ---")
            # Convert initial curvilinear state to Cartesian
            initial_cartesian = spline_dynamics.spline_curvilinear_to_cartesian(curvilinear_state)
            current_cartesian = initial_cartesian.copy()
            vehicle_states = [current_cartesian.copy()]
            curvilinear_from_vehicle = [curvilinear_state.copy()]
            
            for i in range(n_test_steps):
                # Simulate vehicle dynamics in Cartesian coordinates
                next_cartesian = vehicle_model.simulate_step(current_cartesian, input_vec, dt)
                vehicle_states.append(next_cartesian.copy())
                
                # Transform back to curvilinear coordinates
                next_curvilinear_from_vehicle = spline_dynamics.cartesian_to_spline_curvilinear(next_cartesian)
                curvilinear_from_vehicle.append(next_curvilinear_from_vehicle.copy())
                
                current_cartesian = next_cartesian

            # Validate constraint s = ∫₀ᵘ |dP/du| du for Method 1
            print(f"\n--- Constraint Validation for {path_name} ---")
            print("Checking s = ∫₀ᵘ |dP/du| du constraint:")
            max_error = 0.0
            for i in range(min(5, n_test_steps + 1)):  # Check first 5 steps
                s_actual = spline_states[i][0]
                u_actual = spline_states[i][1]
                
                # Compute expected s from u using the spline's arc length integration
                s_expected = spline_coords.chord_to_arc_length(u_actual)
                constraint_error = abs(s_actual - s_expected)
                max_error = max(max_error, constraint_error)
                
                print(f"Step {i}: s_actual={s_actual:.6f}, s_expected={s_expected:.6f}, error={constraint_error:.8f}")
            
            print(f"Maximum constraint error: {max_error:.8f}")
            
            # For straight line, this should be near zero
            if path_name == "Straight Line":
                if max_error < 1e-6:
                    print("✓ PASS: Constraint maintained for straight line (integrator working correctly)")
                else:
                    print("✗ FAIL: Constraint violated even for straight line (fundamental issue)")

            # Compare the two methods
            print(f"\n--- Comparison ---")
            print("Curvilinear state differences (Method 1 - Method 2):")
            for i in range(min(6, n_test_steps + 1)):  # Show first 6 steps
                curv_diff = spline_states[i] - curvilinear_from_vehicle[i]
                max_curv_diff = np.max(np.abs(curv_diff))
                print(f"Step {i}: {curv_diff} (max abs: {max_curv_diff:.6f})")
            
            print("\nCartesian state differences (Method 1 - Method 2):")
            for i in range(min(6, n_test_steps + 1)):  # Show first 6 steps
                cart_diff = spline_cartesian_states[i] - vehicle_states[i]
                max_cart_diff = np.max(np.abs(cart_diff))
                print(f"Step {i}: {cart_diff} (max abs: {max_cart_diff:.6f})")

            # Create comparison plot
            plot_comparison(spline_states, curvilinear_from_vehicle, f"{path_name} - {test_case['name']}")
        
        print(f"\nCompleted testing: {path_name}")
        print("="*60)


def plot_comparison(spline_states, curvilinear_from_vehicle, test_name):
    """Plot comparison between the two methods."""
    # Convert lists to numpy arrays for plotting
    spline_states = np.array(spline_states)
    curvilinear_from_vehicle = np.array(curvilinear_from_vehicle)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'State Comparison: {test_name}')
    
    state_names = ['s', 'u', 'e_y', 'e_ψ', 'v']
    
    for i, name in enumerate(state_names):
        row = i // 3
        col = i % 3
        
        axes[row, col].plot(spline_states[:, i], 'b-', label='Spline Dynamics', linewidth=2)
        axes[row, col].plot(curvilinear_from_vehicle[:, i], 'r--', label='Vehicle + Transform', linewidth=2)
        axes[row, col].set_title(f'{name}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # Remove the empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_dynamics() 