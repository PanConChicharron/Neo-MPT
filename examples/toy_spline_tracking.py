import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Spline.spline_fitter import SplineFitter
from MPC.spline_path_dynamics import SplinePathDynamics
from Dynamics.vehicle_model import VehicleModel
from CoordinateSystem.curvilinear_spline import CurvilinearCoordinates
from Dynamics.cubic_spline_path_dynamics import CubicSplinePathDynamics


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


def run_toy_simulation():
    # Create test path
    waypoints = create_test_path()
    print("Test path created with waypoints:")
    print(waypoints)

    # Create spline path with chord-length parameterization
    spline_fitter = SplineFitter()
    spline = spline_fitter.fit_spline(waypoints)
    spline_coords = CurvilinearCoordinates(waypoints)

    # Create vehicle model and spline dynamics
    vehicle_model = VehicleModel()
    spline_dynamics = CubicSplinePathDynamics(vehicle_model, spline_coords)

    # Initial state [s, u, e_y, e_ψ, v]
    initial_state = np.array([
        0.1,    # s: start slightly ahead
        0.1,    # u: start slightly ahead
        0.0,    # e_y: on path
        0.0,    # e_ψ: aligned with path
        1.0     # v: constant velocity
    ])

    # Simulation parameters
    sim_time = 10.0  # seconds
    dt = 0.1
    n_steps = int(sim_time / dt)

    # Storage for results
    states = np.zeros((n_steps + 1, 5))  # [s, u, e_y, e_ψ, v]
    positions = np.zeros((n_steps + 1, 2))  # [x, y]
    reference_positions = np.zeros((n_steps + 1, 2))  # [x, y]

    # Initial state
    states[0] = initial_state
    positions[0] = spline_dynamics.spline_curvilinear_to_cartesian(initial_state)[:2]
    reference_positions[0] = spline_dynamics.spline_coords.curvilinear_to_cartesian(initial_state[1], 0.0)

    # Print valid knot range
    print(f"Spline knots: {spline_dynamics.knots}")
    print(f"Valid u range: {spline_dynamics.knots[0]} to {spline_dynamics.knots[-1]}")

    # Simulation loop
    for i in range(n_steps):
        current_state = states[i]
        current_pos = positions[i]

        # Simple proportional controller for lateral error and heading error
        delta = -0.5 * current_state[2] - 0.1 * current_state[3]
        a = 0.0  # constant velocity, no acceleration
        input = np.array([delta, a])

        # Simulate one step
        next_state = spline_dynamics.simulate_step(current_state, input, dt)
        states[i+1] = next_state

        # Debug print for u
        u_val = next_state[1]
        print(f"u at step {i+1}: {u_val}")
        if u_val < spline_dynamics.knots[0] or u_val > spline_dynamics.knots[-1]:
            print(f"WARNING: u={u_val} is out of bounds! Valid range: {spline_dynamics.knots[0]} to {spline_dynamics.knots[-1]}")

        # Convert to Cartesian coordinates for visualization
        next_pos = spline_dynamics.spline_curvilinear_to_cartesian(next_state)[:2]
        positions[i+1] = next_pos

        # Get reference position for visualization
        ref_pos = spline_dynamics.spline_coords.curvilinear_to_cartesian(next_state[1], 0.0)
        reference_positions[i+1] = ref_pos

        print(f"Step {i+1}/{n_steps}")
        print(f"Current state: {current_state}")
        print(f"Current position: {current_pos}")
        print(f"Reference position: {ref_pos}")
        print(f"Input: {input}")
        print(f"Next state: {next_state}")

    # Plot results
    plot_results(positions, reference_positions, spline_coords)


def plot_results(positions, reference_positions, spline_coords):
    """Plot simulation results."""
    plt.figure(figsize=(10, 8))
    plt.plot(reference_positions[:, 0], reference_positions[:, 1], 'b-', label='Reference Path')
    plt.plot(positions[:, 0], positions[:, 1], 'r-', label='Vehicle Trajectory')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Toy Spline Tracking Example')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    run_toy_simulation() 