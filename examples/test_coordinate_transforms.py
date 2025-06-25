import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Spline.spline_fitter import SplineFitter
from Dynamics.cubic_spline_path_dynamics import CubicSplinePathDynamics
from Dynamics.vehicle_model import VehicleModel
from CoordinateSystem.spline_curvilinear_coordinates import CurvilinearCoordinates


def test_coordinate_transforms():
    """Test the coordinate transformations to ensure they are consistent."""
    
    # Create test path
    waypoints = np.array([
        [0, 0],
        [5, 2],
        [10, 3],
        [15, 1],
        [20, -2],
        [25, 0],
        [30, 3]
    ])
    
    print("Test path created with waypoints:")
    print(waypoints)

    # Create spline path with chord-length parameterization
    spline_fitter = SplineFitter()
    spline = spline_fitter.fit_spline(waypoints)
    spline_coords = CurvilinearCoordinates(waypoints)

    # Create vehicle model and spline dynamics
    vehicle_model = VehicleModel()
    spline_dynamics = CubicSplinePathDynamics(vehicle_model, spline_coords)
    
    print(f"\nSpline parameter range: {spline_coords.u_values[0]} to {spline_coords.u_values[-1]}")
    print(f"Path length: {spline_coords.path_length}")
    print(f"Total chord length: {spline_coords.total_chord_length}")

    # Test different spline curvilinear states
    test_states = [
        np.array([1.0, 1.0, 0.0, 0.0, 2.0]),    # On path, aligned
        np.array([2.0, 2.0, 0.5, 0.2, 3.0]),    # Off path, misaligned
        np.array([5.0, 5.0, -0.3, -0.1, 5.0]),  # High speed, turning
    ]
    
    for i, spline_state in enumerate(test_states):
        print(f"\n{'='*50}")
        print(f"Test {i+1}: {spline_state}")
        print(f"{'='*50}")
        
        # Forward transform: spline curvilinear -> Cartesian
        cartesian_state = spline_dynamics.spline_curvilinear_to_cartesian(spline_state)
        print(f"Spline curvilinear: {spline_state}")
        print(f"Cartesian:          {cartesian_state}")
        
        # Backward transform: Cartesian -> spline curvilinear
        recovered_state = spline_dynamics.cartesian_to_spline_curvilinear(cartesian_state)
        print(f"Recovered:          {recovered_state}")
        
        # Check consistency
        diff = spline_state - recovered_state
        print(f"Difference:         {diff}")
        print(f"Max abs difference: {np.max(np.abs(diff)):.6f}")
        
        # Check if the transformation is consistent
        if np.max(np.abs(diff)) < 1e-2:
            print("✓ Transform is consistent")
        else:
            print("✗ Transform is NOT consistent")
            
        # Also test the individual components
        s, u, e_y, e_psi, v = spline_state
        s_rec, u_rec, e_y_rec, e_psi_rec, v_rec = recovered_state
        
        print(f"\nComponent analysis:")
        print(f"  s:    {s:.6f} -> {s_rec:.6f} (diff: {s - s_rec:.6f})")
        print(f"  u:    {u:.6f} -> {u_rec:.6f} (diff: {u - u_rec:.6f})")
        print(f"  e_y:  {e_y:.6f} -> {e_y_rec:.6f} (diff: {e_y - e_y_rec:.6f})")
        print(f"  e_ψ:  {e_psi:.6f} -> {e_psi_rec:.6f} (diff: {e_psi - e_psi_rec:.6f})")
        print(f"  v:    {v:.6f} -> {v_rec:.6f} (diff: {v - v_rec:.6f})")


if __name__ == "__main__":
    test_coordinate_transforms() 