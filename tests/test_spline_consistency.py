import numpy as np
import casadi as ca
from spline_fit.spline_fitter import SplineFitter
from mpc_controller.spline_path_dynamics import SplinePathDynamics
from mpc_controller.vehicle_model import VehicleModel
from spline_fit.curvilinear_coordinates import CurvilinearCoordinates

def test_spline_consistency():
    """Test consistency between SplineFitter and CasADi spline implementations."""
    # Create test waypoints
    waypoints = np.array([
        [0, 0],
        [5, 2],
        [10, 3],
        [15, 1],
        [20, -2],
        [25, 0],
        [30, 3]
    ])
    
    # Create SplineFitter
    spline_fitter = SplineFitter()
    spline = spline_fitter.fit_spline(waypoints)
    
    # Create CurvilinearCoordinates for CasADi
    spline_coords = CurvilinearCoordinates(waypoints)
    
    # Create dummy vehicle model and spline dynamics for CasADi
    vehicle_model = VehicleModel()
    spline_dynamics = SplinePathDynamics(vehicle_model, spline_coords)
    
    # Debug s=0 case
    print("\nDEBUG: s=0 case:")
    print("SplineFitter tangent:", spline.evaluate_derivatives(0))
    print("SplineFitter tangent magnitude:", np.linalg.norm(spline.evaluate_derivatives(0)))
    
    tangent_casadi = spline_dynamics.spline_tangent(0)
    print("CasADi tangent:", tangent_casadi)
    print("CasADi tangent magnitude:", float(ca.sqrt(tangent_casadi[0]**2 + tangent_casadi[1]**2)))
    
    # Print spline coefficients
    print("\nDEBUG: Spline coefficients:")
    print("knots:", spline_dynamics.knots)
    print("coeffs_x:", spline_dynamics.coeffs_x)
    print("coeffs_y:", spline_dynamics.coeffs_y)
    
    # Test points along the spline
    s_values = np.linspace(0, spline.path_length, 100)
    
    print("\nTesting spline consistency:")
    print("s\t|dx/ds| (SplineFitter)\t|dx/ds| (CasADi)\tDiff")
    print("-" * 70)
    
    for s in s_values:
        # Get SplineFitter values
        tangent_fitter = spline.evaluate_derivatives(s)
        tangent_mag_fitter = np.linalg.norm(tangent_fitter)
        
        # Get CasADi values
        tangent_casadi = spline_dynamics.spline_tangent(s)
        tangent_mag_casadi = float(ca.sqrt(tangent_casadi[0]**2 + tangent_casadi[1]**2))
        
        # Print comparison
        print(f"{s:.2f}\t{tangent_mag_fitter:.6f}\t\t{tangent_mag_casadi:.6f}\t\t{abs(tangent_mag_fitter - tangent_mag_casadi):.6f}")
        
        # Check if tangent magnitudes are close
        assert abs(tangent_mag_fitter - tangent_mag_casadi) < 1e-6, \
            f"Tangent magnitude mismatch at s={s:.2f}: {tangent_mag_fitter:.6f} vs {tangent_mag_casadi:.6f}"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_spline_consistency() 