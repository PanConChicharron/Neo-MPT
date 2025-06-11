#!/usr/bin/env python3
"""
Simple spline-only demonstration.

This example demonstrates the spline interpolation functionality
without requiring acados installation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spline_fit import CubicSpline, SplineFitter, CurvilinearSpline


def main():
    """Demonstrate spline functionality."""
    print("=== Spline Interpolation Demo ===\n")
    
    # Create test waypoints
    print("1. Creating test waypoints...")
    waypoints = np.array([
        [0, 0],
        [5, 2],
        [10, 3],
        [15, 1],
        [20, -2],
        [25, 0],
        [30, 3]
    ])
    print(f"   Created {len(waypoints)} waypoints")
    
    # Test basic CubicSpline class
    print("\n2. Testing basic CubicSpline class...")
    t_values = np.linspace(0, 1, len(waypoints))
    basic_spline = CubicSpline(t_values, waypoints)
    
    # Evaluate spline
    t_eval = np.linspace(0, 1, 100)
    spline_points = basic_spline.evaluate(t_eval)
    curvatures = basic_spline.curvature(t_eval)
    
    print(f"   Evaluated spline at {len(t_eval)} points")
    print(f"   Max curvature: {np.max(np.abs(curvatures)):.4f}")
    
    # Test SplineFitter class
    print("\n3. Testing SplineFitter class...")
    spline_fitter = SplineFitter(waypoints)
    trajectory = spline_fitter.generate_trajectory(num_points=100)
    
    print(f"   Path length: {spline_fitter.get_path_length():.2f}")
    print(f"   Generated trajectory with {len(trajectory['positions'])} points")
    
    # Test CurvilinearSpline class
    print("\n4. Testing CurvilinearSpline class...")
    curv_spline = CurvilinearSpline(waypoints)
    
    # Test coordinate transformations
    test_point = np.array([10, 5])
    s, d = curv_spline.cartesian_to_curvilinear(test_point)
    reconstructed = curv_spline.curvilinear_to_cartesian(s, d)
    
    print(f"   Original point: [{test_point[0]:.2f}, {test_point[1]:.2f}]")
    print(f"   Curvilinear coords: s={s:.2f}, d={d:.2f}")
    print(f"   Reconstructed: [{reconstructed[0]:.2f}, {reconstructed[1]:.2f}]")
    print(f"   Reconstruction error: {np.linalg.norm(test_point - reconstructed):.6f}")
    
    # Test tracking error computation
    current_pos = np.array([12, 4])
    current_heading = 0.2
    tracking_error = curv_spline.compute_tracking_error(current_pos, current_heading)
    
    print(f"   Tracking error - lateral: {tracking_error['lateral_error']:.3f}, "
          f"heading: {tracking_error['heading_error']:.3f}")
    
    # Visualization
    print("\n5. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Basic spline comparison
    ax1 = axes[0, 0]
    ax1.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', markersize=8, label='Waypoints')
    ax1.plot(spline_points[:, 0], spline_points[:, 1], 'b-', linewidth=2, label='Cubic Spline')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Basic Cubic Spline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: SplineFitter trajectory
    ax2 = axes[0, 1]
    ax2.plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=8, label='Waypoints')
    ax2.plot(trajectory['positions'][:, 0], trajectory['positions'][:, 1], 
             'g-', linewidth=2, label='SplineFitter')
    
    # Add arrows to show direction
    for i in range(0, len(trajectory['positions']), 10):
        pos = trajectory['positions'][i]
        vel = trajectory['velocities'][i]
        vel_norm = vel / np.linalg.norm(vel) * 2
        ax2.arrow(pos[0], pos[1], vel_norm[0], vel_norm[1], 
                 head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.7)
    
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('SplineFitter with Direction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Curvature profile
    ax3 = axes[1, 0]
    s_values = trajectory['s_values']
    curvatures = trajectory['curvatures']
    ax3.plot(s_values, curvatures, 'purple', linewidth=2)
    ax3.set_xlabel('Arc Length s [m]')
    ax3.set_ylabel('Curvature κ [1/m]')
    ax3.set_title('Curvature Profile')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coordinate transformation visualization
    ax4 = axes[1, 1]
    
    # Plot reference path
    ref_traj = curv_spline.generate_reference_trajectory(num_points=100)
    ax4.plot(ref_traj['positions'][:, 0], ref_traj['positions'][:, 1], 
             'g-', linewidth=2, alpha=0.7, label='Reference Path')
    
    # Plot waypoints
    ax4.plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=8, label='Waypoints')
    
    # Show coordinate transformation examples
    test_points = np.array([[5, 3], [15, -1], [25, 2]])
    
    for i, point in enumerate(test_points):
        # Find closest point on curve using Newton's method
        closest_point = curv_spline.find_closest_point(point)
        
        # Plot the actual query point
        ax4.plot(point[0], point[1], 'bo', markersize=8, 
                label='Query Points' if i == 0 else "")
        
        # Plot the closest point on the curve
        ax4.plot(closest_point[0], closest_point[1], 'rs', markersize=8,
                label='Closest Points' if i == 0 else "")
        
        # Draw line connecting query point to closest point on curve
        ax4.plot([point[0], closest_point[0]], [point[1], closest_point[1]], 
                'k--', linewidth=2, alpha=0.7,
                label='Distance Lines' if i == 0 else "")
    
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_title('Closest Point Finding on Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Spline Demo Completed Successfully! ===")
    print("\nThe spline interpolation module is working correctly.")
    print("You can now proceed to install acados for full MPC functionality.")


if __name__ == "__main__":
    main() 