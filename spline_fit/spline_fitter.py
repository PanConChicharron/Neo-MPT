import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from .cubic_spline import CubicSpline


class SplineFitter:
    """
    High-level interface for fitting splines to waypoints and generating trajectories.
    
    This class uses the modular CubicSpline class and provides:
    - Waypoint management and parameterization
    - Trajectory generation with specified resolution
    - Visualization and analysis tools
    - Path queries and utilities
    """
    
    def __init__(self, waypoints: np.ndarray, closed_path: bool = False):
        """
        Initialize the SplineFitter with waypoints.
        
        Args:
            waypoints: Array of shape (N, 2) containing [x, y] coordinates
            closed_path: Whether the path should be closed (connect last to first point)
        """
        self.waypoints = np.array(waypoints)
        self.closed_path = closed_path
        self.spline = None
        self.path_length = None
        self.s_values = None
        
        if self.waypoints.shape[1] != 2:
            raise ValueError("Waypoints must be of shape (N, 2) for [x, y] coordinates")
        
        self._parameterize_waypoints()
        self._create_spline()
    
    def _parameterize_waypoints(self):
        """Parameterize waypoints using cumulative arc length."""
        # Calculate cumulative distance along waypoints
        distances = np.zeros(len(self.waypoints))
        for i in range(1, len(self.waypoints)):
            dx = self.waypoints[i, 0] - self.waypoints[i-1, 0]
            dy = self.waypoints[i, 1] - self.waypoints[i-1, 1]
            distances[i] = distances[i-1] + np.sqrt(dx**2 + dy**2)
        
        self.s_values = distances
        self.path_length = distances[-1]
    
    def _create_spline(self):
        """Create the cubic spline using the modular CubicSpline class."""
        if self.closed_path:
            # For closed paths, use natural boundary conditions instead of periodic
            # to avoid issues with parameter ordering
            self.spline = CubicSpline(self.s_values, self.waypoints, bc_type='natural')
        else:
            self.spline = CubicSpline(self.s_values, self.waypoints, bc_type='natural')
    
    def evaluate(self, s: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the spline at given parameter values.
        
        Args:
            s: Parameter value(s) along the spline (0 to path_length)
            
        Returns:
            Array of [x, y] coordinates
        """
        return self.spline.evaluate(s)
    
    def evaluate_derivatives(self, s: Union[float, np.ndarray], order: int = 1) -> np.ndarray:
        """
        Evaluate derivatives of the spline.
        
        Args:
            s: Parameter value(s) along the spline
            order: Derivative order (1 for velocity, 2 for acceleration)
            
        Returns:
            Array of derivative values [dx/ds, dy/ds] or [d²x/ds², d²y/ds²]
        """
        return self.spline.derivative(s, order=order)
    
    def compute_curvature(self, s: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute curvature at given parameter values.
        
        Args:
            s: Parameter value(s) along the spline
            
        Returns:
            Curvature values
        """
        return self.spline.curvature(s)
    
    def generate_trajectory(self, num_points: int = 100) -> dict:
        """
        Generate a smooth trajectory with specified number of points.
        
        Args:
            num_points: Number of points in the generated trajectory
            
        Returns:
            Dictionary containing trajectory data:
            - 'positions': [x, y] coordinates
            - 'velocities': [dx/ds, dy/ds] derivatives
            - 'accelerations': [d²x/ds², d²y/ds²] second derivatives
            - 'curvatures': curvature values
            - 's_values': parameter values
            - 'path_length': total path length
        """
        s_trajectory = np.linspace(0, self.path_length, num_points)
        
        positions = self.evaluate(s_trajectory)
        velocities = self.evaluate_derivatives(s_trajectory, order=1)
        accelerations = self.evaluate_derivatives(s_trajectory, order=2)
        curvatures = self.compute_curvature(s_trajectory)
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'curvatures': curvatures,
            's_values': s_trajectory,
            'path_length': self.path_length
        }
    
    def get_closest_point(self, query_point: np.ndarray, num_samples: int = 100) -> Tuple[float, np.ndarray]:
        """
        Find the closest point on the spline to a given query point using Newton's method.
        
        Args:
            query_point: [x, y] coordinates of query point
            num_samples: Number of initial samples for coarse search (ignored, uses waypoints instead)
            
        Returns:
            Tuple of (parameter_value, closest_point)
        """
        # Use waypoints for coarse search instead of dense sampling
        # This is much more efficient and leverages the fact that curvature
        # varies monotonically across spline segments
        waypoint_distances = np.linalg.norm(self.waypoints - query_point, axis=1)
        min_waypoint_idx = np.argmin(waypoint_distances)
        s_initial = self.s_values[min_waypoint_idx]
        
        # Newton's method to refine the closest point
        # We want to minimize ||P(s) - query_point||^2
        # The derivative is: 2 * (P(s) - query_point) · P'(s) = 0
        # Newton's method: s_new = s - f(s)/f'(s)
        # where f(s) = (P(s) - query_point) · P'(s)
        # and f'(s) = P'(s) · P'(s) + (P(s) - query_point) · P''(s)
        
        s = s_initial
        max_iterations = 20
        tolerance = 1e-8
        
        for i in range(max_iterations):
            # Evaluate spline and derivatives at current s
            point = self.evaluate(np.array([s]))[0]  # P(s)
            first_deriv = self.evaluate_derivatives(np.array([s]), order=1)[0]  # P'(s)
            second_deriv = self.evaluate_derivatives(np.array([s]), order=2)[0]  # P''(s)
            
            # Vector from spline point to query point
            diff = point - query_point  # P(s) - query_point
            
            # First derivative of objective function: f(s) = diff · P'(s)
            # For 2D vectors, this is diff[0]*first_deriv[0] + diff[1]*first_deriv[1]
            f = np.sum(diff * first_deriv)
            
            # Second derivative of objective function: f'(s) = P'(s) · P'(s) + diff · P''(s)
            f_prime = np.sum(first_deriv * first_deriv) + np.sum(diff * second_deriv)
            
            # Check for convergence
            if abs(f) < tolerance:
                break
                
            # Avoid division by zero
            if abs(f_prime) < 1e-12:
                break
                
            # Newton's method update
            s_new = s - f / f_prime
            
            # Clamp to valid range
            s_new = np.clip(s_new, 0, self.path_length)
            
            # Check for convergence in parameter space
            if abs(s_new - s) < tolerance:
                break
                
            s = s_new
        
        # Return final result
        closest_point = self.evaluate(np.array([s]))[0]
        
        # Debug: verify the result
        if hasattr(closest_point, 'shape') and closest_point.shape != (2,):
            print(f"Warning: closest_point has unexpected shape {closest_point.shape}")
            print(f"closest_point value: {closest_point}")
        
        return s, closest_point
    
    def find_closest_point(self, query_point: np.ndarray) -> np.ndarray:
        """
        Find the closest point on the spline to a given query point.
        
        Args:
            query_point: [x, y] coordinates of query point
            
        Returns:
            [x, y] coordinates of closest point on spline
        """
        _, closest_point = self.get_closest_point(query_point)
        return closest_point
    
    def get_path_length(self) -> float:
        """Get the total path length."""
        return self.path_length
    
    def get_waypoints(self) -> np.ndarray:
        """Get the original waypoints."""
        return self.waypoints.copy()
    
    def get_parameter_values(self) -> np.ndarray:
        """Get the parameter values for waypoints."""
        return self.s_values.copy()
    
    def plot_spline(self, num_points: int = 200, show_waypoints: bool = True, 
                   show_curvature: bool = False, figsize: tuple = (12, 5)):
        """
        Plot the fitted spline.
        
        Args:
            num_points: Number of points for smooth curve visualization
            show_waypoints: Whether to show original waypoints
            show_curvature: Whether to show curvature plot
            figsize: Figure size for the plot
        """
        trajectory = self.generate_trajectory(num_points)
        
        if show_curvature:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        
        # Plot trajectory
        ax1.plot(trajectory['positions'][:, 0], trajectory['positions'][:, 1], 
                'b-', linewidth=2, label='Spline')
        
        if show_waypoints:
            ax1.plot(self.waypoints[:, 0], self.waypoints[:, 1], 
                    'ro', markersize=8, label='Waypoints')
            
            # Number the waypoints
            for i, (x, y) in enumerate(self.waypoints):
                ax1.annotate(f'{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10)
        
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Cubic Spline Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot curvature
        if show_curvature:
            ax2.plot(trajectory['s_values'], trajectory['curvatures'], 'g-', linewidth=2)
            ax2.set_xlabel('Path Parameter s [m]')
            ax2.set_ylabel('Curvature [1/m]')
            ax2.set_title('Curvature along Path')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def resample_trajectory(self, ds: float) -> dict:
        """
        Resample trajectory with constant arc length spacing.
        
        Args:
            ds: Arc length spacing between points
            
        Returns:
            Dictionary containing resampled trajectory data
        """
        num_points = int(np.ceil(self.path_length / ds)) + 1
        return self.generate_trajectory(num_points)
    
    def get_tangent_angle(self, s: Union[float, np.ndarray]) -> np.ndarray:
        """
        Get tangent angle at given parameter values.
        
        Args:
            s: Parameter value(s) along the spline
            
        Returns:
            Tangent angles in radians
        """
        derivatives = self.evaluate_derivatives(s, order=1)
        if derivatives.ndim == 1:
            return np.arctan2(derivatives[1], derivatives[0])
        else:
            return np.arctan2(derivatives[:, 1], derivatives[:, 0]) 