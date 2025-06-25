import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from .cubic_spline import CubicSpline


class ChordLengthParametricSpline2D:
    """
    2D cubic spline with chord-length parameterization for waypoint fitting.
    
    This class uses chord-length parameterization where parameter values are based
    on cumulative Euclidean distances between waypoints. This provides:
    - Natural curve shapes without clustering artifacts
    - Proportional parameter spacing based on waypoint distances
    - Efficient computation for path planning applications
    
    The class uses the modular CubicSpline class and provides:
    - Waypoint management and chord-length parameterization
    - Trajectory generation with specified resolution
    - Visualization and analysis tools
    - Path queries and utilities
    
    Note: Uses 'u' for chord-length parameter to distinguish from arc-length 's'.
    The actual path length is computed using RK4 integration for accurate arc length.
    """
    
    def __init__(self, num_waypoints: int, closed_path: bool = False):
        """
        Initialize the chord-length parameterized spline with waypoints.
        
        Args:
            waypoints: Array of shape (N, 2) containing [x, y] coordinates
            closed_path: Whether the path should be closed (connect last to first point)
        """
        self.closed_path = closed_path
        self.spline = None
        self.path_length = None
        self.u_values = None  # Chord-length parameters
        
    def set_waypoints(self, waypoints: np.ndarray):
        """
        Set the waypoints for the spline.
        
        Args:
            waypoints: Array of shape (N, 2) containing [x, y] coordinates
        """        
        if waypoints.shape[1] != 2:
            raise ValueError("Waypoints must be of shape (N, 2) for [x, y] coordinates")
        
        self.waypoints = waypoints
        
        self._parameterize_waypoints()
        self._create_spline()
    
    def _parameterize_waypoints(self):
        """Parameterize waypoints using cumulative chord lengths."""
        # Calculate cumulative distance along waypoints
        distances = np.zeros(len(self.waypoints))
        for i in range(1, len(self.waypoints)):
            dx = self.waypoints[i, 0] - self.waypoints[i-1, 0]
            dy = self.waypoints[i, 1] - self.waypoints[i-1, 1]
            distances[i] = distances[i-1] + np.sqrt(dx**2 + dy**2)
        
        self.u_values = distances  # Chord-length parameters
        self.chord_length = distances[-1]
        # Note: path_length will be computed as actual arc length using RK4 after spline creation
    
    def _create_spline(self):
        """Create the cubic spline using the modular CubicSpline class."""
        if self.closed_path:
            # For closed paths, use natural boundary conditions instead of periodic
            # to avoid issues with parameter ordering
            self.spline = CubicSpline(self.u_values, self.waypoints, bc_type='natural')
        else:
            self.spline = CubicSpline(self.u_values, self.waypoints, bc_type='natural')
        
        # Compute actual arc length using the spline's built-in RK4 integration
        self.path_length = self.spline.arc_length()
    
    def evaluate(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the spline at given chord-length parameter values.
        
        Args:
            u: Chord-length parameter value(s) along the spline (0 to path_length)
            
        Returns:
            Array of [x, y] coordinates
        """
        return self.spline.evaluate(u)
    
    def evaluate_derivatives(self, u: Union[float, np.ndarray], order: int = 1) -> np.ndarray:
        """
        Evaluate derivatives of the spline.
        
        Args:
            u: Chord-length parameter value(s) along the spline
            order: Derivative order (1 for velocity, 2 for acceleration)
            
        Returns:
            Array of derivative values [dx/du, dy/du] or [d²x/du², d²y/du²]
        """
        return self.spline.derivative(u, order=order)
    
    def compute_curvature(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute curvature at given chord-length parameter values.
        
        Args:
            u: Chord-length parameter value(s) along the spline
            
        Returns:
            Curvature values
        """
        return self.spline.curvature(u)
    
    def generate_trajectory(self, num_points: int = 100) -> dict:
        """
        Generate a smooth trajectory with specified number of points.
        
        Args:
            num_points: Number of points in the generated trajectory
            
        Returns:
            Dictionary containing trajectory data:
            - 'positions': [x, y] coordinates
            - 'velocities': [dx/du, dy/du] derivatives
            - 'accelerations': [d²x/du², d²y/du²] second derivatives
            - 'curvatures': curvature values
            - 'u_values': chord-length parameter values
            - 'path_length': total path length (chord length)
        """
        u_trajectory = np.linspace(0, self.path_length, num_points)
        
        positions = self.evaluate(u_trajectory)
        velocities = self.evaluate_derivatives(u_trajectory, order=1)
        accelerations = self.evaluate_derivatives(u_trajectory, order=2)
        curvatures = self.compute_curvature(u_trajectory)
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'curvatures': curvatures,
            'u_values': u_trajectory,
            'path_length': self.path_length
        }
    
    def get_closest_point(self, query_point: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Find the closest point on the spline to a given query point using Newton's method.
        
        Args:
            query_point: [x, y] coordinates of query point
            num_samples: Number of initial samples for coarse search (ignored, uses waypoints instead)
            
        Returns:
            Tuple of (chord_length_parameter, closest_point)
        """
        # Use waypoints for coarse search instead of dense sampling
        # This is much more efficient and leverages the fact that curvature
        # varies monotonically across spline segments
        waypoint_distances = np.linalg.norm(self.waypoints - query_point, axis=1)
        min_waypoint_idx = np.argmin(waypoint_distances)
        u_initial = self.u_values[min_waypoint_idx]
        
        # Newton's method to refine the closest point
        # We want to minimize ||P(u) - query_point||^2
        # The derivative is: 2 * (P(u) - query_point) · P'(u) = 0
        # Newton's method: u_new = u - f(u)/f'(u)
        # where f(u) = (P(u) - query_point) · P'(u)
        # and f'(u) = P'(u) · P'(u) + (P(u) - query_point) · P''(u)
        
        u = u_initial
        max_iterations = 20
        tolerance = 1e-8
        
        for i in range(max_iterations):
            # Evaluate spline and derivatives at current u
            point = self.evaluate(np.array([u]))[0]  # P(u)
            first_deriv = self.evaluate_derivatives(np.array([u]), order=1)[0]  # P'(u)
            second_deriv = self.evaluate_derivatives(np.array([u]), order=2)[0]  # P''(u)
            
            # Vector from spline point to query point
            diff = point - query_point  # P(u) - query_point
            
            # First derivative of objective function: f(u) = diff · P'(u)
            # For 2D vectors, this is diff[0]*first_deriv[0] + diff[1]*first_deriv[1]
            f = np.sum(diff * first_deriv)
            
            # Second derivative of objective function: f'(u) = P'(u) · P'(u) + diff · P''(u)
            f_prime = np.sum(first_deriv * first_deriv) + np.sum(diff * second_deriv)
            
            # Check for convergence
            if abs(f) < tolerance:
                break
                
            # Avoid division by zero
            if abs(f_prime) < 1e-12:
                break
                
            # Newton's method update
            u_new = u - f / f_prime
            
            # Clamp to valid range
            u_new = np.clip(u_new, 0, self.path_length)
            
            # Check for convergence in parameter space
            if abs(u_new - u) < tolerance:
                break
                
            u = u_new
        
        # Return final result
        closest_point = self.evaluate(np.array([u]))[0]
        return u, closest_point
    
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
        """Get total path length (actual arc length computed via RK4 integration)."""
        return self.path_length
    
    def get_chord_length(self) -> float:
        """Get total chord length."""
        return self.chord_length
    
    def get_waypoints(self) -> np.ndarray:
        """Get original waypoints."""
        return self.waypoints.copy()
    
    def get_knots(self) -> np.ndarray:
        """Get cubic spline parameter values."""
        return self.u_values.copy()
    
    def resample_trajectory(self, du: float) -> dict:
        """
        Resample trajectory with uniform chord-length parameter spacing.
        
        Args:
            du: Chord-length parameter step size
            
        Returns:
            Dictionary containing resampled trajectory
        """
        u_resampled = np.arange(0, self.path_length + du, du)
        return self.generate_trajectory_at_parameters(u_resampled)
    
    def get_tangent_angle(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Get tangent angle at given chord-length parameter values.
        
        Args:
            u: Chord-length parameter value(s)
            
        Returns:
            Tangent angles in radians
        """
        derivatives = self.evaluate_derivatives(u, order=1)
        if derivatives.ndim == 1:
            return np.arctan2(derivatives[1], derivatives[0])
        else:
            return np.arctan2(derivatives[:, 1], derivatives[:, 0])
    
    def get_parameters(self) -> np.ndarray:
        """Get the parameters of the spline in the same format as spline_dynamics.get_spline_parameters_vector()."""
        # Flip coefficient order to match CasADi format: [a, b, c, d] instead of scipy's [d, c, b, a]
        coeffs_x_flipped = np.flip(self.spline.spline_x.c, axis=0)
        coeffs_y_flipped = np.flip(self.spline.spline_y.c, axis=0)
        
        return np.concatenate([
            self.spline.spline_x.x,                    # Knots (shared by both X and Y splines)
            coeffs_x_flipped.flatten('F'),             # X coefficients flipped and flattened in Fortran order
            coeffs_y_flipped.flatten('F')              # Y coefficients flipped and flattened in Fortran order
        ])
