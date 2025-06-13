import numpy as np
from scipy.interpolate import CubicSpline

class SplineFitter:
    """
    Fits cubic splines to waypoints using chord-length parameterization.
    """
    
    def __init__(self):
        """Initialize spline fitter."""
        self.spline_x = None
        self.spline_y = None
        self.u_values = None  # Chord-length parameters
    
    def fit_spline(self, waypoints: np.ndarray) -> 'SplineFitter':
        """
        Fit cubic splines to waypoints using chord-length parameterization.
        
        Args:
            waypoints: Array of shape (N, 2) containing [x, y] coordinates
            
        Returns:
            self for method chaining
        """
        if waypoints.shape[1] != 2:
            raise ValueError("Waypoints must be of shape (N, 2) for [x, y] coordinates")
        
        # Calculate chord lengths between consecutive waypoints
        ds = np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))
        
        # Create chord-length parameter values
        self.u_values = np.concatenate(([0], np.cumsum(ds)))
        
        # Fit separate cubic splines for x and y coordinates
        self.spline_x = CubicSpline(self.u_values, waypoints[:, 0], bc_type='natural')
        self.spline_y = CubicSpline(self.u_values, waypoints[:, 1], bc_type='natural')
        
        return self
    
    def evaluate(self, u: float) -> np.ndarray:
        """
        Evaluate the spline at a given chord-length parameter.
        
        Args:
            u: Chord-length parameter value
            
        Returns:
            [x, y] coordinates at parameter u
        """
        if self.spline_x is None or self.spline_y is None:
            raise ValueError("Spline not fitted. Call fit_spline() first.")
        
        x = self.spline_x(u)
        y = self.spline_y(u)
        return np.array([x, y])
    
    def evaluate_derivatives(self, u: float) -> np.ndarray:
        """
        Evaluate the first derivatives of the spline.
        
        Args:
            u: Chord-length parameter value
            
        Returns:
            [dx/du, dy/du] derivatives at parameter u
        """
        if self.spline_x is None or self.spline_y is None:
            raise ValueError("Spline not fitted. Call fit_spline() first.")
        
        dx_du = self.spline_x(u, 1)
        dy_du = self.spline_y(u, 1)
        return np.array([dx_du, dy_du])
    
    def evaluate_second_derivatives(self, u: float) -> np.ndarray:
        """
        Evaluate the second derivatives of the spline.
        
        Args:
            u: Chord-length parameter value
            
        Returns:
            [d²x/du², d²y/du²] second derivatives at parameter u
        """
        if self.spline_x is None or self.spline_y is None:
            raise ValueError("Spline not fitted. Call fit_spline() first.")
        
        d2x_du2 = self.spline_x(u, 2)
        d2y_du2 = self.spline_y(u, 2)
        return np.array([d2x_du2, d2y_du2])
    
    def compute_curvature(self, u: float) -> float:
        """
        Compute curvature at given chord-length parameter.
        
        Args:
            u: Chord-length parameter value
            
        Returns:
            Curvature κ at parameter u
        """
        if self.spline_x is None or self.spline_y is None:
            raise ValueError("Spline not fitted. Call fit_spline() first.")
        
        # Get first and second derivatives
        dx_du = self.spline_x(u, 1)
        dy_du = self.spline_y(u, 1)
        d2x_du2 = self.spline_x(u, 2)
        d2y_du2 = self.spline_y(u, 2)
        
        # Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx_du * d2y_du2 - dy_du * d2x_du2
        denominator = (dx_du**2 + dy_du**2)**(3/2)
        
        # Avoid division by zero
        if abs(denominator) < 1e-12:
            return 0.0
        
        return numerator / denominator
    
    def get_path_length(self) -> float:
        """
        Get total path length (chord length approximation).
        
        Returns:
            Total chord length of the path
        """
        if self.u_values is None:
            raise ValueError("Spline not fitted. Call fit_spline() first.")
        
        return self.u_values[-1] 