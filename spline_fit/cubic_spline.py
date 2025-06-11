import numpy as np
from scipy.interpolate import CubicSpline as ScipyCubicSpline
from typing import Union, Optional


class CubicSpline:
    """
    A modular cubic spline implementation for 2D curve interpolation.
    
    This class handles the core cubic spline mathematics and provides
    methods for evaluation, derivatives, and curvature computation.
    """
    
    def __init__(self, t: np.ndarray, points: np.ndarray, bc_type: str = 'natural'):
        """
        Initialize cubic spline with parameter values and points.
        
        Args:
            t: Parameter values (e.g., arc length or time)
            points: Array of shape (N, 2) containing [x, y] coordinates
            bc_type: Boundary condition type ('natural', 'periodic', 'clamped')
        """
        self.t = np.array(t)
        self.points = np.array(points)
        self.bc_type = bc_type
        
        if len(self.t) != len(self.points):
            raise ValueError("Parameter array and points array must have same length")
        
        if self.points.shape[1] != 2:
            raise ValueError("Points must be of shape (N, 2) for [x, y] coordinates")
        
        # Create separate splines for x and y coordinates
        self.spline_x = ScipyCubicSpline(self.t, self.points[:, 0], bc_type=bc_type)
        self.spline_y = ScipyCubicSpline(self.t, self.points[:, 1], bc_type=bc_type)
        
        self.t_min = np.min(self.t)
        self.t_max = np.max(self.t)
    
    def evaluate(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the spline at given parameter values.
        
        Args:
            t: Parameter value(s) to evaluate at
            
        Returns:
            Array of [x, y] coordinates
        """
        # Store original input type
        is_scalar_input = np.isscalar(t)
        
        t = np.atleast_1d(t)
        t = np.clip(t, self.t_min, self.t_max)
        
        x = self.spline_x(t)
        y = self.spline_y(t)
        
        # Ensure x and y are arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        if is_scalar_input:
            return np.array([x[0], y[0]])
        else:
            return np.column_stack([x, y])
    
    def derivative(self, t: Union[float, np.ndarray], order: int = 1) -> np.ndarray:
        """
        Evaluate derivatives of the spline.
        
        Args:
            t: Parameter value(s) to evaluate at
            order: Derivative order (1, 2, or 3)
            
        Returns:
            Array of derivative values
        """
        if order < 1 or order > 3:
            raise ValueError("Derivative order must be 1, 2, or 3")
        
        # Store original input type
        is_scalar_input = np.isscalar(t)
        
        t = np.atleast_1d(t)
        t = np.clip(t, self.t_min, self.t_max)
        
        dx = self.spline_x.derivative(order)(t)
        dy = self.spline_y.derivative(order)(t)
        
        # Ensure dx and dy are arrays
        dx = np.atleast_1d(dx)
        dy = np.atleast_1d(dy)
        
        if is_scalar_input:
            return np.array([dx[0], dy[0]])
        else:
            return np.column_stack([dx, dy])
    
    def curvature(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute curvature at given parameter values.
        
        Args:
            t: Parameter value(s) to evaluate at
            
        Returns:
            Curvature values
        """
        # First derivatives
        first_deriv = self.derivative(t, order=1)
        if first_deriv.ndim == 1:
            dx_dt, dy_dt = first_deriv[0], first_deriv[1]
        else:
            dx_dt, dy_dt = first_deriv[:, 0], first_deriv[:, 1]
        
        # Second derivatives
        second_deriv = self.derivative(t, order=2)
        if second_deriv.ndim == 1:
            d2x_dt2, d2y_dt2 = second_deriv[0], second_deriv[1]
        else:
            d2x_dt2, d2y_dt2 = second_deriv[:, 0], second_deriv[:, 1]
        
        # Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
        denominator = (dx_dt**2 + dy_dt**2)**(3/2)
        
        # Avoid division by zero
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)
        
        return numerator / denominator
    
    def arc_length(self, t_start: float = None, t_end: float = None, 
                   num_points: int = 1000) -> float:
        """
        Compute arc length of the spline between two parameter values using RK4 integration.
        
        Args:
            t_start: Starting parameter value (default: t_min)
            t_end: Ending parameter value (default: t_max)
            num_points: Number of points for numerical integration
            
        Returns:
            Arc length
        """
        if t_start is None:
            t_start = self.t_min
        if t_end is None:
            t_end = self.t_max
        
        # RK4 integration for arc length
        dt = (t_end - t_start) / (num_points - 1)
        arc_length = 0.0
        
        for i in range(num_points - 1):
            t = t_start + i * dt
            
            # RK4 integration step
            # k1 = f(t, s) where f is the speed function ds/dt = ||dr/dt||
            k1 = np.linalg.norm(self.derivative(t, order=1))
            
            # k2 = f(t + dt/2, s + k1*dt/2)
            k2 = np.linalg.norm(self.derivative(t + dt/2, order=1))
            
            # k3 = f(t + dt/2, s + k2*dt/2)
            k3 = np.linalg.norm(self.derivative(t + dt/2, order=1))
            
            # k4 = f(t + dt, s + k3*dt)
            k4 = np.linalg.norm(self.derivative(t + dt, order=1))
            
            # RK4 formula: s_next = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            arc_length += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return arc_length
    
    def get_parameter_bounds(self) -> tuple:
        """
        Get the parameter bounds of the spline.
        
        Returns:
            Tuple of (t_min, t_max)
        """
        return self.t_min, self.t_max
    
    def is_valid_parameter(self, t: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Check if parameter value(s) are within valid bounds.
        
        Args:
            t: Parameter value(s) to check
            
        Returns:
            Boolean or array of booleans indicating validity
        """
        t = np.atleast_1d(t)
        valid = (t >= self.t_min) & (t <= self.t_max)
        
        if len(t) == 1:
            return valid[0]
        else:
            return valid 