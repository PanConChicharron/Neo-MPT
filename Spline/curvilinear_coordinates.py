import numpy as np
from typing import Tuple
from .chord_length_parametric_spline_2d import ChordLengthParametricSpline2D


class CurvilinearCoordinates(ChordLengthParametricSpline2D):
    """
    Curvilinear coordinate system implementation using cubic splines.
    
    This class extends ChordLengthParametricSpline2D to provide curvilinear coordinate
    transformations, which are essential for path-following MPC controllers.
    
    Curvilinear coordinates (s, d) where:
    - s: arc length along the reference path
    - d: lateral deviation from the reference path
    
    Note: The underlying spline uses chord-length parameter 'u', but this class
    provides methods to work with both 'u' (chord-length) and 's' (arc-length).
    """
    
    def __init__(self, waypoints: np.ndarray, closed_path: bool = False):
        """
        Initialize curvilinear coordinate system with waypoints.
        
        Args:
            waypoints: Array of shape (N, 2) containing [x, y] coordinates
            closed_path: Whether the path should be closed
        """
        super().__init__(waypoints, closed_path)
    
    def cartesian_to_curvilinear(self, point: np.ndarray, 
                                num_samples: int = 1000) -> Tuple[float, float]:
        """
        Convert Cartesian coordinates to curvilinear coordinates.
        
        Args:
            point: [x, y] coordinates in Cartesian system
            num_samples: Number of samples for closest point search (unused, kept for compatibility)
            
        Returns:
            Tuple of (u, d) where u is chord-length parameter and d is lateral distance
            Note: Returns chord-length parameter 'u', not arc-length 's'
        """
        # Use Newton's method to find closest point on spline
        u_guess = 0.0
        max_iter = 50
        tol = 1e-6
        
        for _ in range(max_iter):
            # Get spline point and derivatives at current guess
            spline_point = self.evaluate(u_guess)
            tangent = self.evaluate_derivatives(u_guess, order=1)
            
            # Compute error vector and its derivative
            error = point - spline_point
            
            # Update parameter
            u_new = u_guess + np.dot(error, tangent) / np.dot(tangent, tangent)
            
            # Check convergence
            if abs(u_new - u_guess) < tol:
                break
                
            u_guess = u_new
        
        # Compute lateral distance
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / np.linalg.norm(normal)
        d = np.dot(point - spline_point, normal)
        
        return u_guess, d
    
    def curvilinear_to_cartesian(self, u: float, d: float) -> np.ndarray:
        """
        Convert curvilinear coordinates to Cartesian coordinates.
        
        Args:
            u: Chord-length parameter
            d: Lateral deviation
            
        Returns:
            [x, y] coordinates in Cartesian system
        """
        # Get spline point and tangent
        spline_point = self.evaluate(u)
        tangent = self.evaluate_derivatives(u, order=1)
        
        # Compute normal vector
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Offset point by lateral distance
        return spline_point + d * normal
    
    def get_reference_state(self, u: float) -> dict:
        """
        Get complete reference state at given chord-length parameter.
        
        Args:
            u: Chord-length parameter
            
        Returns:
            Dictionary containing:
            - 'position': [x, y] reference position
            - 'tangent': normalized tangent vector
            - 'normal': normalized normal vector
            - 'curvature': path curvature
            - 'heading': heading angle in radians
        """
        position = self.evaluate(u)
        tangent_raw = self.evaluate_derivatives(u, order=1)
        tangent = tangent_raw / np.linalg.norm(tangent_raw)
        normal = np.array([-tangent[1], tangent[0]])
        curvature = self.compute_curvature(u)
        heading = np.arctan2(tangent[1], tangent[0])
        
        return {
            'position': position,
            'tangent': tangent,
            'normal': normal,
            'curvature': curvature,
            'heading': heading
        }
    
    def generate_reference_trajectory(self, num_points: int = 100, 
                                    include_curvilinear: bool = True) -> dict:
        """
        Generate reference trajectory with curvilinear information.
        
        Args:
            num_points: Number of points in trajectory
            include_curvilinear: Whether to include curvilinear coordinate info
            
        Returns:
            Dictionary containing trajectory data with curvilinear information
        """
        trajectory = self.generate_trajectory(num_points)
        
        if include_curvilinear:
            u_values = trajectory['u_values']
            headings = []
            tangents = []
            normals = []
            
            for u in u_values:
                ref_state = self.get_reference_state(u)
                headings.append(ref_state['heading'])
                tangents.append(ref_state['tangent'])
                normals.append(ref_state['normal'])
            
            trajectory['headings'] = np.array(headings)
            trajectory['tangents'] = np.array(tangents)
            trajectory['normals'] = np.array(normals)
        
        return trajectory
    
    def compute_tracking_error(self, current_position: np.ndarray, 
                             current_heading: float,
                             num_samples: int = 1000) -> dict:
        """
        Compute tracking errors in curvilinear coordinates.
        
        Args:
            current_position: [x, y] current position
            current_heading: current heading angle in radians
            num_samples: Number of samples for closest point search
            
        Returns:
            Dictionary containing tracking errors:
            - 'lateral_error': lateral deviation (d)
            - 'heading_error': heading error in radians
            - 'progress': chord-length parameter (u)
            - 'reference_state': reference state at closest point
        """
        # Convert to curvilinear coordinates
        u, d = self.cartesian_to_curvilinear(current_position, num_samples)
        
        # Get reference state
        ref_state = self.get_reference_state(u)
        
        # Calculate heading error
        heading_error = current_heading - ref_state['heading']
        
        # Normalize heading error to [-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        return {
            'lateral_error': d,
            'heading_error': heading_error,
            'progress': u,  # Note: This is chord-length parameter, not arc-length
            'reference_state': ref_state
        }
    
    def get_local_coordinate_frame(self, u: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get local coordinate frame (Frenet frame) at given chord-length parameter.
        
        Args:
            u: Chord-length parameter
            
        Returns:
            Tuple of (origin, tangent_unit, normal_unit)
        """
        origin = self.evaluate(u)
        tangent_raw = self.evaluate_derivatives(u, order=1)
        tangent_unit = tangent_raw / np.linalg.norm(tangent_raw)
        normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])
        
        return origin, tangent_unit, normal_unit
    
    def predict_reference_trajectory(self, u_start: float, prediction_horizon: float,
                                   dt: float) -> dict:
        """
        Generate reference trajectory for MPC prediction horizon.
        
        Args:
            u_start: Starting chord-length parameter
            prediction_horizon: Time horizon for prediction
            dt: Time step
            
        Returns:
            Dictionary containing predicted reference trajectory
        """
        # Assume constant velocity along path for simplicity
        # In practice, this could be more sophisticated
        num_steps = int(prediction_horizon / dt) + 1
        
        # For now, assume unit velocity along path
        u_values = np.linspace(u_start, u_start + prediction_horizon, num_steps)
        
        # Clip to path bounds
        u_values = np.clip(u_values, 0, self.path_length)
        
        positions = []
        headings = []
        curvatures = []
        
        for u in u_values:
            ref_state = self.get_reference_state(u)
            positions.append(ref_state['position'])
            headings.append(ref_state['heading'])
            curvatures.append(ref_state['curvature'])
        
        return {
            'u_values': u_values,  # Chord-length parameters
            'positions': np.array(positions),
            'headings': np.array(headings),
            'curvatures': np.array(curvatures),
            'dt': dt
        }

    def get_heading(self, u: float) -> float:
        """
        Get path heading at given chord-length parameter.
        
        Args:
            u: chord-length parameter along path
            
        Returns:
            Heading angle in radians
        """
        tangent = self.evaluate_derivatives(u)
        return np.arctan2(tangent[1], tangent[0])
    
    def get_curvature(self, u: float) -> float:
        """
        Get path curvature at given chord-length parameter.
        
        Args:
            u: chord-length parameter along path
            
        Returns:
            Curvature κ
        """
        return self.compute_curvature(u)

    @property
    def total_chord_length(self) -> float:
        """Get total chord length of the path."""
        return self.u_values[-1]
    
    # Arc-length related methods (for when we need actual arc-length 's')
    
    def chord_to_arc_length(self, u: float) -> float:
        """
        Convert chord-length parameter to actual arc-length using integration.
        
        Args:
            u: Chord-length parameter
            
        Returns:
            Actual arc-length from start of path to parameter u
        """
        # Clamp u to valid range
        u = np.clip(u, 0, self.u_values[-1])
        
        # If u is 0, arc length is 0
        if u <= 0:
            return 0.0
        
        # If u is at the end, return total arc length
        if u >= self.u_values[-1]:
            return self.path_length
        
        # Use the spline's arc_length method to compute from 0 to u
        return self.spline.arc_length(t_start=0, t_end=u)
    
    def arc_to_chord_length(self, s: float) -> float:
        """
        Convert arc-length to chord-length parameter using numerical inversion.
        
        Args:
            s: Arc-length
            
        Returns:
            Chord-length parameter u corresponding to arc-length s
        """
        # Clamp s to valid range
        s = np.clip(s, 0, self.path_length)
        
        # If s is 0, u is 0
        if s <= 0:
            return 0.0
        
        # If s is at the end, return maximum u
        if s >= self.path_length:
            return self.u_values[-1]
        
        # Use binary search to find u such that chord_to_arc_length(u) ≈ s
        u_min = 0.0
        u_max = self.u_values[-1]
        tolerance = 1e-6
        max_iterations = 50
        
        for _ in range(max_iterations):
            u_mid = (u_min + u_max) / 2
            s_mid = self.chord_to_arc_length(u_mid)
            
            if abs(s_mid - s) < tolerance:
                return u_mid
            
            if s_mid < s:
                u_min = u_mid
            else:
                u_max = u_mid
        
        # Return best approximation
        return (u_min + u_max) / 2
    
    def curvilinear_to_cartesian_arc(self, s: float, d: float) -> np.ndarray:
        """
        Convert curvilinear coordinates (with arc-length) to Cartesian coordinates.
        
        Args:
            s: Arc-length parameter
            d: Lateral deviation
            
        Returns:
            [x, y] coordinates in Cartesian system
        """
        u = self.arc_to_chord_length(s)
        return self.curvilinear_to_cartesian(u, d)
    
    def cartesian_to_curvilinear_arc(self, point: np.ndarray) -> Tuple[float, float]:
        """
        Convert Cartesian coordinates to curvilinear coordinates with arc-length.
        
        Args:
            point: [x, y] coordinates in Cartesian system
            
        Returns:
            Tuple of (s, d) where s is arc-length and d is lateral distance
        """
        u, d = self.cartesian_to_curvilinear(point)
        s = self.chord_to_arc_length(u)
        return s, d 