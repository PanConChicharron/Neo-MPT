import numpy as np
from typing import Union, Tuple, Optional
from .chord_length_parametric_spline_2d import ChordLengthParametricSpline2D


class CurvilinearCoordinates(ChordLengthParametricSpline2D):
    """
    Curvilinear coordinate system implementation using cubic splines.
    
    This class extends ChordLengthParametricSpline2D to provide curvilinear coordinate
    transformations, which are essential for path-following MPC controllers.
    
    Curvilinear coordinates (s, d) where:
    - s: arc length along the reference path
    - d: lateral deviation from the reference path
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
            num_samples: Number of samples for closest point search
            
        Returns:
            Tuple of (s, d) curvilinear coordinates
        """
        # Find closest point on the reference path
        s_closest, closest_point = self.get_closest_point(point, num_samples)
        
        # Get tangent vector at closest point
        tangent = self.evaluate_derivatives(s_closest, order=1)
        tangent_normalized = tangent / np.linalg.norm(tangent)
        
        # Get normal vector (perpendicular to tangent)
        normal = np.array([-tangent_normalized[1], tangent_normalized[0]])
        
        # Calculate lateral deviation
        deviation_vector = point - closest_point
        d = np.dot(deviation_vector, normal)
        
        return s_closest, d
    
    def curvilinear_to_cartesian(self, s: float, d: float) -> np.ndarray:
        """
        Convert curvilinear coordinates to Cartesian coordinates.
        
        Args:
            s: Arc length parameter
            d: Lateral deviation
            
        Returns:
            [x, y] coordinates in Cartesian system
        """
        # Get point on reference path
        reference_point = self.evaluate(s)
        
        # Get tangent vector at this point
        tangent = self.evaluate_derivatives(s, order=1)
        tangent_normalized = tangent / np.linalg.norm(tangent)
        
        # Get normal vector
        normal = np.array([-tangent_normalized[1], tangent_normalized[0]])
        
        # Calculate Cartesian coordinates
        cartesian_point = reference_point + d * normal
        
        return cartesian_point
    
    def get_reference_state(self, s: float) -> dict:
        """
        Get complete reference state at given arc length parameter.
        
        Args:
            s: Arc length parameter
            
        Returns:
            Dictionary containing:
            - 'position': [x, y] reference position
            - 'tangent': normalized tangent vector
            - 'normal': normalized normal vector
            - 'curvature': path curvature
            - 'heading': heading angle in radians
        """
        position = self.evaluate(s)
        tangent_raw = self.evaluate_derivatives(s, order=1)
        tangent = tangent_raw / np.linalg.norm(tangent_raw)
        normal = np.array([-tangent[1], tangent[0]])
        curvature = self.compute_curvature(s)
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
            s_values = trajectory['s_values']
            headings = []
            tangents = []
            normals = []
            
            for s in s_values:
                ref_state = self.get_reference_state(s)
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
            - 'progress': arc length progress (s)
            - 'reference_state': reference state at closest point
        """
        # Convert to curvilinear coordinates
        s, d = self.cartesian_to_curvilinear(current_position, num_samples)
        
        # Get reference state
        ref_state = self.get_reference_state(s)
        
        # Calculate heading error
        heading_error = current_heading - ref_state['heading']
        
        # Normalize heading error to [-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        return {
            'lateral_error': d,
            'heading_error': heading_error,
            'progress': s,
            'reference_state': ref_state
        }
    
    def get_local_coordinate_frame(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get local coordinate frame (Frenet frame) at given arc length.
        
        Args:
            s: Arc length parameter
            
        Returns:
            Tuple of (origin, tangent_unit, normal_unit)
        """
        origin = self.evaluate(s)
        tangent_raw = self.evaluate_derivatives(s, order=1)
        tangent_unit = tangent_raw / np.linalg.norm(tangent_raw)
        normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])
        
        return origin, tangent_unit, normal_unit
    
    def predict_reference_trajectory(self, s_start: float, prediction_horizon: float,
                                   dt: float) -> dict:
        """
        Generate reference trajectory for MPC prediction horizon.
        
        Args:
            s_start: Starting arc length parameter
            prediction_horizon: Time horizon for prediction
            dt: Time step
            
        Returns:
            Dictionary containing predicted reference trajectory
        """
        # Assume constant velocity along path for simplicity
        # In practice, this could be more sophisticated
        num_steps = int(prediction_horizon / dt) + 1
        
        # For now, assume unit velocity along path
        s_values = np.linspace(s_start, s_start + prediction_horizon, num_steps)
        
        # Clip to path bounds
        s_values = np.clip(s_values, 0, self.path_length)
        
        positions = []
        headings = []
        curvatures = []
        
        for s in s_values:
            ref_state = self.get_reference_state(s)
            positions.append(ref_state['position'])
            headings.append(ref_state['heading'])
            curvatures.append(ref_state['curvature'])
        
        return {
            's_values': s_values,
            'positions': np.array(positions),
            'headings': np.array(headings),
            'curvatures': np.array(curvatures),
            'dt': dt
        } 