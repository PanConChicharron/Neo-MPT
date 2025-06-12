import numpy as np
import casadi as ca
from typing import Dict, Optional, Callable
from .curvilinear_dynamics import CurvilinearDynamics
from ..spline_fit.curvilinear_coordinates import CurvilinearCoordinates


class SplinePathDynamics(CurvilinearDynamics):
    """
    Extended curvilinear dynamics for chord-length parameterized splines.
    
    This class extends CurvilinearDynamics to handle the mapping between:
    - s: actual arc-length along the path
    - u: chord-length parameter used by the spline
    
    The state becomes (s, u, e_y, e_ψ, v) where:
    - s: arc-length progress along path
    - u: spline parameter (chord-length)
    - e_y: lateral error from reference path
    - e_ψ: heading error relative to path tangent
    - v: velocity magnitude
    
    Uses fully symbolic spline representation for complete differentiability.
    """
    
    def __init__(self, vehicle_model, spline_coords: CurvilinearCoordinates):
        """
        Initialize spline path dynamics.
        
        Args:
            vehicle_model: Base vehicle model in Cartesian coordinates
            spline_coords: CurvilinearCoordinates object with chord-length parameterization
        """
        super().__init__(vehicle_model)
        self.spline_coords = spline_coords
        self.n_states = 5  # [s, u, e_y, e_ψ, v]
        
        # Extract spline coefficients for symbolic representation
        self._extract_spline_coefficients()
        
        # Create symbolic spline functions
        self._create_symbolic_spline_functions()
        
        # Override symbolic model creation
        self._create_spline_symbolic_model()
    
    def _extract_spline_coefficients(self):
        """Extract cubic spline coefficients from scipy splines for symbolic representation."""
        # Get the underlying scipy splines
        spline_x = self.spline_coords.spline.spline_x
        spline_y = self.spline_coords.spline.spline_y
        
        # Extract knot points and coefficients
        self.knots = spline_x.x  # Parameter values (same for both x and y)
        self.n_segments = len(self.knots) - 1
        
        # Coefficients for each segment: [a, b, c, d] where spline = a*t^3 + b*t^2 + c*t + d
        # scipy stores coefficients in reverse order: [d, c, b, a]
        self.coeffs_x = np.flip(spline_x.c, axis=0)  # Shape: (4, n_segments)
        self.coeffs_y = np.flip(spline_y.c, axis=0)  # Shape: (4, n_segments)
        
        # Store as CasADi parameters for symbolic computation
        self.knots_ca = ca.DM(self.knots)
        self.coeffs_x_ca = ca.DM(self.coeffs_x)
        self.coeffs_y_ca = ca.DM(self.coeffs_y)
    
    def _create_symbolic_spline_functions(self):
        """Create symbolic functions for spline evaluation and derivatives."""
        # Symbolic parameter
        u = ca.SX.sym('u')
        
        # Find which segment the parameter belongs to
        # For symbolic computation, we'll use a piecewise approach
        
        # Initialize outputs
        x_val = ca.SX.zeros(1)
        y_val = ca.SX.zeros(1)
        dx_du = ca.SX.zeros(1)
        dy_du = ca.SX.zeros(1)
        d2x_du2 = ca.SX.zeros(1)
        d2y_du2 = ca.SX.zeros(1)
        
        # Create piecewise symbolic spline evaluation
        for i in range(self.n_segments):
            # Condition: knots[i] <= u < knots[i+1] (or u <= knots[i+1] for last segment)
            if i == self.n_segments - 1:
                condition = ca.logic_and(u >= self.knots_ca[i], u <= self.knots_ca[i+1])
            else:
                condition = ca.logic_and(u >= self.knots_ca[i], u < self.knots_ca[i+1])
            
            # Local parameter within segment
            t = u - self.knots_ca[i]
            
            # Cubic polynomial: f(t) = a*t^3 + b*t^2 + c*t + d
            a_x, b_x, c_x, d_x = self.coeffs_x_ca[0,i], self.coeffs_x_ca[1,i], self.coeffs_x_ca[2,i], self.coeffs_x_ca[3,i]
            a_y, b_y, c_y, d_y = self.coeffs_y_ca[0,i], self.coeffs_y_ca[1,i], self.coeffs_y_ca[2,i], self.coeffs_y_ca[3,i]
            
            # Position
            x_seg = a_x*t**3 + b_x*t**2 + c_x*t + d_x
            y_seg = a_y*t**3 + b_y*t**2 + c_y*t + d_y
            
            # First derivatives: f'(t) = 3*a*t^2 + 2*b*t + c
            dx_seg = 3*a_x*t**2 + 2*b_x*t + c_x
            dy_seg = 3*a_y*t**2 + 2*b_y*t + c_y
            
            # Second derivatives: f''(t) = 6*a*t + 2*b
            d2x_seg = 6*a_x*t + 2*b_x
            d2y_seg = 6*a_y*t + 2*b_y
            
            # Use conditional assignment
            x_val = ca.if_else(condition, x_seg, x_val)
            y_val = ca.if_else(condition, y_seg, y_val)
            dx_du = ca.if_else(condition, dx_seg, dx_du)
            dy_du = ca.if_else(condition, dy_seg, dy_du)
            d2x_du2 = ca.if_else(condition, d2x_seg, d2x_du2)
            d2y_du2 = ca.if_else(condition, d2y_seg, d2y_du2)
        
        # Create symbolic functions
        self.spline_position = ca.Function('spline_pos', [u], [ca.vertcat(x_val, y_val)])
        self.spline_tangent = ca.Function('spline_tangent', [u], [ca.vertcat(dx_du, dy_du)])
        self.spline_second_deriv = ca.Function('spline_second_deriv', [u], [ca.vertcat(d2x_du2, d2y_du2)])
        
        # Tangent magnitude: |dx/du|
        tangent_mag = ca.sqrt(dx_du**2 + dy_du**2)
        self.spline_tangent_magnitude = ca.Function('spline_tangent_mag', [u], [tangent_mag])
        
        # Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx_du * d2y_du2 - dy_du * d2x_du2
        denominator = (dx_du**2 + dy_du**2)**(3/2)
        # Add small epsilon to avoid division by zero
        curvature = numerator / (denominator + 1e-12)
        self.spline_curvature = ca.Function('spline_curvature', [u], [curvature])
    
    def _create_spline_symbolic_model(self):
        """Create symbolic model with spline parameter mapping."""
        # Extended curvilinear state variables
        self.s = ca.SX.sym('s')           # arc length along path
        self.u = ca.SX.sym('u')           # spline parameter (chord-length)
        self.e_y = ca.SX.sym('e_y')       # lateral error
        self.e_ψ = ca.SX.sym('e_ψ')       # heading error
        self.v = ca.SX.sym('v')           # velocity
        
        self.state = ca.vertcat(self.s, self.u, self.e_y, self.e_ψ, self.v)
        
        # Input variables (same as vehicle model)
        self.delta = ca.SX.sym('delta')   # steering angle
        self.a = ca.SX.sym('a')           # acceleration
        
        self.input = ca.vertcat(self.delta, self.a)
        
        # Get spline properties symbolically
        self.kappa = self.spline_curvature(self.u)  # path curvature at current u
        self.ds_du = self.spline_tangent_magnitude(self.u)  # tangent magnitude |dx/du|
        
        # Vehicle slip angle (from bicycle model)
        lf = self.vehicle_model.wheelbase_front
        lr = self.vehicle_model.wheelbase_rear
        L = lf + lr  # total wheelbase
        
        beta = ca.atan(lr * ca.tan(self.delta) / L)
        
        # Extended curvilinear dynamics equations:
        # ds/dt = v * cos(e_ψ + β) / (1 - κ*e_y)
        # du/dt = (ds/dt) / (ds/du)
        # de_y/dt = v * sin(e_ψ + β)  
        # de_ψ/dt = v * sin(β) / L - κ * ds/dt
        # dv/dt = a
        
        # Denominator term for s dynamics
        denom = 1 - self.kappa * self.e_y
        
        # State derivatives
        s_dot = self.v * ca.cos(self.e_ψ + beta) / denom
        u_dot = s_dot / self.ds_du  # du/dt = (ds/dt) / (ds/du)
        e_y_dot = self.v * ca.sin(self.e_ψ + beta)
        e_ψ_dot = (self.v * ca.sin(beta) / L) - (self.kappa * s_dot)
        v_dot = self.a
        
        self.dynamics = ca.vertcat(s_dot, u_dot, e_y_dot, e_ψ_dot, v_dot)
        
        # Create function for dynamics evaluation
        self.dynamics_func = ca.Function('spline_path_dynamics',
                                       [self.state, self.input],
                                       [self.dynamics])
    
    def get_discrete_dynamics(self, dt: float) -> ca.Function:
        """
        Get discrete-time spline path dynamics using RK4 integration.
        
        Args:
            dt: Time step for discretization
            
        Returns:
            CasADi function for discrete spline path dynamics
        """
        # RK4 integration - now fully symbolic
        k1 = self.dynamics
        k2 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k1)
        k3 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k2)
        k4 = ca.substitute(self.dynamics, self.state, self.state + dt * k3)
        
        state_next = self.state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return ca.Function('discrete_spline_path_dynamics',
                          [self.state, self.input],
                          [state_next])
    
    def compute_tangent_magnitude(self, u: float) -> float:
        """
        Compute ds/du: magnitude of tangent vector |dx/du|.
        
        For a parametric curve x(u), the arc-length differential is:
        ds/du = |dx/du| = ||tangent||
        
        Note: The spline uses 's' as its parameter name, but this is actually
        the chord-length parameter, which we call 'u' in our dynamics.
        
        Args:
            u: Current spline parameter (chord-length)
            
        Returns:
            ds/du: magnitude of tangent vector
        """
        # Use symbolic function for numerical evaluation
        return float(self.spline_tangent_magnitude(u))
    
    def get_spline_state_at_progress(self, s: float) -> Dict:
        """
        Get spline state (u, curvature, ds/du) at given arc-length progress.
        
        Args:
            s: Arc-length progress along path
            
        Returns:
            Dictionary containing spline parameter info
        """
        # For chord-length parameterized splines, we need to find u such that
        # the arc-length from start to u equals s
        
        # This is an approximation - in practice you might want a lookup table
        # or more sophisticated mapping
        # Note: spline.s_values contains the chord-length parameters
        u_approx = s / self.spline_coords.path_length * self.spline_coords.s_values[-1]
        u_approx = np.clip(u_approx, self.spline_coords.s_values[0], self.spline_coords.s_values[-1])
        
        # Get curvature at this spline parameter using symbolic function
        curvature = float(self.spline_curvature(u_approx))
        
        # Get tangent magnitude using symbolic function
        ds_du = float(self.spline_tangent_magnitude(u_approx))
        
        return {
            'u': u_approx,
            'curvature': curvature,
            'ds_du': ds_du
        }
    
    def cartesian_to_spline_curvilinear(self, cartesian_state: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian state to spline curvilinear coordinates.
        
        Args:
            cartesian_state: [x, y, theta, v] in Cartesian coordinates
            
        Returns:
            [s, u, e_y, e_ψ, v] in spline curvilinear coordinates
        """
        x, y, theta, v = cartesian_state
        
        # Convert position to curvilinear coordinates (this gives us u and e_y)
        # Note: spline returns (s_param, d) where s_param is chord-length parameter
        u, e_y = self.spline_coords.cartesian_to_curvilinear(np.array([x, y]))
        
        # Approximate arc-length from spline parameter
        # This is a simplification - you might want a more accurate mapping
        s_approx = u / self.spline_coords.s_values[-1] * self.spline_coords.path_length
        
        # Get reference heading at this spline parameter using symbolic function
        tangent = self.spline_tangent(u)
        ref_heading = float(ca.atan2(tangent[1], tangent[0]))
        
        # Calculate heading error
        e_ψ = theta - ref_heading
        e_ψ = np.arctan2(np.sin(e_ψ), np.cos(e_ψ))  # Normalize to [-π, π]
        
        return np.array([s_approx, u, e_y, e_ψ, v])
    
    def spline_curvilinear_to_cartesian(self, spline_curvilinear_state: np.ndarray) -> np.ndarray:
        """
        Convert spline curvilinear state to Cartesian coordinates.
        
        Args:
            spline_curvilinear_state: [s, u, e_y, e_ψ, v] in spline curvilinear coordinates
            
        Returns:
            [x, y, theta, v] in Cartesian coordinates
        """
        s, u, e_y, e_ψ, v = spline_curvilinear_state
        
        # Get reference position using symbolic function
        ref_pos = self.spline_position(u)
        
        # Get reference heading using symbolic function
        tangent = self.spline_tangent(u)
        ref_heading = float(ca.atan2(tangent[1], tangent[0]))
        
        # Get normal vector for lateral offset
        tangent_norm = float(ca.sqrt(tangent[0]**2 + tangent[1]**2))
        normal = np.array([-float(tangent[1])/tangent_norm, float(tangent[0])/tangent_norm])
        
        # Convert to Cartesian position
        x = float(ref_pos[0]) + e_y * normal[0]
        y = float(ref_pos[1]) + e_y * normal[1]
        
        # Calculate absolute heading
        theta = ref_heading + e_ψ
        
        return np.array([x, y, theta, v])
    
    def simulate_step(self, spline_curvilinear_state: np.ndarray, 
                     input: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate one step of spline path dynamics.
        
        Args:
            spline_curvilinear_state: Current state [s, u, e_y, e_ψ, v]
            input: Control input [delta, a]
            dt: Time step
            
        Returns:
            Next spline curvilinear state
        """
        # Simulate dynamics using symbolic functions
        discrete_dynamics = self.get_discrete_dynamics(dt)
        return np.array(discrete_dynamics(spline_curvilinear_state, input)).flatten()
    
    def get_state_names(self) -> list:
        """Get names of spline curvilinear state variables."""
        return ['s', 'u', 'e_y', 'e_ψ', 'v']
    
    def get_constraints(self) -> Dict:
        """Get constraints (same as base vehicle model)."""
        constraints = self.vehicle_model.get_constraints()
        
        # Add spline parameter constraints
        constraints['u_min'] = self.spline_coords.s_values[0]
        constraints['u_max'] = self.spline_coords.s_values[-1]
        
        return constraints 