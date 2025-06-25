import numpy as np
import casadi as ca
from typing import Dict
from Dynamics.curvilinear_dynamics import CurvilinearDynamics
from CoordinateSystem.spline_curvilinear_path import SplineCurvilinearPath


class CubicSplinePathDynamics(CurvilinearDynamics):
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
    
    def __init__(self, vehicle_model, num_waypoints: int):
        """
        Initialize spline path dynamics.
        
        Args:
            vehicle_model: Base vehicle model in Cartesian coordinates
            spline_coords: CurvilinearCoordinates object with chord-length parameterization
        """
        super().__init__(vehicle_model)
        self.num_waypoints = num_waypoints
        self.n_segments = self.num_waypoints - 1
        self.n_states = 5  # [s, u, e_y, e_ψ, v]

        # Store as CasADi parameters for symbolic computation
        self.knots_ca = ca.SX.sym('knots', self.num_waypoints)
        self.coeffs_x_ca = ca.SX.sym('coeffs_x', 4, self.n_segments)
        self.coeffs_y_ca = ca.SX.sym('coeffs_y', 4, self.n_segments)

        # Create symbolic spline functions
        self._create_symbolic_spline_functions()
        
        # Override symbolic model creation
        self._create_spline_symbolic_model()
        
        # Cache for discrete dynamics function
        self._discrete_dynamics_cache = {}
        self._current_dt = None
    
    def _create_symbolic_spline_functions(self):
        """Create symbolic functions for spline evaluation and derivatives."""
        # Symbolic parameter
        u = ca.SX.sym('u')
        
        # Create a single flattened parameter vector
        n_knots = self.num_waypoints
        
        # Extract parameters from the flattened vector
        knots = ca.SX.sym('knots', n_knots)
        coeffs_x = ca.SX.sym('coeffs_x', 4, self.n_segments)
        coeffs_y = ca.SX.sym('coeffs_y', 4, self.n_segments)

        # Single parameter vector: [knots, coeffs_x_flat, coeffs_y_flat]
        # Flatten coefficient matrices column-wise
        coeffs_x_flat = ca.reshape(coeffs_x, -1, 1)
        coeffs_y_flat = ca.reshape(coeffs_y, -1, 1)

        # Concatenate all into a single parameter vector
        params = ca.vertcat(knots, coeffs_x_flat, coeffs_y_flat)
        
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
            if i == 0:
                # For first segment, include u = knots[0]
                condition = ca.logic_and(u >= knots[i], u < knots[i+1])
            elif i == self.n_segments - 1:
                # For last segment, include u = knots[-1]
                condition = ca.logic_and(u >= knots[i], u <= knots[i+1])
            else:
                condition = ca.logic_and(u >= knots[i], u < knots[i+1])
            
            # Local parameter within segment
            t = u - knots[i]
            
            # Cubic polynomial: f(t) = a*t^3 + b*t^2 + c*t + d
            # Note: scipy stores coefficients in reverse order [d, c, b, a]
            d_x, c_x, b_x, a_x = coeffs_x[0,i], coeffs_x[1,i], coeffs_x[2,i], coeffs_x[3,i]
            d_y, c_y, b_y, a_y = coeffs_y[0,i], coeffs_y[1,i], coeffs_y[2,i], coeffs_y[3,i]
            
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
        
        # Create symbolic functions with single parameter vector
        self.spline_position = ca.Function('spline_pos', [u, params], 
                                         [ca.vertcat(x_val, y_val)])
        self.spline_tangent = ca.Function('spline_tangent', [u, params], 
                                        [ca.vertcat(dx_du, dy_du)])
        self.spline_second_deriv = ca.Function('spline_second_deriv', [u, params], 
                                             [ca.vertcat(d2x_du2, d2y_du2)])
        
        # Tangent magnitude: |dx/du|
        tangent_mag = ca.sqrt(dx_du**2 + dy_du**2)
        self.spline_tangent_magnitude = ca.Function('spline_tangent_mag', 
                                                  [u, params], 
                                                  [tangent_mag])
        
        # Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx_du * d2y_du2 - dy_du * d2x_du2
        denominator = (dx_du**2 + dy_du**2)**(3/2)
        # Add small epsilon to avoid division by zero
        curvature = numerator / (denominator + 1e-12)
        self.spline_curvature = ca.Function('spline_curvature', 
                                          [u, params], 
                                          [curvature])
    
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
        
        # Create the flattened parameter vector for symbolic computation
        n_knots = self.knots_ca.shape[0]

        n_coeffs_x = 4 * self.n_segments
        n_coeffs_y = 4 * self.n_segments
        total_params = n_knots + n_coeffs_x + n_coeffs_y
        
        self.parameters = ca.SX.sym('params', total_params)
        
        # Get spline properties symbolically using the chord-length parameter u
        self.kappa = self.spline_curvature(self.u, self.parameters)
        self.ds_du = self.spline_tangent_magnitude(self.u, self.parameters)
        
        # Vehicle slip angle (from bicycle model)
        lf = self.vehicle_model.wheelbase_front
        lr = self.vehicle_model.wheelbase_rear
        L = lf + lr  # total wheelbase
        
        beta = ca.atan(lr * ca.tan(self.delta) / L)
        
        # Corrected curvilinear dynamics equations:
        # 
        # Key insight: Standard curvilinear coordinates use arc-length (s) as the primary parameter.
        # We should evolve s using standard curvilinear dynamics, then compute u from the 
        # inverse relationship.
        #
        # Standard curvilinear dynamics (with s as primary parameter):
        # ds/dt = v * cos(e_ψ + β) / (1 - κ*e_y)  [standard formulation]
        # de_y/dt = v * sin(e_ψ + β)  
        # de_ψ/dt = v * sin(β) / L - κ * ds/dt
        # dv/dt = a
        #
        # Then compute u from the constraint: u = f^(-1)(s) where s = ∫₀ᵘ |dP/du| du
        # For dynamics: du/dt = (du/ds) * (ds/dt) = (1 / |dP/du|) * (ds/dt)
        
        # Denominator term for dynamics
        denom = 1 - self.kappa * self.e_y
        
        # State derivatives - standard curvilinear formulation
        s_dot = self.v * ca.cos(self.e_ψ + beta) / denom  # Standard curvilinear dynamics
        
        # Compute the raw u_dot
        u_dot = s_dot / self.ds_du  # Convert arc-length rate to chord-length parameter rate
        
        # Note: State bounds (u_min, u_max) are handled as proper constraints in the MPC formulation,
        # not as saturations in the dynamics equations. The dynamics should remain unsaturated.
        
        e_y_dot = self.v * ca.sin(self.e_ψ + beta)
        e_ψ_dot = (self.v * ca.sin(beta) / L) - (self.kappa * s_dot)
        v_dot = self.a
        
        self.dynamics = ca.vertcat(s_dot, u_dot, e_y_dot, e_ψ_dot, v_dot)

        # Create function for dynamics evaluation with parameter vector
        self.dynamics_func = ca.Function('spline_path_dynamics',
                                       [self.state, self.input, self.parameters],
                                       [self.dynamics])
    
    def get_discrete_dynamics(self, dt: float) -> ca.Function:
        """
        Get discrete-time spline path dynamics using RK4 integration (parametric in parameters).
        Args:
            dt: Time step for discretization
        Returns:
            CasADi function for discrete spline path dynamics (parametric in parameters)
        """
        if dt == self._current_dt and dt in self._discrete_dynamics_cache:
            return self._discrete_dynamics_cache[dt]
        # Use the symbolic expression for RK4
        x = self.state
        u = self.input
        p = self.parameters
        f = self.dynamics  # This is an SX expression
        
        k1 = ca.substitute([f], [x, u, p], [x, u, p])[0]
        k2 = ca.substitute([f], [x, u, p], [x + dt/2 * k1, u, p])[0]
        k3 = ca.substitute([f], [x, u, p], [x + dt/2 * k2, u, p])[0]
        k4 = ca.substitute([f], [x, u, p], [x + dt * k3, u, p])[0]
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        discrete_dynamics = ca.Function('discrete_spline_path_dynamics', [x, u, p], [x_next])
        self._discrete_dynamics_cache[dt] = discrete_dynamics
        self._current_dt = dt
        return discrete_dynamics
    
    def get_continuous_dynamics(self) -> ca.Function:
        """
        Get continuous-time spline path dynamics function.
        
        Returns:
            CasADi function for continuous spline path dynamics (state derivatives)
        """
        return self.dynamics_func
    
    def simulate_step(self, spline_curvilinear_state: np.ndarray, 
                     input: np.ndarray, parameters: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate one step of spline path dynamics (parametric in parameters).
        Args:
            spline_curvilinear_state: Current state [s, u, e_y, e_ψ, v]
            input: Control input [delta, a]
            parameters: Spline parameter vector (knots, coeffs_x, coeffs_y)
            dt: Time step
        Returns:
            Next spline curvilinear state
        """
        discrete_dynamics = self.get_discrete_dynamics(dt)
        result = discrete_dynamics(spline_curvilinear_state, input, parameters)
        if hasattr(result, 'full'):
            return np.array(result.full()).flatten()
        return np.array(result).flatten()
    
    def get_state_names(self) -> list:
        """Get names of spline curvilinear state variables."""
        return ['s', 'u', 'e_y', 'e_ψ', 'v']
    