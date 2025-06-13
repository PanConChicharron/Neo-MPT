import numpy as np
import casadi as ca
from typing import Dict, Optional, Callable, Tuple
from .curvilinear_dynamics import CurvilinearDynamics
from spline_fit.curvilinear_coordinates import CurvilinearCoordinates


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
        
        # Cache for discrete dynamics function
        self._discrete_dynamics_cache = {}
        self._current_dt = None
    
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
        self.knots_ca = ca.SX.sym('knots', self.knots.shape[0])
        self.coeffs_x_ca = ca.SX.sym('coeffs_x', self.coeffs_x.shape[0], self.coeffs_x.shape[1])
        self.coeffs_y_ca = ca.SX.sym('coeffs_y', self.coeffs_y.shape[0], self.coeffs_y.shape[1])
        
        # Store numerical values for parameter updates
        self.knots_np = self.knots
        self.coeffs_x_np = self.coeffs_x
        self.coeffs_y_np = self.coeffs_y
    
    def _create_symbolic_spline_functions(self):
        """Create symbolic functions for spline evaluation and derivatives."""
        # Symbolic parameter
        u = ca.SX.sym('u')
        
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
                condition = ca.logic_and(u >= self.knots_ca[i], u < self.knots_ca[i+1])
            elif i == self.n_segments - 1:
                # For last segment, include u = knots[-1]
                condition = ca.logic_and(u >= self.knots_ca[i], u <= self.knots_ca[i+1])
            else:
                condition = ca.logic_and(u >= self.knots_ca[i], u < self.knots_ca[i+1])
            
            # Local parameter within segment
            t = u - self.knots_ca[i]
            
            # Cubic polynomial: f(t) = a*t^3 + b*t^2 + c*t + d
            # Note: scipy stores coefficients in reverse order [d, c, b, a]
            d_x, c_x, b_x, a_x = self.coeffs_x_ca[0,i], self.coeffs_x_ca[1,i], self.coeffs_x_ca[2,i], self.coeffs_x_ca[3,i]
            d_y, c_y, b_y, a_y = self.coeffs_y_ca[0,i], self.coeffs_y_ca[1,i], self.coeffs_y_ca[2,i], self.coeffs_y_ca[3,i]
            
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
        self.spline_position = ca.Function('spline_pos', [u, self.knots_ca, self.coeffs_x_ca, self.coeffs_y_ca], 
                                         [ca.vertcat(x_val, y_val)])
        self.spline_tangent = ca.Function('spline_tangent', [u, self.knots_ca, self.coeffs_x_ca, self.coeffs_y_ca], 
                                        [ca.vertcat(dx_du, dy_du)])
        self.spline_second_deriv = ca.Function('spline_second_deriv', [u, self.knots_ca, self.coeffs_x_ca, self.coeffs_y_ca], 
                                             [ca.vertcat(d2x_du2, d2y_du2)])
        
        # Tangent magnitude: |dx/du|
        tangent_mag = ca.sqrt(dx_du**2 + dy_du**2)
        self.spline_tangent_magnitude = ca.Function('spline_tangent_mag', 
                                                  [u, self.knots_ca, self.coeffs_x_ca, self.coeffs_y_ca], 
                                                  [tangent_mag])
        
        # Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = dx_du * d2y_du2 - dy_du * d2x_du2
        denominator = (dx_du**2 + dy_du**2)**(3/2)
        # Add small epsilon to avoid division by zero
        curvature = numerator / (denominator + 1e-12)
        self.spline_curvature = ca.Function('spline_curvature', 
                                          [u, self.knots_ca, self.coeffs_x_ca, self.coeffs_y_ca], 
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
        
        # Get spline properties symbolically using the chord-length parameter u
        # Use the symbolic placeholders but substitute numerical values at function creation
        self.kappa = self.spline_curvature(self.u, self.knots_ca, self.coeffs_x_ca, self.coeffs_y_ca)
        self.ds_du = self.spline_tangent_magnitude(self.u, self.knots_ca, self.coeffs_x_ca, self.coeffs_y_ca)
        
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
        u_dot = s_dot / self.ds_du  # Convert arc-length rate to chord-length parameter rate
        e_y_dot = self.v * ca.sin(self.e_ψ + beta)
        e_ψ_dot = (self.v * ca.sin(beta) / L) - (self.kappa * s_dot)
        v_dot = self.a
        
        self.dynamics = ca.vertcat(s_dot, u_dot, e_y_dot, e_ψ_dot, v_dot)
        
        # Substitute numerical values for spline parameters to eliminate free variables
        self.substitutions = {}
        
        # Add knots substitutions
        for i in range(len(self.knots_np)):
            self.substitutions[self.knots_ca[i]] = self.knots_np[i]
        
        # Add coefficients substitutions
        for i in range(self.coeffs_x_np.shape[0]):
            for j in range(self.coeffs_x_np.shape[1]):
                self.substitutions[self.coeffs_x_ca[i,j]] = self.coeffs_x_np[i,j]
                self.substitutions[self.coeffs_y_ca[i,j]] = self.coeffs_y_np[i,j]
        
        # Apply substitutions to dynamics
        dynamics_with_values = self.dynamics
        for sym_var, num_val in self.substitutions.items():
            dynamics_with_values = ca.substitute(dynamics_with_values, sym_var, ca.SX(num_val))
        
        # Create function for dynamics evaluation (no parameter vector needed)
        self.dynamics_func = ca.Function('spline_path_dynamics',
                                       [self.state, self.input],
                                       [dynamics_with_values])
    
    def get_discrete_dynamics(self, dt: float) -> ca.Function:
        """
        Get discrete-time spline path dynamics using RK4 integration.
        
        Args:
            dt: Time step for discretization
            
        Returns:
            CasADi function for discrete spline path dynamics
        """
        # Check if we already have a cached function for this dt
        if dt == self._current_dt and dt in self._discrete_dynamics_cache:
            return self._discrete_dynamics_cache[dt]
        
        # Create new discrete dynamics function
        # RK4 integration - now fully symbolic
        k1 = self.dynamics
        k2 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k1)
        k3 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k2)
        k4 = ca.substitute(self.dynamics, self.state, self.state + dt * k3)
        
        state_next = self.state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Apply the same substitutions to the discrete dynamics
        state_next_with_values = state_next
        for sym_var, num_val in self.substitutions.items():
            state_next_with_values = ca.substitute(state_next_with_values, sym_var, ca.SX(num_val))
        
        # Use the same parameter vector p that we defined in the continuous dynamics
        discrete_dynamics = ca.Function('discrete_spline_path_dynamics',
                                      [self.state, self.input],
                                      [state_next_with_values])
        
        # Cache the function
        self._discrete_dynamics_cache[dt] = discrete_dynamics
        self._current_dt = dt
        
        return discrete_dynamics
    
    def update_spline_parameters(self, solver):
        """Update spline parameters in the acados solver."""
        # Get current parameter values
        current_params = solver.get(0, "p")
        
        # Debug prints
        print("\nDEBUG: Parameter dimensions:")
        print(f"Current ACADOS param dimension: {len(current_params)}")
        print(f"Knots shape: {self.knots_np.shape}")
        print(f"Coeffs_x shape: {self.coeffs_x_np.shape}")
        print(f"Coeffs_y shape: {self.coeffs_y_np.shape}")
        
        # Update only the spline parameters while preserving other parameters
        new_params = np.concatenate([
            self.knots_np,
            self.coeffs_x_np.flatten(),
            self.coeffs_y_np.flatten()
        ])
        
        print(f"New params dimension: {len(new_params)}")
        
        # Ensure we don't exceed the original parameter dimension
        if len(new_params) > len(current_params):
            print(f"Warning: New parameters ({len(new_params)}) exceed original dimension ({len(current_params)})")
            print("This suggests the waypoints or spline fitting has changed the parameter structure")
            # Truncate to match original dimension
            new_params = new_params[:len(current_params)]
        elif len(new_params) < len(current_params):
            print(f"Warning: New parameters ({len(new_params)}) are shorter than original dimension ({len(current_params)})")
            # Pad with zeros if needed
            new_params = np.pad(new_params, (0, len(current_params) - len(new_params)))
        
        # Update parameters for all nodes in the horizon
        for i in range(solver.acados_ocp.dims.N + 1):
            solver.set(i, "p", new_params)
    
    def get_spline_parameters(self) -> Dict:
        """Get current spline parameters."""
        return {
            'knots': self.knots_np,
            'coeffs_x': self.coeffs_x_np,
            'coeffs_y': self.coeffs_y_np
        }
    
    def update_waypoints(self, waypoints: np.ndarray):
        """Update waypoints and recompute spline parameters."""
        # Update spline coordinates
        self.spline_coords = CurvilinearCoordinates(waypoints)
        
        # Extract new coefficients
        self._extract_spline_coefficients()
        
        # Clear dynamics cache since parameters changed
        self.clear_dynamics_cache()
    
    def clear_dynamics_cache(self):
        """Clear the cached discrete dynamics functions."""
        self._discrete_dynamics_cache.clear()
        self._current_dt = None
    
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
        # Use symbolic function for numerical evaluation with numerical parameters
        result = self.spline_tangent_magnitude(u, self.knots_np, self.coeffs_x_np, self.coeffs_y_np)
        return float(result)
    
    def get_spline_state_at_progress(self, s: float) -> Dict:
        """
        Get spline state (u, curvature, ds/du) at given arc-length progress.
        
        Args:
            s: Arc-length progress along path
            
        Returns:
            Dictionary containing spline parameter info
        """
        # Convert arc-length to chord-length parameter
        u = self.spline_coords.arc_to_chord_length(s)
        u = np.clip(u, self.spline_coords.u_values[0], self.spline_coords.u_values[-1])
        
        # Get curvature at this chord-length parameter using symbolic function
        curvature = float(self.spline_curvature(u, self.knots_np, self.coeffs_x_np, self.coeffs_y_np))
        
        # Get tangent magnitude using symbolic function
        ds_du = float(self.spline_tangent_magnitude(u, self.knots_np, self.coeffs_x_np, self.coeffs_y_np))
        
        return {
            'u': u,
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
        
        # Use the existing Newton's method in CurvilinearCoordinates to find the closest point
        # Note: CurvilinearCoordinates.cartesian_to_curvilinear returns (u, d) 
        # where u is the chord-length parameter and d is the lateral distance
        u, e_y = self.spline_coords.cartesian_to_curvilinear(np.array([x, y]))
        
        # Convert chord-length parameter to arc-length using the proper method
        s = self.spline_coords.chord_to_arc_length(u)
        
        # Get reference heading at this chord-length parameter using numerical evaluation
        tangent_result = self.spline_tangent(u, self.knots_np, self.coeffs_x_np, self.coeffs_y_np)
        
        # Extract tangent components (handle CasADi DM if needed)
        if hasattr(tangent_result, 'full'):
            tangent = tangent_result.full().flatten()
        else:
            tangent = np.array(tangent_result).flatten()
        
        # Calculate reference heading
        ref_heading = np.arctan2(tangent[1], tangent[0])
        
        # Calculate heading error
        e_ψ = theta - ref_heading
        e_ψ = np.arctan2(np.sin(e_ψ), np.cos(e_ψ))  # Normalize to [-π, π]
        
        return np.array([s, u, e_y, e_ψ, v])
    
    def spline_curvilinear_to_cartesian(self, spline_curvilinear_state: np.ndarray) -> np.ndarray:
        """
        Convert spline curvilinear state to Cartesian coordinates.
        
        Args:
            spline_curvilinear_state: [s, u, e_y, e_ψ, v] in spline curvilinear coordinates
            
        Returns:
            [x, y, theta, v] in Cartesian coordinates
        """
        s, u, e_y, e_ψ, v = spline_curvilinear_state
        
        # Get reference position using the chord-length parameter u
        ref_pos = self.spline_position(u, self.knots_np, self.coeffs_x_np, self.coeffs_y_np)
        ref_pos = np.array([float(ref_pos[0]), float(ref_pos[1])])
        
        # Get reference heading using the chord-length parameter u
        tangent = self.spline_tangent(u, self.knots_np, self.coeffs_x_np, self.coeffs_y_np)
        tangent = np.array([float(tangent[0]), float(tangent[1])])
        tangent_norm = np.linalg.norm(tangent)
        
        # Safety check for zero tangent
        if tangent_norm < 1e-6:
            print(f"WARNING: Zero tangent vector at u={u}")
            # Use previous valid tangent or default
            ref_heading = 0.0
            normal = np.array([0.0, 1.0])
        else:
            ref_heading = np.arctan2(tangent[1], tangent[0])
            normal = np.array([-tangent[1]/tangent_norm, tangent[0]/tangent_norm])
        
        # Convert to Cartesian position
        x = ref_pos[0] + e_y * normal[0]
        y = ref_pos[1] + e_y * normal[1]
        
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
        # Call the function and convert result to numpy array
        result = discrete_dynamics(spline_curvilinear_state, input)
        # If result is a CasADi DM, use .full()
        if hasattr(result, 'full'):
            return np.array(result.full()).flatten()
        return np.array(result).flatten()
    
    def get_state_names(self) -> list:
        """Get names of spline curvilinear state variables."""
        return ['s', 'u', 'e_y', 'e_ψ', 'v']
    
    def get_constraints(self) -> Dict:
        """Get constraints (same as base vehicle model)."""
        constraints = self.vehicle_model.get_constraints()
        
        # Add spline parameter constraints (using chord-length parameters)
        constraints['u_min'] = self.spline_coords.u_values[0]
        constraints['u_max'] = self.spline_coords.u_values[-1]
        
        return constraints 