import numpy as np
import casadi as ca
from typing import Dict
from .vehicle_model import VehicleModel


class CurvilinearDynamics:
    """
    Curvilinear coordinate dynamics for path-following MPC.
    
    This class transforms standard vehicle dynamics into curvilinear coordinates
    (s, e_y, e_ψ, v) where:
    - s: progress along the reference path
    - e_y: lateral error (cross-track error) from the reference path  
    - e_ψ: heading error relative to path tangent
    - v: velocity magnitude
    
    The dynamics are derived for a bicycle model following a parametric path
    with curvature κ(s).
    """
    
    def __init__(self, vehicle_model: VehicleModel):
        """
        Initialize curvilinear dynamics.
        
        Args:
            vehicle_model: Base vehicle model in Cartesian coordinates
        """
        self.vehicle_model = vehicle_model
        self.n_states = 4  # [s, e_y, e_ψ, v]
        self.n_inputs = 2  # [delta, a]
        
        # Create symbolic variables
        self._create_symbolic_model()
    
    def _create_symbolic_model(self):
        """Create symbolic curvilinear dynamics model."""
        # Curvilinear state variables
        self.s = ca.SX.sym('s')           # arc length along path
        self.e_y = ca.SX.sym('e_y')       # lateral error (cross-track error)
        self.e_ψ = ca.SX.sym('e_ψ')   # heading error
        self.v = ca.SX.sym('v')           # velocity
        
        self.state = ca.vertcat(self.s, self.e_y, self.e_ψ, self.v)
        
        # Input variables (same as vehicle model)
        self.delta = ca.SX.sym('delta')   # steering angle
        self.a = ca.SX.sym('a')           # acceleration
        
        self.input = ca.vertcat(self.delta, self.a)
        
        # Path curvature as parameter
        self.kappa = ca.SX.sym('kappa')   # path curvature at current s
        
        # Curvilinear dynamics derivation:
        # For a bicycle model in curvilinear coordinates following a path with curvature κ(s)
        
        # Vehicle slip angle (from bicycle model)
        lf = self.vehicle_model.wheelbase_front
        lr = self.vehicle_model.wheelbase_rear
        L = lf + lr  # total wheelbase
        
        beta = ca.atan(lr * ca.tan(self.delta) / L)
        
        # Curvilinear dynamics equations:
        # ds/dt = v * cos(e_ψ + β) / (1 - κ*e_y)
        # de_y/dt = v * sin(e_ψ + β)  
        # de_ψ/dt = v * sin(β) / L - κ * ds/dt
        # dv/dt = a
        
        # Denominator term for s dynamics
        denom = 1 - self.kappa * self.e_y
        
        # State derivatives
        s_dot = self.v * ca.cos(self.e_ψ + beta) / denom
        e_y_dot = self.v * ca.sin(self.e_ψ + beta)
        e_ψ_dot = (self.v * ca.sin(beta) / L) - (self.kappa * s_dot)
        v_dot = self.a
        
        self.dynamics = ca.vertcat(s_dot, e_y_dot, e_ψ_dot, v_dot)
        
        # Create function for dynamics evaluation
        self.dynamics_func = ca.Function('curvilinear_dynamics',
                                       [self.state, self.input, self.kappa],
                                       [self.dynamics])
    
    def get_discrete_dynamics(self, dt: float) -> ca.Function:
        """
        Get discrete-time curvilinear dynamics using RK4 integration.
        
        Args:
            dt: Time step for discretization
            
        Returns:
            CasADi function for discrete curvilinear dynamics
        """
        # RK4 integration with curvature parameter
        k1 = self.dynamics
        k2 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k1)
        k3 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k2)
        k4 = ca.substitute(self.dynamics, self.state, self.state + dt * k3)
        
        state_next = self.state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return ca.Function('discrete_curvilinear_dynamics',
                          [self.state, self.input, self.kappa],
                          [state_next])
    
    def get_continuous_dynamics(self) -> ca.Function:
        """Get continuous-time curvilinear dynamics function."""
        return self.dynamics_func
    
    def simulate_step(self, curvilinear_state: np.ndarray, 
                     input: np.ndarray, curvature: float, dt: float) -> np.ndarray:
        """
        Simulate one step of curvilinear dynamics.
        
        Args:
            curvilinear_state: Current state [s, e_y, e_ψ, v]
            input: Control input [delta, a]
            curvature: Path curvature at current s
            dt: Time step
            
        Returns:
            Next curvilinear state
        """
        discrete_dynamics = self.get_discrete_dynamics(dt)
        return np.array(discrete_dynamics(curvilinear_state, input, curvature)).flatten()
    
    def get_state_names(self) -> list:
        """Get names of curvilinear state variables."""
        return ['s', 'e_y', 'e_ψ', 'v']
    
    def get_input_names(self) -> list:
        """Get names of input variables."""
        return ['delta', 'a']
    
    def get_constraints(self) -> Dict:
        """Get constraints (same as base vehicle model)."""
        return self.vehicle_model.get_constraints() 