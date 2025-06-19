import numpy as np
import casadi as ca
from typing import Dict, Optional


class VehicleModel:
    """
    Vehicle dynamics model for MPC path tracking.
    
    This class implements a bicycle model suitable for path tracking applications.
    The model can be used in curvilinear coordinates for better path following performance.
    """
    
    def __init__(self, wheelbase_front: float = 1.5, wheelbase_rear: float = 1.0, 
                 max_steering: float = None, max_velocity: float = None, max_acceleration: float = None,
                 min_steering: float = None, min_acceleration: float = None, min_velocity: float = None):
        """
        Initialize vehicle model parameters.
        
        Args:
            wheelbase_front: Distance from center of gravity to front axle [m]
            wheelbase_rear: Distance from center of gravity to rear axle [m]
            max_steering: Maximum steering angle [rad] (default: None = unconstrained)
            max_velocity: Maximum velocity [m/s] (default: None = unconstrained)
            max_acceleration: Maximum acceleration [m/s²] (default: None = unconstrained)
            min_steering: Minimum steering angle [rad] (default: None = unconstrained)
            min_acceleration: Minimum acceleration [m/s²] (default: None = unconstrained)
            min_velocity: Minimum velocity [m/s] (default: None = unconstrained)
        """
        self.wheelbase_front = wheelbase_front
        self.wheelbase_rear = wheelbase_rear
        self.max_steering = max_steering
        self.min_steering = min_steering
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        
        # State and input dimensions
        self.n_states = 4  # [x, y, theta, v]
        self.n_inputs = 2  # [steering_angle, acceleration]
        
        # Create symbolic variables for CasADi
        self._create_symbolic_model()
    
    def _create_symbolic_model(self):
        """Create symbolic model using CasADi for optimization."""
        # State variables: [x, y, theta, v]
        self.x = ca.SX.sym('x')      # x position
        self.y = ca.SX.sym('y')      # y position
        self.theta = ca.SX.sym('theta')  # heading angle
        self.v = ca.SX.sym('v')      # velocity
        
        self.state = ca.vertcat(self.x, self.y, self.theta, self.v)
        
        # Input variables: [delta, a]
        self.delta = ca.SX.sym('delta')  # steering angle
        self.a = ca.SX.sym('a')          # acceleration
        
        self.input = ca.vertcat(self.delta, self.a)
        
        # Bicycle model dynamics
        # Slip angle: beta = atan(lr * tan(delta) / (lf + lr))
        # where lf = wheelbase_front, lr = wheelbase_rear
        beta = ca.atan(self.wheelbase_rear * ca.tan(self.delta) / (self.wheelbase_front + self.wheelbase_rear))
        
        self.dynamics = ca.vertcat(
            self.v * ca.cos(self.theta + beta),           # dx/dt
            self.v * ca.sin(self.theta + beta),           # dy/dt
            (self.v / (self.wheelbase_front + self.wheelbase_rear)) * ca.sin(beta),     # dtheta/dt
            self.a                                         # dv/dt
        )
        
        # Create function for dynamics evaluation
        self.dynamics_func = ca.Function('dynamics', 
                                       [self.state, self.input], 
                                       [self.dynamics])
    
    def get_discrete_dynamics(self, dt: float) -> ca.Function:
        """
        Get discrete-time dynamics using RK4 integration.
        
        Args:
            dt: Time step for discretization
            
        Returns:
            CasADi function for discrete dynamics
        """
        # RK4 integration
        k1 = self.dynamics
        k2 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k1)
        k3 = ca.substitute(self.dynamics, self.state, self.state + dt/2 * k2)
        k4 = ca.substitute(self.dynamics, self.state, self.state + dt * k3)
        
        state_next = self.state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return ca.Function('discrete_dynamics', 
                          [self.state, self.input], 
                          [state_next])
    
    def get_constraints(self) -> Dict:
        """Get vehicle constraints. None values indicate unconstrained."""
        return {
            'steering_min': self.min_steering,
            'steering_max': self.max_steering,
            'acceleration_min': self.min_acceleration,
            'acceleration_max': self.max_acceleration,
            'velocity_min': self.min_velocity,
            'velocity_max': self.max_velocity
        }
    
    def simulate_step(self, state: np.ndarray, input: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate one step of vehicle dynamics.
        
        Args:
            state: Current state [x, y, theta, v]
            input: Control input [delta, a]
            dt: Time step
            
        Returns:
            Next state
        """
        discrete_dynamics = self.get_discrete_dynamics(dt)
        return np.array(discrete_dynamics(state, input)).flatten()
    
    def get_state_names(self) -> list:
        """Get names of state variables."""
        return ['x', 'y', 'theta', 'v']
    
    def get_input_names(self) -> list:
        """Get names of input variables."""
        return ['delta', 'a']
    
    def get_curvilinear_state_names(self) -> list:
        """Get names of curvilinear state variables."""
        return ['s', 'd', 'theta_e', 'v'] 