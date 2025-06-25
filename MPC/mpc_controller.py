import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from typing import Dict
from Dynamics.vehicle_model import VehicleModel
from Dynamics.cubic_spline_path_dynamics import CubicSplinePathDynamics


class MPCController:
    """
    Model Predictive Controller for spline path tracking using acados.
    
    This controller uses spline path dynamics for optimal path following performance,
    with costs on heading error, cross-track error, and progress towards goal.
    """
    
    def __init__(self,
                 vehicle_model: VehicleModel, 
                 state_weights: np.ndarray, input_weights: np.ndarray, 
                 terminal_state_weights: np.ndarray,
                 prediction_horizon: float = 2.0, 
                 dt: float = 0.1,):
        """
        Initialize MPC controller.
        
        Args:
            vehicle_model: Vehicle dynamics model
            prediction_horizon: Prediction horizon in seconds
            dt: Time step for discretization
        """
        self.vehicle_model = vehicle_model
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.N = int(prediction_horizon / dt)  # Number of prediction steps
        
        # Initialize spline path dynamics
        self.spline_dynamics = None  # Will be set when path is provided

        self.state_weights = state_weights
        self.input_weights = input_weights
        self.terminal_state_weights = terminal_state_weights
        
        # Initialize solver
        self.solver = None

        self.u_min = u_min
        self.u_max = u_max

        self.path_length = path_length
    
    def set_path(self, spline_dynamics: CubicSplinePathDynamics):
        """
        Set the spline path for tracking.
        
        Args:
            spline_dynamics: SplinePathDynamics object containing the path
        """
        self.spline_dynamics = spline_dynamics
        self._setup_ocp()
    
    def _handle_constraints(self, constraints: Dict) -> Dict:
        """
        Handle None constraints by replacing with large values for unconstrained behavior.
        
        Args:
            constraints: Dictionary with constraint values (may contain None)
            
        Returns:
            Dictionary with None values replaced by large numbers
        """
        # Large values for "unconstrained" behavior
        LARGE_VALUE = 1e6
        
        handled = {}
        for key, value in constraints.items():
            if value is None:
                if 'min' in key:
                    handled[key] = -LARGE_VALUE
                else:  # max constraints
                    handled[key] = LARGE_VALUE
            else:
                handled[key] = value
        
        return handled
    
    def _setup_ocp(self):
        """Setup the optimal control problem using acados."""
        if self.spline_dynamics is None:
            raise RuntimeError("Spline path must be set before setting up OCP")
            
        ocp = AcadosOcp()
        
        # Get symbolic variables from spline dynamics
        x = self.spline_dynamics.state  # [s, u, e_y, e_ψ, v]
        u = self.spline_dynamics.input   # [delta, a]

        # Get CONTINUOUS dynamics with numerical parameter substitution
        # ACADOS expects continuous dynamics f(x,u) = dx/dt, not discrete
        dynamics_expr = self.spline_dynamics.dynamics
        
        # Set model - ACADOS expects continuous dynamics, not discrete
        ocp.model.f_expl_expr = dynamics_expr
        ocp.model.x = x
        ocp.model.u = u

        u_min_ca = ca.SX.sym('u_min', 1)
        u_max_ca = ca.SX.sym('u_max', 1)

        path_length_ca = ca.SX.sym('path_length', 1)

        ocp.model.p = ca.vertcat(self.spline_dynamics.parameters, u_min_ca, u_max_ca, path_length_ca)
        ocp.model.name = 'spline_path_vehicle'
        
        ocp.dims.np = ocp.model.p.numel()
        ocp.parameter_values = np.zeros(ocp.dims.np)
        
        # Dimensions
        ocp.solver_options.N_horizon = self.N
        ocp.dims.ny = 7    # output dimension (5 states + 2 inputs)
        ocp.dims.ny_e = 5  # terminal output dimension (just states)
        
        # Cost function setup
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # Cost matrices
        ny = 7  # Number of outputs (5 states + 2 inputs)
        ny_e = 5  # Number of terminal outputs (just states)
        
        # Output matrices
        Vx = np.zeros((ny, 5))  # State output matrix
        Vu = np.zeros((ny, 2))  # Input output matrix
        Vx_e = np.zeros((ny_e, 5))  # Terminal state output matrix
        
        # State terms
        Vx[0:5, 0:5] = np.eye(5)  # All states
        # Input terms
        Vu[5:7, 0:2] = np.eye(2)  # All inputs
        
        # Terminal cost only includes states
        Vx_e[0:5, 0:5] = np.eye(5)
        
        # Cost weights
        W = np.zeros((ny, ny))
        W[0:5, 0:5] = np.diag(self.state_weights)
        W[5:7, 5:7] = np.diag(self.input_weights)
        
        W_e = np.diag(self.terminal_state_weights)
        
        # Set cost matrices
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = Vx_e
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        
        # Constraints
        constraints = self._handle_constraints(self.spline_dynamics.get_constraints())
        
        # Input constraints
        ocp.constraints.lbu = np.array([constraints['steering_min'], 
                                       constraints['acceleration_min']])
        ocp.constraints.ubu = np.array([constraints['steering_max'], 
                                       constraints['acceleration_max']])
        ocp.constraints.idxbu = np.array([0, 1])
        
        # State constraints
        # Use numerical bounds for non-parametric constraints
        # Parametric bounds (u_min, u_max, path_length) will be handled via constraint functions
        ocp.constraints.lbx = np.array([
            -1e6,                    # s: will be constrained via constraint function
            -1e6,                    # u: will be constrained via constraint function  
            -1.0,                    # e_y: relaxed lateral error bounds
            -np.pi/3,                # e_ψ: heading error bounds
            constraints['velocity_min']  # v: velocity bounds
        ])
        ocp.constraints.ubx = np.array([
            1e6,                     # s: will be constrained via constraint function
            1e6,                     # u: will be constrained via constraint function
            1.0,                     # e_y: relaxed lateral error bounds
            np.pi/3,                 # e_ψ: heading error bounds
            constraints['velocity_max']  # v: velocity bounds
        ])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])
        
        # Add parametric constraint functions for s, u bounds
        # Constraint: 0 <= s <= path_length
        # Constraint: u_min <= u <= u_max
        h_expr = ca.vertcat(
            x[0],                    # s >= 0 (will be constrained as h >= 0)
            path_length_ca - x[0],   # s <= path_length (path_length - s >= 0)
            x[1] - u_min_ca,         # u >= u_min (u - u_min >= 0)
            u_max_ca - x[1]          # u <= u_max (u_max - u >= 0)
        )
        
        ocp.model.con_h_expr = h_expr
        ocp.constraints.lh = np.array([0.0, 0.0, 0.0, 0.0])  # All constraints >= 0
        ocp.constraints.uh = np.array([1e6, 1e6, 1e6, 1e6])  # Upper bounds (effectively unbounded)
        ocp.dims.nh = 4
        
        # Initial condition constraint - will be set in solve()
        ocp.constraints.x0 = np.zeros(5)  # Placeholder, will be updated
        
        # Initialize reference values
        ocp.cost.yref = np.zeros(7)
        ocp.cost.yref_e = np.zeros(5)  # 5-dimensional terminal reference
        
        # Solver settings
        ocp.solver_options.tf = self.prediction_horizon
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.print_level = 0
        
        # Create solver
        self.solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        
        # No longer need to set parameter values since we use direct substitution
        # self.spline_dynamics.update_spline_parameters(self.solver)
    
    def solve(self, current_state: np.ndarray, 
              parameters: np.ndarray,
              reference_trajectory: Dict) -> Dict:
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current vehicle state [s, u, e_y, e_ψ, v]
            reference_trajectory: Reference trajectory data
            
        Returns:
            Dictionary containing optimal control sequence and predicted trajectory
        """
        if self.solver is None:
            raise RuntimeError("MPC solver not initialized")

        # Set initial condition
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        # Set initial guess for optimization variables
        for i in range(self.N + 1):
            self.solver.set(i, "x", current_state)
        for i in range(self.N):
            self.solver.set(i, "u", np.zeros(2))
        for i in range(self.N):
            self.solver.set(i, "p", parameters)
        
        # Extract reference trajectory data
        # Handle both old 's_values' and new 'u_values' naming for backward compatibility
        trajectory_params = reference_trajectory.get('u_values', reference_trajectory.get('s_values', []))
        velocities = reference_trajectory.get('velocities', [10.0] * len(trajectory_params))
        
        # Set reference for each node in the horizon (excluding terminal node)
        for i in range(self.N):  # Only go to N-1, not N+1
            # Set reference state + input: [s, u, e_y, e_ψ, v, delta, a]
            yref_vector = np.array([
                trajectory_params[i],  # s (arc-length progress)
                trajectory_params[i],  # u (chord-length parameter) - approximation
                0.0,                   # e_y (lateral error)
                0.0,                   # e_ψ (heading error)
                velocities[i] if i < len(velocities) else 10.0,  # v (velocity)
                0.0,                   # delta (steering reference)
                0.0                    # a (acceleration reference)
            ])
            self.solver.set(i, "yref", yref_vector)
        
        # Set terminal reference (only for node N)
        if len(trajectory_params) > 0:
            yref_e_vector = np.array([
                trajectory_params[-1],
                trajectory_params[-1],
                0.0,
                0.0,
                velocities[-1] if len(velocities) > 0 else 10.0
            ])
            self.solver.set(self.N, "y_ref", yref_e_vector)
        
        # Solve optimization problem
        status = self.solver.solve()
        
        if status != 0:
            print(f"Warning: MPC solver returned status {status}")
        
        # Extract solution
        optimal_inputs = []
        predicted_states = []
        
        for i in range(self.N):
            u_opt = self.solver.get(i, "u")
            optimal_inputs.append(u_opt)
            
            x_opt = self.solver.get(i, "x")
            predicted_states.append(x_opt)
        
        # Get terminal state
        x_terminal = self.solver.get(self.N, "x")
        predicted_states.append(x_terminal)

        result = {
            'optimal_input': optimal_inputs[0],  # First control input
            'optimal_sequence': np.array(optimal_inputs),
            'predicted_trajectory': np.array(predicted_states),
            'solver_status': status,
            'cost': self.solver.get_cost()
        }

        if result['solver_status'] != 0:
            for i in range(self.N):
                if i < self.N - 1:
                    self.solver.set(i, "x", result['predicted_trajectory'][i + 1])
                    self.solver.set(i, "u", result['optimal_sequence'][i + 1])
                else:
                    # For the last step, use the terminal state
                    self.solver.set(i, "x", result['predicted_trajectory'][-1])
                    self.solver.set(i, "u", result['optimal_sequence'][-1])
            
            # Set terminal state guess
            self.solver.set(self.N, "x", result['predicted_trajectory'][-1])
    
        
        return result
    
    def update_waypoints(self, waypoints: np.ndarray):
        """Update waypoints and recompute spline parameters."""        
        # Update spline parameters
        self.spline_dynamics.update_waypoints(waypoints)