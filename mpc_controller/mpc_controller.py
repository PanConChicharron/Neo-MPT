import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from typing import Dict, Optional, Tuple
from .vehicle_model import VehicleModel
from .curvilinear_dynamics import CurvilinearDynamics


class MPCController:
    """
    Model Predictive Controller for path tracking using acados.
    
    This controller can use either Cartesian or curvilinear coordinates
    for optimal path following performance.
    """
    
    def __init__(self, vehicle_model: VehicleModel, 
                 prediction_horizon: float = 2.0, 
                 dt: float = 0.1,
                 use_curvilinear: bool = True):
        """
        Initialize MPC controller.
        
        Args:
            vehicle_model: Vehicle dynamics model
            prediction_horizon: Prediction horizon in seconds
            dt: Time step for discretization
            use_curvilinear: Whether to use curvilinear coordinates
        """
        self.vehicle_model = vehicle_model
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.use_curvilinear = use_curvilinear
        self.N = int(prediction_horizon / dt)  # Number of prediction steps
        
        # Initialize curvilinear dynamics if needed
        if self.use_curvilinear:
            self.curvilinear_dynamics = CurvilinearDynamics(vehicle_model)
        else:
            self.curvilinear_dynamics = None
        
        # Cost function weights
        self.Q = np.diag([10.0, 100.0, 10.0, 1.0])  # State weights
        self.R = np.diag([1.0, 0.1])                # Input weights
        self.Q_terminal = 2 * self.Q                # Terminal state weights
        
        # Initialize solver
        self.solver = None
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
        ocp = AcadosOcp()
        
        # Model setup
        if self.use_curvilinear:
            self._setup_curvilinear_model(ocp)
        else:
            self._setup_cartesian_model(ocp)
        
        # Solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 50  # Reduced from 100
        ocp.solver_options.qp_solver_iter_max = 50   # Add QP iteration limit
        ocp.solver_options.print_level = 0          # Reduced from 1 for cleaner output
        ocp.solver_options.nlp_solver_tol_stat = 1e-4  # Relaxed tolerance
        ocp.solver_options.nlp_solver_tol_eq = 1e-4    # Relaxed tolerance
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4  # Relaxed tolerance
        
        # Create solver
        self.solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    
    def _setup_curvilinear_model(self, ocp):
        """Setup OCP with curvilinear coordinates."""
        # State: [s, d, theta_e, v]
        # Input: [delta, a]
        
        # Use the curvilinear dynamics symbolic variables
        x = self.curvilinear_dynamics.state
        u = self.curvilinear_dynamics.input
        kappa = self.curvilinear_dynamics.kappa
        
        # Get discrete curvilinear dynamics
        discrete_dynamics = self.curvilinear_dynamics.get_discrete_dynamics(self.dt)
        x_next = discrete_dynamics(x, u, kappa)
        
        # Set model - use explicit discrete dynamics
        ocp.model.f_expl_expr = x_next
        ocp.model.x = x
        ocp.model.u = u
        ocp.model.p = ca.vertcat(kappa)
        ocp.model.name = 'curvilinear_vehicle'
        
        # Dimensions MUST be set first
        ocp.solver_options.N_horizon = self.N
        ocp.dims.ny = 4    # output dimension (states only)
        ocp.dims.ny_e = 4  # terminal output dimension
        
        # Set parameter dimensions
        ocp.dims.np = 1  # One parameter: curvature
        
        # Initialize parameter values
        ocp.parameter_values = np.zeros(1)  # Initialize curvature to zero
        
        # Cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # Output selection matrices
        ocp.cost.Vx = np.eye(4)  # Select all 4 states
        ocp.cost.Vu = np.zeros((4, 2))  # No input in output
        
        # Cost weights
        ocp.cost.W = self.Q      # 4x4 matrix for 4 outputs
        
        # Terminal cost
        ocp.cost.Vx_e = np.eye(4)
        ocp.cost.W_e = self.Q_terminal
        
        # Constraints
        constraints = self._handle_constraints(self.curvilinear_dynamics.get_constraints())
        
        ocp.constraints.lbu = np.array([constraints['steering_min'], 
                                       constraints['acceleration_min']])
        ocp.constraints.ubu = np.array([constraints['steering_max'], 
                                       constraints['acceleration_max']])
        ocp.constraints.idxbu = np.array([0, 1])
        
        # State constraints for curvilinear coordinates
        # [s, d, theta_e, v] - constrain lateral deviation, heading error, and velocity
        ocp.constraints.lbx = np.array([-5.0, -np.pi, constraints['velocity_min']])  # [d, theta_e, v]
        ocp.constraints.ubx = np.array([5.0, np.pi, constraints['velocity_max']])
        ocp.constraints.idxbx = np.array([1, 2, 3])  # constrain d, theta_e, v (not s)
        
        # Initial condition constraint
        ocp.constraints.x0 = np.zeros(4)
        
        # Initialize reference values to avoid dimension errors
        ocp.cost.yref = np.zeros(4)    # 4-dimensional reference
        ocp.cost.yref_e = np.zeros(4)  # 4-dimensional terminal reference
        
        # Solver settings
        ocp.solver_options.tf = self.prediction_horizon
    
    def _setup_cartesian_model(self, ocp):
        """Setup OCP with Cartesian coordinates."""
        # State: [x, y, theta, v]
        # Input: [delta, a]
        
        x_pos = ca.SX.sym('x')
        y_pos = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        
        x = ca.vertcat(x_pos, y_pos, theta, v)
        
        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        u = ca.vertcat(delta, a)
        
        # Get discrete dynamics
        discrete_dynamics = self.vehicle_model.get_discrete_dynamics(self.dt)
        x_next = discrete_dynamics(x, u)
        
        # Set model
        ocp.model.f_expl_expr = x_next
        ocp.model.x = x
        ocp.model.u = u
        ocp.model.name = 'cartesian_vehicle'
        
        # Dimensions MUST be set first
        ocp.solver_options.N_horizon = self.N
        ocp.dims.ny = 4    # output dimension (states only)
        ocp.dims.ny_e = 4  # terminal output dimension
        
        # Cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # Output selection matrices - this is what was missing!
        ocp.cost.Vx = np.eye(4)  # Select all 4 states
        ocp.cost.Vu = np.zeros((4, 2))  # No input in output (4 outputs, 2 inputs)
        
        # Cost weights
        ocp.cost.W = self.Q      # 4x4 matrix for 4 outputs
        
        # Terminal cost
        ocp.cost.Vx_e = np.eye(4)
        ocp.cost.W_e = self.Q_terminal
        
        # Constraints
        constraints = self._handle_constraints(self.vehicle_model.get_constraints())
        
        ocp.constraints.lbu = np.array([constraints['steering_min'], 
                                       constraints['acceleration_min']])
        ocp.constraints.ubu = np.array([constraints['steering_max'], 
                                       constraints['acceleration_max']])
        ocp.constraints.idxbu = np.array([0, 1])
        
        # State constraints
        ocp.constraints.lbx = np.array([constraints['velocity_min']])  # minimum velocity
        ocp.constraints.ubx = np.array([constraints['velocity_max']])  # maximum velocity
        ocp.constraints.idxbx = np.array([3])  # constrain velocity only
        
        # Initial condition constraint
        ocp.constraints.x0 = np.zeros(4)
        
        # Initialize reference values to avoid dimension errors
        ocp.cost.yref = np.zeros(4)    # 4-dimensional reference
        ocp.cost.yref_e = np.zeros(4)  # 4-dimensional terminal reference
        
        ocp.solver_options.tf = self.prediction_horizon
    
    def solve(self, current_state: np.ndarray, 
              reference_trajectory: Dict,
              current_curvature: Optional[np.ndarray] = None) -> Dict:
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current vehicle state
            reference_trajectory: Reference trajectory data
            current_curvature: Path curvature values (for curvilinear mode)
            
        Returns:
            Dictionary containing optimal control sequence and predicted trajectory
        """
        if self.solver is None:
            raise RuntimeError("MPC solver not initialized")
        
        # Set initial condition
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)
        
        # Set reference trajectory
        self._set_reference(reference_trajectory, current_curvature)
        
        # Solve optimization problem
        status = self.solver.solve()
        
        if status != 0:
            print(f"MPC solver failed with status {status}")
        
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
        
        return {
            'optimal_input': optimal_inputs[0],  # First control input
            'optimal_sequence': np.array(optimal_inputs),
            'predicted_trajectory': np.array(predicted_states),
            'solver_status': status,
            'cost': self.solver.get_cost()
        }
    
    def _set_reference(self, reference_trajectory: Dict, 
                      current_curvature: Optional[np.ndarray] = None):
        """Set reference trajectory for the optimization problem."""
        
        if self.use_curvilinear and current_curvature is not None:
            # Set curvature parameters
            for i in range(self.N):
                kappa_i = current_curvature[min(i, len(current_curvature)-1)]
                self.solver.set(i, "p", np.array([kappa_i]))
        
        # Get reference data
        positions = reference_trajectory['positions']
        headings = reference_trajectory['headings']
        velocities = reference_trajectory.get('velocities', [10.0] * len(positions))
        
        # Set reference for each prediction step
        for i in range(self.N):
            if i < len(positions):
                if self.use_curvilinear:
                    # Reference in curvilinear coordinates
                    ref_state = np.array([
                        reference_trajectory['s_values'][i],
                        0.0,  # desired lateral deviation = 0
                        0.0,  # desired heading error = 0
                        velocities[i]
                    ])
                else:
                    # Reference in Cartesian coordinates
                    ref_state = np.array([
                        positions[i, 0],
                        positions[i, 1],
                        headings[i],
                        velocities[i]
                    ])
            else:
                # Use last reference for remaining steps
                if self.use_curvilinear:
                    ref_state = np.array([
                        reference_trajectory['s_values'][-1],
                        0.0, 0.0,
                        velocities[-1]
                    ])
                else:
                    ref_state = np.array([
                        positions[-1, 0],
                        positions[-1, 1],
                        headings[-1],
                        velocities[-1]
                    ])
            
            # Set reference (state only)
            self.solver.set(i, "yref", ref_state)
        
        # Set terminal reference (state only)
        if self.use_curvilinear:
            terminal_ref = np.array([
                reference_trajectory['s_values'][-1],
                0.0, 0.0,
                velocities[-1]
            ])
        else:
            terminal_ref = np.array([
                positions[-1, 0],
                positions[-1, 1],
                headings[-1],
                velocities[-1]
            ])
        
        self.solver.set(self.N, "yref", terminal_ref)
    
    def set_weights(self, Q: np.ndarray, R: np.ndarray, Q_terminal: Optional[np.ndarray] = None):
        """
        Update cost function weights.
        
        Args:
            Q: State weights
            R: Input weights  
            Q_terminal: Terminal state weights (optional)
        """
        self.Q = Q
        self.R = R
        if Q_terminal is not None:
            self.Q_terminal = Q_terminal
        else:
            self.Q_terminal = 2 * Q
        
        # Note: Weights must be set during OCP setup, not after solver creation
        # If you need to change weights, recreate the solver
        if self.solver is not None:
            print("Warning: Weights can only be set during solver initialization.")
            print("To change weights, recreate the MPC controller.")
    
    def get_prediction_horizon(self) -> float:
        """Get prediction horizon in seconds."""
        return self.prediction_horizon
    
    def get_time_step(self) -> float:
        """Get time step."""
        return self.dt 