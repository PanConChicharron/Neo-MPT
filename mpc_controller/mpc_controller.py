import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from typing import Dict, Optional, Tuple
from .vehicle_model import VehicleModel
from .spline_path_dynamics import SplinePathDynamics


class MPCController:
    """
    Model Predictive Controller for spline path tracking using acados.
    
    This controller uses spline path dynamics for optimal path following performance,
    with costs on heading error, cross-track error, and progress towards goal.
    """
    
    def __init__(self, vehicle_model: VehicleModel, 
                 prediction_horizon: float = 2.0, 
                 dt: float = 0.1):
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
        
        # Cost function weights for spline path tracking
        # State weights: [s, u, e_y, e_ψ, v]
        self.Q = np.diag([
            1.0,    # s (progress) - small weight since we use terminal cost
            0.0,    # u (spline parameter) - not directly penalized
            100.0,  # e_y (cross-track error) - heavily penalized
            10.0,   # e_ψ (heading error) - moderate weight
            1.0     # v (velocity) - small weight
        ])
        
        # Input weights: [delta, a]
        self.R = np.diag([
            1.0,    # steering
            0.1     # acceleration
        ])
        
        # Terminal state weights (heavier than stage costs)
        self.Q_terminal = np.diag([
            10.0,   # s (progress) - higher weight to reach goal
            0.0,    # u (spline parameter)
            200.0,  # e_y (cross-track error) - very heavy terminal penalty
            20.0,   # e_ψ (heading error) - heavier terminal penalty
            2.0     # v (velocity)
        ])
        
        # Initialize solver
        self.solver = None
        self.total_path_length = None
    
    def set_path(self, spline_dynamics: SplinePathDynamics):
        """
        Set the spline path for tracking.
        
        Args:
            spline_dynamics: SplinePathDynamics object containing the path
        """
        self.spline_dynamics = spline_dynamics
        self.total_path_length = spline_dynamics.spline_coords.path_length
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
        
        # Get spline parameters
        knots = self.spline_dynamics.knots_ca
        coeffs_x = self.spline_dynamics.coeffs_x_ca
        coeffs_y = self.spline_dynamics.coeffs_y_ca
        
        # Get discrete dynamics
        discrete_dynamics = self.spline_dynamics.get_discrete_dynamics(self.dt)
        
        # Set model
        ocp.model.f_expl_expr = discrete_dynamics(x, u, ocp.model.p)  # Pass parameters as single vector
        ocp.model.x = x
        ocp.model.u = u
        ocp.model.name = 'spline_path_vehicle'
        
        # Set parameters - ensure consistent dimension
        param_dim = (self.spline_dynamics.knots.shape[0] + 
                    self.spline_dynamics.coeffs_x.shape[0] * self.spline_dynamics.coeffs_x.shape[1] + 
                    self.spline_dynamics.coeffs_y.shape[0] * self.spline_dynamics.coeffs_y.shape[1])
        
        print("\nDEBUG: Initial OCP parameter setup:")
        print(f"Knots shape: {self.spline_dynamics.knots.shape}")
        print(f"Coeffs_x shape: {self.spline_dynamics.coeffs_x.shape}")
        print(f"Coeffs_y shape: {self.spline_dynamics.coeffs_y.shape}")
        print(f"Total parameter dimension: {param_dim}")
        
        ocp.model.p = ca.SX.sym('p', param_dim)
        
        # Initialize parameter values
        param_values = np.concatenate([
            self.spline_dynamics.knots_np,
            self.spline_dynamics.coeffs_x_np.flatten(),
            self.spline_dynamics.coeffs_y_np.flatten()
        ])
        ocp.parameter_values = param_values
        
        # Dimensions
        ocp.solver_options.N_horizon = self.N
        ocp.dims.ny = 5    # output dimension (all states)
        ocp.dims.ny_e = 5  # terminal output dimension
        ocp.dims.np = param_dim  # parameter dimension
        
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
        W[0:5, 0:5] = np.diag([1.0, 1.0, 10.0, 10.0, 1.0])  # State weights
        W[5:7, 5:7] = np.diag([1.0, 1.0])  # Input weights
        
        W_e = np.diag([1.0, 1.0, 10.0, 10.0, 1.0])  # Terminal weights
        
        # Set cost matrices
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = Vx_e
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        
        # Constraints
        constraints = self._handle_constraints(self.spline_dynamics.get_constraints())
        
        print("\nDEBUG: Constraint setup:")
        print("Vehicle constraints:")
        print(f"  Steering: [{constraints['steering_min']}, {constraints['steering_max']}]")
        print(f"  Acceleration: [{constraints['acceleration_min']}, {constraints['acceleration_max']}]")
        print(f"  Velocity: [{constraints['velocity_min']}, {constraints['velocity_max']}]")
        print(f"  Spline parameter u: [{constraints['u_min']}, {constraints['u_max']}]")
        
        # Input constraints
        ocp.constraints.lbu = np.array([constraints['steering_min'], 
                                       constraints['acceleration_min']])
        ocp.constraints.ubu = np.array([constraints['steering_max'], 
                                       constraints['acceleration_max']])
        ocp.constraints.idxbu = np.array([0, 1])
        
        # State constraints
        # Constrain all states: [s, u, e_y, e_ψ, v]
        total_chord_length = self.spline_dynamics.spline_coords.total_chord_length
        ocp.constraints.lbx = np.array([
            constraints['u_min'],     # s: bounded by path length
            0.0,                      # u: bounded by total chord length
            -10.0,                    # e_y: relaxed lateral error bounds
            -np.pi,                   # e_ψ: heading error bounds
            constraints['velocity_min']  # v: velocity bounds
        ])
        ocp.constraints.ubx = np.array([
            constraints['u_max'],     # s: bounded by path length
            total_chord_length,       # u: bounded by total chord length
            10.0,                     # e_y: relaxed lateral error bounds
            np.pi,                    # e_ψ: heading error bounds
            constraints['velocity_max']  # v: velocity bounds
        ])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])  # indices of all states
        
        print("\nDEBUG: State constraint indices:")
        print(f"Constrained state indices: {ocp.constraints.idxbx}")
        print(f"State bounds:")
        print(f"  s: [{ocp.constraints.lbx[0]}, {ocp.constraints.ubx[0]}]")
        print(f"  u: [{ocp.constraints.lbx[1]}, {ocp.constraints.ubx[1]}]")
        print(f"  e_y: [{ocp.constraints.lbx[2]}, {ocp.constraints.ubx[2]}]")
        print(f"  e_ψ: [{ocp.constraints.lbx[3]}, {ocp.constraints.ubx[3]}]")
        print(f"  v: [{ocp.constraints.lbx[4]}, {ocp.constraints.ubx[4]}]")
        
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
        
        print("\nDEBUG: Solver settings:")
        print(f"Prediction horizon: {self.prediction_horizon}")
        print(f"Time step: {self.dt}")
        print(f"Number of steps: {self.N}")
        print(f"QP solver: {ocp.solver_options.qp_solver}")
        print(f"NLP solver: {ocp.solver_options.nlp_solver_type}")
        
        # Create solver
        self.solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        
        # Set initial parameter values
        self.spline_dynamics.update_spline_parameters(self.solver)
    
    def solve(self, current_state: np.ndarray, 
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
        
        # Debug prints for initial state and constraints
        print("\nDEBUG: Initial state:")
        print(f"Current state: {current_state}")
        print(f"State bounds:")
        print(f"  s: [0, {self.total_path_length}]")
        print(f"  u: [{self.spline_dynamics.spline_coords.u_values[0]}, {self.spline_dynamics.spline_coords.u_values[-1]}]")
        print(f"  e_y: [{self.solver.acados_ocp.constraints.lbx[2]}, {self.solver.acados_ocp.constraints.ubx[2]}]")
        print(f"  e_ψ: [{self.solver.acados_ocp.constraints.lbx[3]}, {self.solver.acados_ocp.constraints.ubx[3]}]")
        print(f"  v: [{self.solver.acados_ocp.constraints.lbx[4]}, {self.solver.acados_ocp.constraints.ubx[4]}]")
        
        print("\nDEBUG: Input constraints:")
        print(f"  delta: [{self.solver.acados_ocp.constraints.lbu[0]}, {self.solver.acados_ocp.constraints.ubu[0]}]")
        print(f"  a: [{self.solver.acados_ocp.constraints.lbu[1]}, {self.solver.acados_ocp.constraints.ubu[1]}]")
        
        # Print actual constraint arrays and initial state before solve
        print("\n=== DEBUG: Constraints before solve ===")
        print("lbx:", self.solver.acados_ocp.constraints.lbx)
        print("ubx:", self.solver.acados_ocp.constraints.ubx)
        print("lbu:", self.solver.acados_ocp.constraints.lbu)
        print("ubu:", self.solver.acados_ocp.constraints.ubu)
        print("Initial state:", current_state)

        # Set initial guess for optimization variables
        for i in range(self.N + 1):
            self.solver.set(i, "x", current_state)
        for i in range(self.N):
            self.solver.set(i, "u", np.zeros(2))
        
        # Set reference trajectory
        self._set_reference(reference_trajectory)
        
        # Debug print reference trajectory
        print("\nDEBUG: Reference trajectory:")
        print(f"Final desired progress: {reference_trajectory.get('u_values', reference_trajectory.get('s_values', []))[-1] if reference_trajectory.get('u_values', reference_trajectory.get('s_values', [])) else 'N/A'}")
        print(f"Reference velocities: {reference_trajectory.get('velocities', [10.0] * len(reference_trajectory.get('u_values', reference_trajectory.get('s_values', []))))}")
        
        # Extract reference trajectory data
        # Handle both old 's_values' and new 'u_values' naming for backward compatibility
        trajectory_params = reference_trajectory.get('u_values', reference_trajectory.get('s_values', []))
        velocities = reference_trajectory.get('velocities', [10.0] * len(trajectory_params))
        
        # Set reference for each node in the horizon
        for i in range(self.N + 1):
            if i < len(trajectory_params):
                # Set reference state: [s, u, e_y, e_ψ, v]
                self.solver.set(i, "yref", [
                    trajectory_params[i],  # s (arc-length progress)
                    trajectory_params[i],  # u (chord-length parameter) - approximation
                    0.0,                   # e_y (lateral error)
                    0.0,                   # e_ψ (heading error)
                    velocities[i] if i < len(velocities) else 10.0  # v (velocity)
                ])
            else:
                # Use final values for remaining nodes
                self.solver.set(i, "yref", [
                    trajectory_params[-1] if trajectory_params else 0.0,
                    trajectory_params[-1] if trajectory_params else 0.0,
                    0.0,
                    0.0,
                    velocities[-1] if velocities else 10.0
                ])
        
        # Set terminal reference
        if trajectory_params:
            self.solver.set(self.N, "yref_e", [
                trajectory_params[-1],
                trajectory_params[-1],
                0.0,
                0.0,
                velocities[-1] if velocities else 10.0
            ])
        
        # Print initial guess for optimization variables
        print("\nDEBUG: Initial guess for optimization variables:")
        for i in range(self.N):
            x_guess = self.solver.get(i, "x")
            u_guess = self.solver.get(i, "u")
            print(f"Step {i}:")
            print(f"  x_guess: {x_guess}")
            print(f"  u_guess: {u_guess}")
        
        # Solve optimization problem
        status = self.solver.solve()
        
        if status != 0:
            print(f"\nMPC solver failed with status {status}")
            print("Cost function weights:")
            print(f"Q (state weights):\n{self.Q}")
            print(f"R (input weights):\n{self.R}")
            print(f"Q_terminal (terminal weights):\n{self.Q_terminal}")
        
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
    
    def _set_reference(self, reference_trajectory: Dict):
        """Set reference trajectory for the optimization problem."""
        # This method is now handled inline in the solve method
        pass
    
    def update_waypoints(self, waypoints: np.ndarray):
        """Update waypoints and recompute spline parameters."""
        if self.spline_dynamics is None:
            raise RuntimeError("Spline path must be set before updating waypoints")
        
        # Update spline parameters
        self.spline_dynamics.update_waypoints(waypoints)
        
        # Update parameters in solver
        self.spline_dynamics.update_spline_parameters(self.solver)
    
    def set_weights(self, Q: np.ndarray, R: np.ndarray, Q_terminal: Optional[np.ndarray] = None):
        """
        Update cost function weights.
        
        Args:
            Q: State weights [s, u, e_y, e_ψ, v]
            R: Input weights [delta, a]
            Q_terminal: Terminal state weights (optional)
        """
        self.Q = Q
        self.R = R
        if Q_terminal is not None:
            self.Q_terminal = Q_terminal
        else:
            self.Q_terminal = 2 * Q
        
        # Note: Weights must be set during OCP setup
        if self.solver is not None:
            print("Warning: Weights can only be set during solver initialization.")
            print("To change weights, recreate the MPC controller.")
    
    def get_prediction_horizon(self) -> float:
        """Get prediction horizon in seconds."""
        return self.prediction_horizon
    
    def get_time_step(self) -> float:
        """Get time step."""
        return self.dt 

    def update_spline_parameters(self, solver):
        """Update spline parameters in the acados solver."""
        # Get current parameter values
        current_params = solver.get(0, "p")
        
        print("\n=== DEBUG: Parameter vector consistency ===")
        print("Current param length:", len(current_params))
        print("New param length:", len(new_params))
        print("First 10 params:", new_params[:10]) 