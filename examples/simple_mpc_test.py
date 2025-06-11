#!/usr/bin/env python3
"""
Simple MPC test to verify acados setup.
"""

import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

def test_simple_mpc():
    """Test a simple MPC setup."""
    print("Testing simple MPC setup...")
    
    # Create OCP
    ocp = AcadosOcp()
    
    # Model
    x = ca.SX.sym('x', 2)  # Simple 2D integrator
    u = ca.SX.sym('u', 2)  # 2D control input
    
    # Simple dynamics: x_dot = u
    f_expl = u
    
    ocp.model.f_expl_expr = f_expl
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.name = 'simple_integrator'
    
    # Dimensions - set this FIRST
    ocp.solver_options.N_horizon = 10  # Use N_horizon instead of dims.N
    ocp.dims.ny = 2    # output dimension
    ocp.dims.ny_e = 2  # terminal output dimension
    
    # Cost
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    # Output selection matrices - use correct names!
    ocp.cost.Vx = np.eye(2)      # Select states (2x2)
    ocp.cost.Vu = np.zeros((2, 2))  # No inputs in output (2x2)
    ocp.cost.W = np.eye(2)       # Cost weights (2x2)
    
    # Terminal cost
    ocp.cost.Vx_e = np.eye(2)    # Terminal state selection (2x2)
    ocp.cost.W_e = np.eye(2)     # Terminal weights (2x2)
    
    # Initialize reference values to avoid dimension errors
    ocp.cost.yref = np.zeros(2)    # 2-dimensional reference
    ocp.cost.yref_e = np.zeros(2)  # 2-dimensional terminal reference
    
    # Constraints
    ocp.constraints.lbu = np.array([-1, -1])
    ocp.constraints.ubu = np.array([1, 1])
    ocp.constraints.idxbu = np.array([0, 1])
    
    ocp.constraints.x0 = np.array([1.0, 1.0])
    
    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.tf = 1.0
    
    try:
        solver = AcadosOcpSolver(ocp, json_file='simple_test.json')
        print("✓ Acados solver created successfully!")
        
        # Test solve
        N = 10  # horizon length
        for i in range(N):
            solver.set(i, "yref", np.zeros(2))
        solver.set(N, "yref", np.zeros(2))
        
        status = solver.solve()
        print(f"✓ Solver status: {status}")
        
        if status == 0:
            u_opt = solver.get(0, "u")
            print(f"✓ Optimal control: {u_opt}")
            return True
        else:
            print("✗ Solver failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_mpc()
    if success:
        print("\n✓ Acados is working correctly!")
    else:
        print("\n✗ Acados setup has issues") 