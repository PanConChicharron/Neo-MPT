# author: Arjun Jagdish Ram
import numpy as np
import scipy.linalg
import time

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from MPC_race_cars_simplified.bicycle_model_spatial_with_body_points import bicycle_model_spatial
from Utils.clothoid_spline import ClothoidSpline

class PathTrackingMPCSpatialWithBodyPoints:
    def __init__(self, Tf, N, n_points, lf, lr, w, front_overhang, rear_overhang, left_overhang, right_overhang):
        self.Tf = Tf
        self.N = N
        self.n_points = n_points

        self.lf = lf
        self.lr = lr
        self.w = w
        self.front_overhang = front_overhang
        self.rear_overhang = rear_overhang
        self.left_overhang = left_overhang
        self.right_overhang = right_overhang

        self.constraint, self.model, self.acados_solver = self.acados_settings(lf, lr, w, front_overhang, rear_overhang, left_overhang, right_overhang)

    def acados_settings(self, lf, lr, w, front_overhang, rear_overhang, left_overhang, right_overhang):
        # create render arguments
        ocp = AcadosOcp()

        # export model
        model, constraint = bicycle_model_spatial(self.n_points, lf, lr, w, front_overhang, rear_overhang, left_overhang, right_overhang)

        # define acados ODE
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl_expr
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.name = model.name
        ocp.model = model_ac

        # dimensions
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx

        # discretization
        ocp.solver_options.N_horizon = self.N

        # set cost
        Q = np.diag([1e-2, 1e-2])

        R = np.eye(nu)
        R[0, 0] = 2e-1

        Qe = 5*Q

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        unscale = self.N / self.Tf

        ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Qe / unscale

        Vx = np.zeros((ny, nx))
        Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx = Vx

        Vu = np.zeros((ny, nu))
        Vu[ny-1, 0] = 1.0
        ocp.cost.Vu = Vu

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx_e = Vx_e

        # set initial references
        ocp.cost.yref = np.array([0, 0, 0.])
        ocp.cost.yref_e = np.array([0, 0])

        # setting constraints
        ocp.constraints.lbx = np.array([
            model.eY_min + self.w/2 + self.right_overhang,
        ])
        ocp.constraints.ubx = np.array([
            model.eY_max - self.w/2 - self.left_overhang,
        ])
        ocp.constraints.idxbx = np.array([1])

        ocp.constraints.lbu = np.array([
            model.delta_min,
        ])
        ocp.constraints.ubu = np.array([
            model.delta_max,
        ])
        ocp.constraints.idxbu = np.array([0])

        # set initial condition
        ocp.constraints.x0 = np.zeros(nx)
        ocp.parameter_values = np.zeros(model.p.shape[0])

        # set QP solver and integration
        ocp.solver_options.tf = self.Tf
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.num_steps = 1
        ocp.solver_options.nlp_solver_max_iter = 20
        ocp.solver_options.tol = 1e-4

        # create solver
        acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

        return constraint, model, acados_solver

    def get_optimised_steering(self, x0, clothoid_spline: ClothoidSpline):
        # load model
        Sf = clothoid_spline.pathlength
        constraint, model, acados_solver = self.constraint, self.model, self.acados_solver

        # dimensions
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu

        # initialize data structs
        simX = np.zeros((self.N, nx))
        simU = np.zeros((self.N, nu))
        s0 = 0
        tcomp_sum = 0
        tcomp_max = 0

        acados_solver.set(0, "lbx", x0)
        acados_solver.set(0, "ubx", x0)

        v_ref = 5.0

        # Extract sub-spline for the current position
        sub_knots, sub_coefficients = clothoid_spline.get_sub_spline_knots_and_coefficients_from_window_size(s0, self.n_points)

        sref = clothoid_spline.pathlength
        for j in range(self.N):
            # Ensure all arrays are properly shaped before concatenation
            s_interp = np.array([s0 + (sref - s0) * j / self.N])  # Convert scalar to 1D array
            parameters = np.concatenate((s_interp, sub_knots, sub_coefficients.flatten()), axis=0)
            yref = np.array([0, 0, 0.])
            acados_solver.set(j, "yref", yref)
            acados_solver.set(j, "p", parameters)
        yref_N = np.array([0, 0])
        acados_solver.set(self.N, "yref", yref_N)

        # solve ocp
        t = time.time()

        status = acados_solver.solve()
        if status != 0:
            print("acados returned status {} in closed loop iteration.".format(status))

        elapsed = time.time() - t

        # manage timings
        tcomp_sum += elapsed
        if elapsed > tcomp_max:
            tcomp_max = elapsed

        # update initial condition
        x0 = acados_solver.get(1, "x")
        acados_solver.set(0, "lbx", x0)
        acados_solver.set(0, "ubx", x0)

        for idx in range(0, self.N):
            x = acados_solver.get(idx, "x")
            u = acados_solver.get(idx, "u")

            simX[idx, :] = x
            simU[idx, :] = u

        final_idx = int(clothoid_spline.pathlength/(Sf/self.N))

        simX = simX[:final_idx, :]
        simU = simU[:final_idx, :]

        return simX, simU, Sf, elapsed
