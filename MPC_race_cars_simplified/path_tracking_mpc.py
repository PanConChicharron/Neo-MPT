#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from MPC_race_cars_simplified.bicycle_model import bicycle_model
import scipy.linalg
import numpy as np


def acados_settings(Tf, N, n_points):
    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint = bicycle_model(n_points)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.p = model.p
    model_ac.name = model.name

    # define constraint
    model_ac.con_h_expr = constraint.expr

    ocp.model = model_ac

    # dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # discretization
    ocp.solver_options.N_horizon = N

    # set cost
    #           (s, eY  , e_ψ , v)
    Q = np.diag([1, 1e-2, 5e-2, 2e-3])

    R = np.eye(nu)
    R[0, 0] = 1e-3
    R[1, 1] = 1e-2

    Qe = 5*Q

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    unscale = N / Tf

    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe / unscale

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[4, 0] = 1.0
    Vu[5, 1] = 1.0
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # ocp.cost.zl = 100 * np.ones((ns,))
    # ocp.cost.zu = 100 * np.ones((ns,))
    # ocp.cost.Zl = 1 * np.ones((ns,))
    # ocp.cost.Zu = 1 * np.ones((ns,))

    # set initial references
    ocp.cost.yref = np.array([1, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0])

    # setting constraints
    ocp.constraints.lbx = np.array([
        model.eY_min,
    ])
    ocp.constraints.ubx = np.array([
        model.eY_max,
    ])
    ocp.constraints.idxbx = np.array([1])

    ocp.constraints.lbu = np.array([
        model.a_min,
        model.delta_min,
    ])
    ocp.constraints.ubu = np.array([
        model.a_max,
        model.delta_max,
    ])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lh = np.array(
        [
            constraint.alat_min,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            constraint.alat_max,
        ]
    )

    # set initial condition
    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(model.p.shape[0])

    # set QP solver and integration
    ocp.solver_options.tf = Tf
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
