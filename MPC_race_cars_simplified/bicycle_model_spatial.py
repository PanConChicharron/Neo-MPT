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

from casadi import *

from Utils.symbolic_cubic_spline import SymbolicCubicSpline


def bicycle_model_spatial(n_points=20, lf=2.79, lr=0.0, w=1.64, front_overhang=1.0, rear_overhang=1.1, left_overhang=0.128, right_overhang=0.128):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "curvilinear_bicycle_model_spatial"

    L = lf + lr

    corner_points = np.array([
        [lf + front_overhang, w +right_overhang],  # front right
        [lf + front_overhang, -w - left_overhang],  # front left
        [-rear_overhang, -w - left_overhang],  # rear left
        [-rear_overhang, w + right_overhang],  # rear right
    ])

    ## CasADi Model
    # set up states & controls
    eY = SX.sym("eY")
    eψ = SX.sym("eψ")

    eY_corners = []

    # corner points
    for idx in range(0, len(corner_points), 1):
        eY_idx = SX.sym(f"eY_{idx}")

        eY_corners.append(eY_idx)
    
    x = vertcat(eY, eψ, *eY_corners)

    print(x)

    s_sym = SX.sym("s")  # symbolic independent variable
    symbolic_curvature_cubic_spline = SymbolicCubicSpline(n_points=n_points, u=s_sym)
    kappa_ref_s = symbolic_curvature_cubic_spline.get_symbolic_spline()
    p = vertcat(s_sym, symbolic_curvature_cubic_spline.get_parameters())

    # controls
    delta = SX.sym("delta")
    u = vertcat(delta)

    # xdot
    eYdot = SX.sym("eYdot")
    eψdot = SX.sym("eψdot")

    eYdot_corners = []

    for idx in range(0, len(corner_points), 1):
        eYdot_idx = SX.sym(f"eYdot_{idx}")
        eYdot_corners.append(eYdot_idx)
    xdot = vertcat(eYdot, eψdot, *eYdot_corners)

    beta = atan(lr * tan(delta) / (lf + lr))
    kappa = cos(beta) * tan(delta) / (lf + lr)

    # dynamics
    deY_ds = tan(eψ + beta) *(1-kappa_ref_s * eY)
    deψ_ds = (kappa - kappa_ref_s)*(1 - kappa_ref_s * eY) / cos(eψ)

    deY_ds_corners = []

    #corner point dynamics
    for corner_point, idx in zip(corner_points, range(0, len(corner_points), 1)):
        deY_idx = deY_ds +(-sin(eψ) * corner_point[1] + cos(eψ) * corner_point[0]) * deψ_ds

        deY_ds_corners.append(deY_idx)

    f_expl = vertcat(
        deY_ds,
        deψ_ds,
        *deY_ds_corners
    )

    # Model bounds
    model.eY_min = -1.5  # width of the track [m]
    model.eY_max = 1.5  # width of the track [m]

    # input bounds
    model.delta_min = -np.pi/4  # minimum steering angle [rad]
    model.delta_max = np.pi/4  # maximum steering angle [rad]

    # Define model struct
    params = types.SimpleNamespace()
    params.lf = lf
    params.lr = lr
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name
    model.params = params
    return model, constraint
