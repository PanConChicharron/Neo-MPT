from casadi import *

from Utils.symbolic_cubic_spline import SymbolicCubicSpline


def bicycle_model_spatial(n_points, lf, lr, w, front_overhang, rear_overhang, left_overhang, right_overhang):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "curvilinear_bicycle_model_spatial"

    L = lf + lr

    corner_points = np.array([
        [lf + front_overhang, w/2 +right_overhang],  # front left
        [lf + front_overhang, -w/2 - left_overhang],  # front right
        [-rear_overhang, -w/2 - left_overhang],  # rear right
        [-rear_overhang, w/2 + right_overhang],  # rear left
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
    
    x = vertcat(eY,
                eψ, 
        #        *eY_corners,
        )

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
    xdot = vertcat(
        eYdot,
        eψdot, 
    #    *eYdot_corners
    )

    beta = atan(lr * tan(delta) / (lf + lr))
    kappa = cos(beta) * tan(delta) / (lf + lr)

    # dynamics
    deY_ds = tan(eψ + beta) *(1-kappa_ref_s * eY)
    deψ_ds = kappa*(1 - kappa_ref_s * eY) / cos(eψ) - kappa_ref_s

    deY_ds_corners = []

    #corner point dynamics
    for corner_point, idx in zip(corner_points, range(0, len(corner_points), 1)):
        deY_idx = deY_ds +(-sin(eψ) * corner_point[1] + cos(eψ) * corner_point[0]) * deψ_ds

        deY_ds_corners.append(deY_idx)

    f_expl = vertcat(
        deY_ds,
        deψ_ds,
        #*deY_ds_corners
    )

    # Model bounds
    model.eY_min = -2.0  # width of the track [m]
    model.eY_max =  2.0  # width of the track [m]

    # input bounds
    model.delta_min = -0.7  # minimum steering angle [rad]
    model.delta_max = 0.7  # maximum steering angle [rad]

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
