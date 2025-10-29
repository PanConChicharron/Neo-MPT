from casadi import *

from Utils.symbolic_cubic_spline import SymbolicCubicSpline


def bicycle_model_spatial_with_body_points(n_points, num_body_points, lf, lr, w, front_overhang, rear_overhang, left_overhang, right_overhang):
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

    s_body_points = SX.sym("s_body_points", num_body_points)
    eY_body_points = SX.sym("eY_body_points", num_body_points)
    
    x = vertcat(
        eY,
        eψ,
        s_body_points,
        eY_body_points,
    )

    x_body_points = SX.sym("x_body_points", num_body_points)
    y_body_points = SX.sym("y_body_points", num_body_points)

    s_sym = SX.sym("s")  # symbolic independent variable
    x_ref_s_symbolic_curvature_cubic_spline = SymbolicCubicSpline(n_points=n_points, u=s_sym)
    x_ref_s = x_ref_s_symbolic_curvature_cubic_spline.get_symbolic_spline()
    y_ref_s_symbolic_curvature_cubic_spline = SymbolicCubicSpline(n_points=n_points, u=s_sym)
    y_ref_s = y_ref_s_symbolic_curvature_cubic_spline.get_symbolic_spline()
    psi_ref_s_symbolic_curvature_cubic_spline = SymbolicCubicSpline(n_points=n_points, u=s_sym)
    psi_ref_s = psi_ref_s_symbolic_curvature_cubic_spline.get_symbolic_spline()
    kappa_ref_s_symbolic_curvature_cubic_spline = SymbolicCubicSpline(n_points=n_points, u=s_sym)
    kappa_ref_s = kappa_ref_s_symbolic_curvature_cubic_spline.get_symbolic_spline()
    
    p = vertcat(s_sym, x_ref_s_symbolic_curvature_cubic_spline.get_parameters(), y_ref_s_symbolic_curvature_cubic_spline.get_parameters(), psi_ref_s_symbolic_curvature_cubic_spline.get_parameters(), kappa_ref_s_symbolic_curvature_cubic_spline.get_parameters(), x_body_points, y_body_points)

    # controls
    delta = SX.sym("delta")
    u = vertcat(delta)

    # xdot
    eYdot = SX.sym("eYdot")
    eψdot = SX.sym("eψdot")

    sdot_body_points = SX.sym("sdot_body_points", num_body_points)
    eYdot_body_points = SX.sym("eYdot_body_points", num_body_points)

    xdot = vertcat(
        eYdot,
        eψdot,
        sdot_body_points,
        eYdot_body_points,
    )

    beta = atan(lr * tan(delta) / (lf + lr))
    kappa = cos(beta) * tan(delta) / (lf + lr)

    # dynamics
    deY_ds = tan(eψ + beta) *(1-kappa_ref_s * eY)
    deψ_ds = kappa*(1 - kappa_ref_s * eY) / cos(eψ) - kappa_ref_s

    ds_body_points_ds = []
    deY_body_points_ds = []

    for i in range(num_body_points):
        s_i = s_body_points[i]
        eY_i = eY_body_points[i]

        x_body = x_body_points[i]
        y_body = y_body_points[i]

        # evaluate curvature spline at body point s position
        x_ref_body_s_i = substitute(x_ref_s, s_sym, s_i)
        y_ref_body_s_i = substitute(y_ref_s, s_sym, s_i)
        psi_ref_body_s_i = substitute(psi_ref_s, s_sym, s_i)
        kappa_ref_s_i = substitute(kappa_ref_s, s_sym, s_i)

        # dynamics for body point s position
        ds_i_ds = -(kappa*(eY_i*cos(eψ + psi_ref_body_s_i) + (x_ref_body_s_i-x_body)*sin(eψ) + (y_ref_body_s_i-y_body)*cos(eψ) - cos(beta + eψ))*(kappa_ref_s*eY - 1)/((kappa_ref_s_i*eY_i - 1)*cos(beta + eψ)))
        ds_body_points_ds.append(ds_i_ds)

        # dynamics for body point eY position
        deY_i_ds = (kappa*(eY*sin(eψ + psi_ref_body_s_i) + x_body*cos(eψ) - y_body*sin(eψ) - x_ref_body_s_i*cos(eψ) + y_ref_body_s_i*sin(eψ)) - sin(beta + eψ))*(kappa_ref_s*eY - 1)/cos(beta + eψ)
        deY_body_points_ds.append(deY_i_ds)

    f_expl = vertcat(
        deY_ds,
        deψ_ds,
        *ds_body_points_ds,
        *deY_body_points_ds,
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
