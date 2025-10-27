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

        # evaluate curvature spline at body point s position
        kappa_ref_s_i = substitute(symbolic_curvature_cubic_spline.get_symbolic_spline(), s_sym, s_i)

        # dynamics for body point s position
        ds_i_ds = (kappa_ref_s * eY - 1) / ((kappa_ref_s_i * eY_i - 1) * cos(eψ))
        ds_body_points_ds.append(ds_i_ds)

        # dynamics for body point eY position
        deY_i_ds = kappa*(kappa_ref_s*eY - 1)*(kappa_ref_s_i*eY_i - 1)/cos(eψ)
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
