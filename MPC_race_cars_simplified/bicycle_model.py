from casadi import *

from Utils.symbolic_cubic_spline import SymbolicCubicSpline


def bicycle_model(n_points=20):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "curvilinear_bicycle_model"

    # copy loop to beginning and end
    # s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    # kapparef = np.append(kapparef, kapparef[1:length])
    # s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    # kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    ## Race car parameters
    lf = 0.030
    lr = 0.010

    ## CasADi Model
    # set up states & controls
    s = SX.sym("s")
    eY = SX.sym("eY")
    e_ψ = SX.sym("e_ψ")
    v = SX.sym("v")
    x = vertcat(s, eY, e_ψ, v)
    
    symbolic_curvature_cubic_spline = SymbolicCubicSpline(n_points=n_points, u=s)
    kappa_ref_s = symbolic_curvature_cubic_spline.get_symbolic_spline()
    p = symbolic_curvature_cubic_spline.get_parameters()

    # controls
    a = SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)

    # xdot
    sdot = SX.sym("sdot")
    eYdot = SX.sym("eYdot")
    e_ψdot = SX.sym("e_ψdot")
    vdot = SX.sym("vdot")
    xdot = vertcat(sdot, eYdot, e_ψdot, vdot)

    beta = atan(lr * tan(delta) / (lf + lr))
    kappa = cos(beta) * tan(delta) / (lf + lr)

    # dynamics
    sdot = (v * cos(e_ψ + beta)) / (1 - kappa_ref_s * eY)
    f_expl = vertcat(
        sdot,
        v * sin(e_ψ + beta),
        v * kappa - kappa_ref_s * sdot,
        a,
    )

    # constraint on forces
    a_lat = v * v * kappa

    # Model bounds
    model.eY_min = -0.12  # width of the track [m]
    model.eY_max = 0.12  # width of the track [m]

    # input bounds
    model.a_min = -4.0
    model.a_max = 4.0

    model.delta_min = -np.pi/4  # minimum steering angle [rad]
    model.delta_max = np.pi/4  # maximum steering angle [rad]

    # nonlinear constraint
    constraint.alat_min = -4  # maximum lateral force [m/s^2]
    constraint.alat_max = 4  # maximum lateral force [m/s^1]

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.expr = vertcat(a_lat)

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
