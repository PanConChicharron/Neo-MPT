# author: Arjun Jagdish Ram

import time, os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MPC_race_cars_simplified.path_tracking_mpc_spatial import *
from MPC_race_cars_simplified.plotFcn import *
from Utils.clothoid_spline import ClothoidSpline
import matplotlib.pyplot as plt

"""
Example of the frc_racecars in simulation without obstacle avoidance:
This example is for the optimal racing of the frc race cars. The model is a simple bicycle model and the lateral acceleration is constraint in order to validate the model assumptions.
The simulation starts at s=-2m until one round is completed(s=8.71m). The beginning is cut in the final plots to simulate a 'warm start'. 
"""

def get_optimised_steering(N: int, x0, clothoid_spline: ClothoidSpline):
    Sf = clothoid_spline.pathlength  # pathlength

    num_points = N
    # load model
    constraint, model, acados_solver = acados_settings(Sf, N, num_points)

    # dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu

    # initialize data structs
    simX = np.zeros((N, nx))
    simU = np.zeros((N, nu))
    s0 = 0
    tcomp_sum = 0
    tcomp_max = 0

    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

    v_ref = 5.0

    # Extract sub-spline for the current position
    sub_knots, sub_coefficients = clothoid_spline.get_sub_spline_knots_and_coefficients_from_window_size(s0, num_points)
    sref = clothoid_spline.pathlength
    for j in range(N):
        # Ensure all arrays are properly shaped before concatenation
        s_interp = np.array([s0 + (sref - s0) * j / N])  # Convert scalar to 1D array
        parameters = np.concatenate((s_interp, sub_knots, sub_coefficients.flatten()), axis=0)
        yref = np.array([0, 0, 0])
        acados_solver.set(j, "yref", yref)
        acados_solver.set(j, "p", parameters)
    yref_N = np.array([0, 0])
    acados_solver.set(N, "yref", yref_N)

    # solve ocp
    t = time.time()

    status = acados_solver.solve()
    if status != 0:
        print("acados returned status {} in closed loop iteration {}.".format(status, i))

    elapsed = time.time() - t

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # update initial condition
    x0 = acados_solver.get(1, "x")
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

    for idx in range(0, N):
        x = acados_solver.get(idx, "x")
        u = acados_solver.get(idx, "u")

        simX[idx, :] = x
        simU[idx, :] = u

    final_idx = int(clothoid_spline.pathlength/(Sf/N))

    simX = simX[:final_idx, :]
    simU = simU[:final_idx, :]

    return simX, simU, Sf, elapsed

def plot_result(Sf, simX, simU, track, N):
    # Plot Results
    t = np.linspace(0.0, Sf, np.shape(simX)[0])
    plotRes(simX, simU, t)
    plotTrackProj(simX, track, Sf, N)

    plt.show()

def main():
    N = 100  # number of discretization steps
    track = "../../MPC_race_cars_simplified/tracks/LMS_Track.txt"
    [s0, _, _, _, kapparef] = getTrack(track)

    clothoid_spline = ClothoidSpline(s0, kapparef, N)

    # Define initial conditions
    x0 = np.array([0.08, np.pi/4])

    simX, simU, Sf, elapsed = get_optimised_steering(N, x0, clothoid_spline)

    plot_result(Sf, simX, simU, track, N)

    # Print some stats
    print("Computation time: {}s".format(elapsed))
    print("Frequency: {}Hz".format(1/elapsed))

if __name__ == "__main__":
    main()