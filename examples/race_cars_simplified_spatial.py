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

track = "../../MPC_race_cars_simplified/tracks/LMS_Track.txt"

Tf = 10.0  # prediction horizon
N = 50  # number of discretization steps
T = 5000.0  # maximum simulation time[s]

clothoid_spline = ClothoidSpline(track)

Tf = clothoid_spline.pathlength  # pathlength divided by prediction horizon

dt = T/N

num_points = 50
# load model
constraint, model, acados_solver = acados_settings(Tf, N, num_points)

# dimensions
nx = model.x.rows()
nu = model.u.rows()
ny = nx + nu
Nsim = int(T * N / Tf)


# Define initial conditions
x0 = np.array([0, 0])  # Start with v=1.0 m/s

# initialize data structs
simX = np.zeros((Nsim, nx))
simU = np.zeros((Nsim, nu))
s0 = 0
tcomp_sum = 0
tcomp_max = 0

acados_solver.set(0, "lbx", x0)
acados_solver.set(0, "ubx", x0)

v_ref = 5.0

# simulate
for i in range(Nsim):
    # update reference
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

    # get solution
    x0 = acados_solver.get(0, "x")
    u0 = acados_solver.get(0, "u")

    # for i in range(acados_solver.N + 1):  # N shooting nodes → N+1 stages
    #     lam = acados_solver.get(i, "lam")
    #     print(f"Stage {i} lam: {lam}")

    #     if i < acados_solver.N:
    #         pi = acados_solver.get(i, "pi")
    #         print(f"Stage {i} pi (dynamics multipliers): {pi}")

    for j in range(nx):
        simX[i, j] = x0[j]
    for j in range(nu):
        simU[i, j] = u0[j]

    # update initial condition
    x0 = acados_solver.get(1, "x")
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

    s0_prev = s0
    s0 = (i/N)*Tf
        
    print("s: {}, s_max: {}".format(s0, clothoid_spline.pathlength))

    # check if one lap is done and break and remove entries beyond
    if s0 > clothoid_spline.pathlength:
        # find where vehicle first crosses start line
        # import pdb; pdb.set_trace()
        N0 = np.where(np.diff(np.sign(simX[:, 0])))[0][0]
        Nsim = i - N0  # correct to final number of simulation steps for plotting
        simX = simX[N0:i, :]
        simU = simU[N0:i, :]
        break

# Plot Results
t = np.linspace(0.0, Nsim * Tf / N, Nsim)
plotRes(simX, simU, t)
plotTrackProj(simX, track, Tf, N)

# Print some stats
print("Average computation time: {}".format(tcomp_sum / Nsim))
print("Maximum computation time: {}".format(tcomp_max))
print("Lap time: {}s".format(Tf * Nsim / N))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
