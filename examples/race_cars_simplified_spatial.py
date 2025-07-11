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

N = 100  # number of discretization steps

clothoid_spline = ClothoidSpline(track)

Tf = clothoid_spline.pathlength  # pathlength

num_points = 50
# load model
constraint, model, acados_solver = acados_settings(Tf, N, num_points)

# dimensions
nx = model.x.rows()
nu = model.u.rows()
ny = nx + nu


# Define initial conditions
x0 = np.array([0.12, 0])  # Start with v=1.0 m/s

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

simX = simX[:N, :]
simU = simU[:N, :]

# Plot Results
t = np.linspace(0.0, Tf, np.shape(simX)[0])
plotRes(simX, simU, t)
plotTrackProj(simX, track, Tf, N)

# Print some stats
print("Computation time: {}s".format(elapsed))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
