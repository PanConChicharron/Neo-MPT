# author: Arjun Jagdish Ram

import time, os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MPC_race_cars_simplified.path_tracking_mpc_spatial import *
from MPC_race_cars_simplified.plotFcn import *
from MPC_race_cars_simplified.path_tracking_mpc_spatial import PathTrackingMPCSpatial
from Utils.clothoid_spline import ClothoidSpline
import matplotlib.pyplot as plt

"""
Example of the frc_racecars in simulation without obstacle avoidance:
This example is for the optimal racing of the frc race cars. The model is a simple bicycle model and the lateral acceleration is constraint in order to validate the model assumptions.
The simulation starts at s=-2m until one round is completed(s=8.71m). The beginning is cut in the final plots to simulate a 'warm start'. 
"""

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


    Sf = clothoid_spline.pathlength  # pathlength
    num_points = N

    path_tracking_mpc_spatial = PathTrackingMPCSpatial(Sf, N, num_points)
    simX, simU, Sf, elapsed = path_tracking_mpc_spatial.get_optimised_steering(x0, clothoid_spline)

    plot_result(Sf, simX, simU, track, N)

    # Print some stats
    print("Computation time: {}s".format(elapsed))
    print("Frequency: {}Hz".format(1/elapsed))

if __name__ == "__main__":
    main()