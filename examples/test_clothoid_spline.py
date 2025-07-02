import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.symbolic_cubic_spline import SymbolicCubicSpline

clothoid_spline = SymbolicCubicSpline("../../MPC_race_cars_simplified/tracks/LMS_Track.txt")

print("knots: ", clothoid_spline.get_knots())
print("coefficients: ", clothoid_spline.get_coefficients())

print("spline: ", clothoid_spline.get_spline())

sub_knots, sub_coefficients = clothoid_spline.get_sub_spline_knots_and_coefficients_from_window_size(0.0, 10)
print("sub_knots: ", sub_knots)
print("sub_coefficients: ", sub_coefficients)