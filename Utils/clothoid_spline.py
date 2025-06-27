import numpy as np
from scipy.interpolate import BSpline

from MPC_race_cars_simplified.tracks.readDataFcn import getTrack

class ClothoidSpline:
    def __init__(self, track_file: str):
        self.track_file = track_file
        [s0, _, _, _, kapparef] = getTrack(track_file)
        degree = 3  # Cubic B-spline
        n = len(kapparef)
        # For a clamped B-spline, repeat the first and last value degree+1 times
        knots = np.concatenate((
            np.repeat(s0[0], degree+1),
            s0[1:-(degree)],
            np.repeat(s0[-1], degree+1)
        ))
        coeffs = kapparef
        print(f"len(s0)={len(s0)}, len(kapparef)={len(kapparef)}, len(knots)={len(knots)}, degree={degree}")
        print(f"s0[0]={s0[0]}, s0[-1]={s0[-1]}")
        print(f"knots[:10]={knots[:10]}")
        print(f"knots[-10:]={knots[-10:]}")
        self.spline = BSpline(knots, coeffs, degree)
        self.knots = self.spline.t
        self.coefficients = self.spline.c
        self.degree = self.spline.k

    def get_spline(self):
        return self.spline
    
    def get_knots(self):
        return self.knots
    
    def get_coefficients(self):
        return self.coefficients
    
    def get_sub_spline_knots_and_coefficients_from_window_size(self, s, window_size):
        """Extract a B-spline sub-spline for the given window size."""
        # For B-spline, we need to extract a proper sub-spline
        # Find the closest knot to s
        closest_knot_idx = np.argmin(np.abs(self.knots - s))
        
        # For a B-spline with window_size coefficients, we need:
        # - window_size coefficients
        # - window_size + degree + 1 knots
        
        # Extract coefficients (window_size of them)
        start_coeff = max(0, closest_knot_idx - self.degree)
        end_coeff = min(len(self.coefficients), start_coeff + window_size)
        sub_coefficients = self.coefficients[start_coeff:end_coeff]
        
        # Pad coefficients if needed
        if len(sub_coefficients) < window_size:
            sub_coefficients = np.append(sub_coefficients, np.zeros(window_size - len(sub_coefficients)))
        
        # Extract knots (window_size + degree + 1 of them)
        start_knot = max(0, start_coeff)
        end_knot = min(len(self.knots), start_knot + window_size + self.degree + 1)
        sub_knots = self.knots[start_knot:end_knot]
        
        # Pad knots if needed
        if len(sub_knots) < window_size + self.degree + 1:
            sub_knots = np.append(sub_knots, np.ones(window_size + self.degree + 1 - len(sub_knots)) * 0.)
        
        return sub_knots, sub_coefficients


