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

from casadi import *
from MPC_race_cars_simplified.tracks.readDataFcn import getTrack
import numpy as np


class MovingWindowBicycleModel:
    """
    Bicycle model with moving window B-spline interpolation for track curvature.
    
    This class implements a moving window approach where only a fixed-length
    segment of the track is used for B-spline interpolation, and the window
    shifts as the vehicle progresses along the track.
    """
    
    def __init__(self, track="LMS_Track.txt", window_length=2.0, overlap=0.5):
        """
        Initialize the moving window bicycle model.
        
        Args:
            track: Track file name
            window_length: Length of the moving window in meters
            overlap: Overlap between consecutive windows in meters
        """
        self.track_file = track
        self.window_length = window_length
        self.overlap = overlap
        
        # Load full track data
        [self.s0_full, _, _, _, self.kapparef_full] = getTrack(track)
        self.pathlength = self.s0_full[-1]
        
        # Initialize window parameters
        self.current_window_start = 0.0
        self.current_window_end = window_length
        self.kapparef_s = None
        
        # Create initial window
        self._update_window(0.0)
    
    def _update_window(self, current_s):
        """
        Update the moving window based on current position.
        
        Args:
            current_s: Current arc length position along the track
        """
        # Handle track wrapping (circular track)
        current_s = current_s % self.pathlength
        
        # Determine if we need to update the window
        window_center = (self.current_window_start + self.current_window_end) / 2
        if abs(current_s - window_center) > (self.window_length - self.overlap) / 2:
            # Update window position
            self.current_window_start = current_s - self.window_length / 2
            self.current_window_end = current_s + self.window_length / 2
            
            # Handle track wrapping for window boundaries
            if self.current_window_start < 0:
                # Window extends before track start, wrap to end
                s0_before = self.s0_full + self.pathlength
                kappa_before = self.kapparef_full
                
                # Get points before current position
                mask_before = (s0_before >= self.current_window_start) & (s0_before <= self.pathlength)
                s0_window_before = s0_before[mask_before]
                kappa_window_before = kappa_before[mask_before]
                
                # Get points after track start
                mask_after = (self.s0_full >= 0) & (self.s0_full <= self.current_window_end)
                s0_window_after = self.s0_full[mask_after]
                kappa_window_after = self.kapparef_full[mask_after]
                
                # Combine and sort
                s0_window = np.concatenate([s0_window_before, s0_window_after])
                kappa_window = np.concatenate([kappa_window_before, kappa_window_after])
                
                # Sort by s0 values and remove duplicates
                sort_idx = np.argsort(s0_window)
                s0_window = s0_window[sort_idx]
                kappa_window = kappa_window[sort_idx]
                
                # Remove duplicates (keep first occurrence)
                unique_mask = np.concatenate([[True], np.diff(s0_window) > 1e-10])
                s0_window = s0_window[unique_mask]
                kappa_window = kappa_window[unique_mask]
                
            elif self.current_window_end > self.pathlength:
                # Window extends beyond track end, wrap to beginning
                s0_after = self.s0_full - self.pathlength
                kappa_after = self.kapparef_full
                
                # Get points before track end
                mask_before = (self.s0_full >= self.current_window_start) & (self.s0_full <= self.pathlength)
                s0_window_before = self.s0_full[mask_before]
                kappa_window_before = self.kapparef_full[mask_before]
                
                # Get points after track end
                mask_after = (s0_after >= 0) & (s0_after <= self.current_window_end - self.pathlength)
                s0_window_after = s0_after[mask_after]
                kappa_window_after = kappa_after[mask_after]
                
                # Combine and sort
                s0_window = np.concatenate([s0_window_before, s0_window_after])
                kappa_window = np.concatenate([kappa_window_before, kappa_window_after])
                
                # Sort by s0 values and remove duplicates
                sort_idx = np.argsort(s0_window)
                s0_window = s0_window[sort_idx]
                kappa_window = kappa_window[sort_idx]
                
                # Remove duplicates (keep first occurrence)
                unique_mask = np.concatenate([[True], np.diff(s0_window) > 1e-10])
                s0_window = s0_window[unique_mask]
                kappa_window = kappa_window[unique_mask]
                
            else:
                # Window is within track bounds
                mask = (self.s0_full >= self.current_window_start) & (self.s0_full <= self.current_window_end)
                s0_window = self.s0_full[mask]
                kappa_window = self.kapparef_full[mask]
            
            # Create new B-spline interpolant for the window
            if len(s0_window) > 3:  # Need at least 4 points for cubic B-spline
                # Ensure strictly increasing grid points
                if np.all(np.diff(s0_window) > 0):
                    self.kapparef_s = interpolant("kapparef_s", "bspline", [s0_window], kappa_window)
                else:
                    # Fallback to linear interpolation if grid points are not strictly increasing
                    self.kapparef_s = interpolant("kapparef_s", "linear", [s0_window], kappa_window)
            else:
                # Fallback to linear interpolation if not enough points
                self.kapparef_s = interpolant("kapparef_s", "linear", [s0_window], kappa_window)
    
    def get_model(self, current_s=None):
        """
        Get the bicycle model with current window spline.
        
        Args:
            current_s: Current arc length position (optional, for window update)
            
        Returns:
            model, constraint: Model and constraint objects
        """
        # Update window if current_s is provided
        if current_s is not None:
            self._update_window(current_s)
        
        # define structs
        constraint = types.SimpleNamespace()
        model = types.SimpleNamespace()

        model_name = "Spatialbicycle_model_moving_window"

        ## Race car parameters
        m = 0.043
        lf = 0.02
        lr = 0.020

        ## CasADi Model
        # set up states & controls
        s = MX.sym("s")
        eY = MX.sym("eY")
        e_ψ = MX.sym("e_ψ")
        v = MX.sym("v")
        x = vertcat(s, eY, e_ψ, v)

        # controls
        a = MX.sym("a")
        delta = MX.sym("delta")
        u = vertcat(a, delta)

        # xdot
        sdot = MX.sym("sdot")
        eYdot = MX.sym("eYdot")
        e_ψdot = MX.sym("e_ψdot")
        vdot = MX.sym("vdot")
        xdot = vertcat(sdot, eYdot, e_ψdot, vdot)

        # algebraic variables
        z = vertcat([])

        # parameters
        p = vertcat([])

        beta = atan(lr * tan(delta) / (lf + lr))
        kappa = sin(beta) / (lf + lr)

        # dynamics - use the current window spline
        if self.kapparef_s is not None:
            sdot = (v * cos(e_ψ + beta)) / (1 - self.kapparef_s(s) * eY)
        else:
            # Fallback if no spline available
            sdot = v * cos(e_ψ + beta)
            
        f_expl = vertcat(
            sdot,
            v * sin(e_ψ + beta),
            v * kappa - (self.kapparef_s(s) if self.kapparef_s is not None else 0) * sdot,
            a,
        )

        # constraint on forces
        a_lat = v * v * kappa
        a_long = a

        # Model bounds
        model.eY_min = -0.12  # width of the track [m]
        model.eY_max = 0.12  # width of the track [m]

        # input bounds
        model.a_min = -4.0
        model.a_max = 4.0

        model.delta_min = -0.40  # minimum steering angle [rad]
        model.delta_max = 0.40  # maximum steering angle [rad]

        # nonlinear constraint
        constraint.alat_min = -4  # maximum lateral force [m/s^2]
        constraint.alat_max = 4  # maximum lateral force [m/s^1]

        # Define initial conditions
        model.x0 = np.array([0, 0, 0, 0])

        # define constraints struct
        constraint.alat = Function("a_lat", [x, u], [a_lat])
        constraint.pathlength = self.pathlength
        constraint.expr = vertcat(a_lat)

        # Define model struct
        params = types.SimpleNamespace()
        params.m = m
        params.lf = lf
        params.lr = lr
        model.f_impl_expr = xdot - f_expl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.z = z
        model.p = p
        model.name = model_name
        model.params = params
        return model, constraint


def bicycle_model_moving_window(track="LMS_Track.txt", window_length=2.0, overlap=0.5, current_s=None):
    """
    Factory function to create a moving window bicycle model.
    
    Args:
        track: Track file name
        window_length: Length of the moving window in meters
        overlap: Overlap between consecutive windows in meters
        current_s: Current arc length position for window initialization
        
    Returns:
        model, constraint: Model and constraint objects
    """
    model_instance = MovingWindowBicycleModel(track, window_length, overlap)
    return model_instance.get_model(current_s) 