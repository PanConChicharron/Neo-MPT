"""
MPC Race Cars Simplified Package

A Python package for Model Predictive Control (MPC) of race cars using simplified bicycle models.
This package provides tools for trajectory optimization, path tracking, and vehicle dynamics simulation.

Main Components:
- bicycle_model: Basic bicycle model with full track spline interpolation
- bicycle_model_moving_window: Bicycle model with moving window B-spline approach
- acados_settings: ACADOS solver configuration for MPC
- tracks: Track data and utilities
- plotFcn: Plotting utilities for simulation results
- time2spatial: Coordinate transformation utilities
"""

__version__ = "1.0.0"
__author__ = "Daniel Kloeser (original), Modified for moving window approach"
__license__ = "2-Clause BSD License"

# Import main modules
from . import bicycle_model
from . import path_tracking_mpc
from . import plotFcn
from . import time2spatial
from . import tracks

# Import commonly used functions
from .bicycle_model import bicycle_model

__all__ = [
    'bicycle_model',
    'MovingWindowBicycleModel', 
    'bicycle_model_moving_window',
    'acados_settings',
    'path_tracking_mpc',
    'plotFcn',
    'time2spatial',
    'tracks'
] 