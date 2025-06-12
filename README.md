# Spline-Based MPC Tracking Project

This project implements a two-part system for trajectory tracking using cubic splines and Model Predictive Control (MPC).

## Components

### 1. Spline Interpolation Module (`spline_fit/`)
- Cubic spline interpolation for smooth trajectory generation
- Curvilinear coordinate system support
- Path parameterization and derivatives

### 2. MPC Controller (`mpc_controller/`)
- Acados-based MPC implementation
- Tracks spline-generated reference trajectories
- Real-time optimization for trajectory following

## Installation

```bash
pip install -r requirements.txt
```

Note: For acados installation, please follow the official acados installation guide at https://docs.acados.org/

## Usage

```python
from spline_fit import ChordLengthParametricSpline2D
from mpc_controller import MPCController
import numpy as np

# Create waypoints
waypoints = np.array([[0, 0], [5, 2], [10, 1], [15, 3]])
spline = ChordLengthParametricSpline2D(waypoints)
trajectory = spline.generate_trajectory()

# Track with MPC
controller = MPCController()
control_input = controller.solve(current_state, trajectory)
```

## Project Structure

```
├── spline_fit/           # Spline interpolation module
├── mpc_controller/       # MPC tracking controller
├── examples/            # Usage examples
└── tests/              # Unit tests
``` 