import numpy as np

def create_test_path():
    """Create a test path with waypoints."""
    # Create a simple path with clear progression
    waypoints = np.array([
        [0, 0],
        [5, 2],
        [10, 3],
        [15, 1],
        [20, -2],
        [25, 0],
        [30, 3]
    ])
    return waypoints

def create_straight_line_path():
    """Create a simple straight-line test path for debugging boundary behavior."""
    # Create a straight line path - this should have ds/du = 1.0 everywhere
    waypoints = np.array([
        [0, 0],
        [5, 0],
        [10, 0],
        [15, 0],
        [20, 0],
        [25, 0],
        [30, 0]
    ])
    return waypoints

def create_racetrack_path():
    #faulty
    """Create a race-track style pseudo-ellipse path with straight sections and curved ends."""
    # Race-track style pseudo-ellipse waypoints
    # Creates an oval with straight sections and curved ends
    waypoints = np.array([
        # Bottom straight section (start)
        [0, 0],
        [5, 0],
        [10, 0],
        [15, 0],
        [20, 0],
        
        # Right curved section (turn 1)
        [25, 2],
        [28, 5],
        [30, 8],
        [30, 12],
        
        # Top straight section
        [28, 15],
        [25, 16],
        [20, 16],
        [15, 16],
        [10, 16],
        [5, 16],
        [0, 16],
        
        # Left curved section (turn 2)
        [-3, 14],
        [-5, 10],
        [-5, 6],
        [-3, 2],
        
        # Return to start
        [0, 0]
    ])
    return waypoints

def create_challenging_track():
    """Create a longer, more challenging race track with multiple complex turns."""
    # Complex race track with chicanes, hairpins, and varying radius turns
    waypoints = np.array([
        # Start/finish straight
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
        
        # Turn 1: Fast right-hander
        [40, -3],
        [50, -8],
        [60, -10],
        [70, -8],
        
        # Short straight into chicane
        [80, -5],
        [90, -2],
        
        # Chicane section (S-curves)
        [95, 2],
        [100, 6],
        [105, 4],
        [110, 0],
        [115, -4],
        [120, -2],
        
        # Long straight
        [130, 0],
        [140, 2],
        [150, 4],
        [160, 6],
        
        # Hairpin turn (180-degree)
        [165, 10],
        [168, 15],
        [170, 20],
        [168, 25],
        [165, 30],
        [160, 32],
        [150, 30],
        [140, 28],
        [130, 26],
        
        # Back straight with slight curves
        [120, 24],
        [110, 22],
        [100, 20],
        [90, 18],
        [80, 16],
        
        # Complex turn sequence
        [70, 12],
        [60, 8],
        [50, 6],
        [40, 8],
        [30, 12],
        [20, 14],
        [10, 12],
        [5, 8],
        [2, 4],
        
        # # Return to start
        # [0, 0]
    ])
    
    return waypoints 