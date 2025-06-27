"""
Generic symbolic cubic spline for optimization applications.
"""

import casadi as ca

class SymbolicCubicSpline:
    """Simple symbolic cubic spline class."""
    
    def __init__(self, n_points: int):
        """Initialize spline with n_points."""
        self.n_points = n_points
        self.n_segments = n_points - 1

        self.knots = ca.SX.sym('knots', n_points)
        self.coefficients_x = ca.SX.sym('coeffs_x', 4, (n_points-1))  # a,b,c,d for each segment
        self.coefficients_y = ca.SX.sym('coeffs_y', 4, (n_points-1))  # a,b,c,d for each segment

        self.x_val = 0
        self.y_val = 0
        self.dx_du = 0
        self.dy_du = 0
        self.d2x_du2 = 0
        self.d2y_du2 = 0

        self.u = ca.SX.sym('u')
        
        # Create piecewise symbolic spline evaluation
        for i in range(self.n_points-1):
            # Condition: knots[i] <= u < knots[i+1] (or u <= knots[i+1] for last segment)
            if i == 0:
                # For first segment, include u = knots[0]
                condition = ca.logic_and(self.u >= self.knots[i], self.u < self.knots[i+1])
            elif i == self.n_segments - 1:
                # For last segment, include u = knots[-1]
                condition = ca.logic_and(self.u >= self.knots[i], self.u <= self.knots[i+1])
            else:
                condition = ca.logic_and(self.u >= self.knots[i], self.u < self.knots[i+1])
            
            # Local parameter within segment
            t = self.u - self.knots[i]
            
            # Cubic polynomial: f(t) = a*t^3 + b*t^2 + c*t + d
            # Note: scipy stores coefficients in reverse order [d, c, b, a]
            d_x, c_x, b_x, a_x = self.coefficients_x[0,i], self.coefficients_x[1,i], self.coefficients_x[2,i], self.coefficients_x[3,i]
            d_y, c_y, b_y, a_y = self.coefficients_y[0,i], self.coefficients_y[1,i], self.coefficients_y[2,i], self.coefficients_y[3,i]
            
            # Position
            x_seg = a_x*t**3 + b_x*t**2 + c_x*t + d_x
            y_seg = a_y*t**3 + b_y*t**2 + c_y*t + d_y
            
            # First derivatives: f'(t) = 3*a*t^2 + 2*b*t + c
            dx_seg = 3*a_x*t**2 + 2*b_x*t + c_x
            dy_seg = 3*a_y*t**2 + 2*b_y*t + c_y
            
            # Second derivatives: f''(t) = 6*a*t + 2*b
            d2x_seg = 6*a_x*t + 2*b_x
            d2y_seg = 6*a_y*t + 2*b_y
            
            # Use conditional assignment
            self.x_val += ca.if_else(condition, x_seg, self.x_val)
            self.y_val += ca.if_else(condition, y_seg, self.y_val)
            self.dx_du += ca.if_else(condition, dx_seg, self.dx_du)
            self.dy_du += ca.if_else(condition, dy_seg, self.dy_du)
            self.d2x_du2 += ca.if_else(condition, d2x_seg, self.d2x_du2)
            self.d2y_du2 += ca.if_else(condition, d2y_seg, self.d2y_du2)

    def get_symbolic_spline(self):
        """Get the symbolic spline."""
        return self.x_val, self.y_val, self.dx_du, self.dy_du, self.d2x_du2, self.d2y_du2
    
    def get_parameters(self):
        """Get the parameters of the spline."""
        return ca.vertcat(self.knots, ca.reshape(self.coefficients_x, -1, 1), ca.reshape(self.coefficients_y, -1, 1))
            
        
