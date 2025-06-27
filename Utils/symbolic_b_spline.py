"""
Generic symbolic B-spline for optimization applications.
"""

import casadi as ca
import numpy as np

class SymbolicBSpline:
    """Symbolic B-spline class for CasADi optimization."""
    
    def __init__(self, n_coeffs: int, u: ca.SX.sym('u')):
        """Initialize B-spline with n_coeffs coefficients."""
        self.n_coeffs = n_coeffs
        self.degree = 3  # Cubic B-spline
        
        # Parameters: knots and coefficients
        self.knots = ca.SX.sym('knots', n_coeffs + self.degree + 1)
        self.coefficients = ca.SX.sym('coefficients', n_coeffs)
        
        self.u = u
        self.x_val = 0
        
        # Create symbolic B-spline evaluation using Cox-de Boor recursion
        self.x_val = self._evaluate_bspline()
    
    def _evaluate_bspline(self):
        """Evaluate B-spline using Cox-de Boor recursion formula."""
        # Clamp u to the valid range [knots[degree], knots[n_coeffs]]
        u_val = ca.fmax(self.knots[self.degree], ca.fmin(self.u, self.knots[self.n_coeffs]))
        
        # Create basis functions for each coefficient
        result = 0
        for i in range(self.n_coeffs):
            basis_val = self._basis_function(i, self.degree, u_val)
            result += self.coefficients[i] * basis_val
        
        return result
    
    def _basis_function(self, i, k, u):
        """Compute B-spline basis function B_{i,k}(u) using Cox-de Boor recursion."""
        if k == 0:
            # Base case: piecewise constant
            return ca.if_else(
                ca.logic_and(u >= self.knots[i], u < self.knots[i+1]),
                1.0, 0.0
            )
        else:
            # Recursive case
            d1 = self.knots[i+k] - self.knots[i]
            d2 = self.knots[i+k+1] - self.knots[i+1]
            
            # Use CasADi conditional functions instead of Python if
            c1 = ca.if_else(d1 > 1e-10, 
                           (u - self.knots[i]) / d1 * self._basis_function(i, k-1, u), 
                           0.0)
            
            c2 = ca.if_else(d2 > 1e-10, 
                           (self.knots[i+k+1] - u) / d2 * self._basis_function(i+1, k-1, u), 
                           0.0)
            
            return c1 + c2

    def get_symbolic_spline(self):
        """Get the symbolic B-spline value."""
        return self.x_val
    
    def get_parameters(self):
        """Get the parameters of the B-spline (knots and coefficients)."""
        return ca.vertcat(self.knots, self.coefficients)
            
        
