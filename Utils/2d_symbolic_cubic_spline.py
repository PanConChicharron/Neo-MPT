import casadi as ca

from Utils.symbolic_b_spline import SymbolicCubicSpline

class Symbolic2DCubicSpline:
    def __init__(self, n_points: int):
        self.spline_x = SymbolicCubicSpline(n_points)
        self.spline_y = SymbolicCubicSpline(n_points)

    def get_symbolic_spline(self):
        return self.spline_x.get_symbolic_spline(), self.spline_y.get_symbolic_spline()
    
    def get_symbolic_derivatives(self):
        return self.spline_x.get_symbolic_derivatives(), self.spline_y.get_symbolic_derivatives()
    
    def get_symbolic_second_derivatives(self):
        return self.spline_x.get_symbolic_second_derivatives(), self.spline_y.get_symbolic_second_derivatives()
    
    def get_parameters(self):
        return ca.vertcat(self.spline_x.get_parameters(), self.spline_y.get_parameters())