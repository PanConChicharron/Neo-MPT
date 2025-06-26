from Utils.symbolic_cubic_spline import SymbolicCubicSpline

spline = SymbolicCubicSpline(n_points=2)

x_val, y_val, dx_du, dy_du, d2x_du2, d2y_du2 = spline.get_symbolic_spline()
coefficients = spline.get_parameters()

print(x_val)
print(y_val)
print(dx_du)
print(dy_du)
print(d2x_du2)
print(d2y_du2)
print(coefficients)