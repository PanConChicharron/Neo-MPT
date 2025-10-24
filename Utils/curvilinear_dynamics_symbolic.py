# ============================================================
# curvilinear_dynamics_symbolic.py
# Curvilinear dynamics with kinematic bicycle (slip β)
# Correct definitions displayed, symbolic derivatives, working lambdify
# ============================================================

import sympy as sp

# =============================================================
# 0. Key symbolic definitions (display only)
# =============================================================
x, y, psi = sp.symbols('x y psi', real=True)
s = sp.symbols('s', real=True)

x_r = sp.Function('x_r')(s)
y_r = sp.Function('y_r')(s)
psi_r = sp.Function('psi_r')(s)
beta, delta, L = sp.symbols('beta delta L', real=True)

# Curvilinear deviations (for display only)
eY_expr = -(x - x_r)*sp.sin(psi_r) + (y - y_r)*sp.cos(psi_r)
ePsi_expr = psi - psi_r

K_expr = sp.cos(beta) * sp.tan(delta) / L  # slip-modified curvature

print("=== Key symbolic definitions (display only) ===")
print("eY   = "); sp.pprint(eY_expr)
print("ePsi = "); sp.pprint(ePsi_expr)
print("K    = "); sp.pprint(K_expr)
print("==============================================\n")

# =============================================================
# 1. Symbolic variables for computation
# =============================================================
v, l_f, l_r, K_ref = sp.symbols('v l_f l_r K_ref', real=True)
eY, ePsi = sp.symbols('eY ePsi', real=True)  # symbolic deviations
K = sp.symbols('K')                           # symbolic curvature

# =============================================================
# 2. Kinematic bicycle velocities using ePsi + psi_r
# =============================================================
psi_subbed = ePsi + psi_r  # replace psi - psi_r with ePsi
xdot = v * sp.cos(psi_subbed)
ydot = v * sp.sin(psi_subbed)
psidot = v * K

# =============================================================
# 3. General geometric relations
# =============================================================
t_r = sp.Matrix([sp.cos(psi_r), sp.sin(psi_r)])     # tangent
n_r = sp.Matrix([-sp.sin(psi_r), sp.cos(psi_r)])    # normal
v_global = sp.Matrix([xdot, ydot])

# =============================================================
# 4. Time derivatives in curvilinear coordinates
# =============================================================
s_dot = sp.simplify((v_global.dot(t_r)) / (1 - K_ref * eY))
eY_dot = sp.simplify(v_global.dot(n_r))
ePsi_dot = sp.simplify(psidot - K_ref * s_dot)

# =============================================================
# 5. Spatial derivatives d()/ds
# =============================================================
eY_prime = sp.simplify(eY_dot / s_dot)
ePsi_prime = K*(1 - K_ref*eY)/sp.cos(ePsi) - K_ref  # factorized for clarity

# =============================================================
# 6. Display derivatives
# =============================================================
print("=== TIME DERIVATIVES (slip β) ===")
print("ṡ ="); sp.pprint(s_dot)
print("\nėY ="); sp.pprint(eY_dot)
print("\nePsi̇ ="); sp.pprint(ePsi_dot)

print("\n=== SPATIAL DERIVATIVES (d/ds) ===")
print("eY' ="); sp.pprint(eY_prime)
print("\nePsi' ="); sp.pprint(ePsi_prime)

# =============================================================
# 7. Lambdify for numerical evaluation (symbolic eY, ePsi)
# =============================================================
curvilinear_dynamics_time = sp.lambdify(
    (v, K, eY, ePsi, s, K_ref, psi_r),
    (s_dot, eY_dot, ePsi_dot),
    "numpy"
)

curvilinear_dynamics_spatial = sp.lambdify(
    (v, K, eY, ePsi, s, K_ref, psi_r),
    (eY_prime, ePsi_prime),
    "numpy"
)
