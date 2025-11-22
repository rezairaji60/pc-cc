"""
pccc_synth_platoon_linear.py

Affine (degree-1) V_v(x) for Path-Complete Closure Certificates (PC-CC)
on the two-car platoon example.

We define node-wise linear functions V_v(x) and set
    C_v(x, x') = V_v(x') - V_v(x).

We enforce three inequalities on a finite grid:

(1) Transition boundedness with margin delta1:
    C_v(x, f_sigma(x)) <= zeta - delta1

(2) Forward contraction along edges with margin delta2:
    C_u(x, x') <= lambda1 * C_v(f_sigma(x), x')
                  + (1 - lambda1)*zeta - delta2

(3) Safety / well-foundedness with margin delta3:
    For all x0 ∈ X0, x ∈ X_safe, x' ∈ X_unsafe:

    C_v(x0, x) - C_v(x0, x') <=
        lambda2 * C_v(x0, x) + lambda3 * C_v(x, x')
        - (theta + lambda2*zeta + lambda3*zeta) - delta3

Equivalently:
    lhs = C_v(x0, x) - C_v(x0, x')
          - lambda2 * C_v(x0, x) - lambda3 * C_v(x, x')
          + (theta + lambda2*zeta + lambda3*zeta) + delta3 <= 0

To avoid the trivial solution, we impose a normalization:
    C_0(x_ref, x_ref_plus) = -1

Domain ("Option A"):
    x1 ∈ [0.0, 2.5]
    x2 ∈ [0.5, 3.0]
    gap = x2 - x1 >= 0

Safe / unsafe sets via gap:
    safe:   gap >= d_safe
    unsafe: gap <= d_unsafe, with 0 <= d_unsafe < d_safe

Template for V_v(x), x ∈ R^2, linear:
    Let x = (x1, x2).
    V_v(x) = c^T phi(x)

    phi(x) = [1, x1, x2]^T

"""

import numpy as np
import cvxpy as cp
import os


# ---------------------------------------------------------------------
# 0. Global hyperparameters
# ---------------------------------------------------------------------

ZETA = 1.2
LAMBDA1 = 0.9

# Ranking / safety parameters
LAMBDA2 = 0.1
LAMBDA3 = 0.1
THETA   = 0.005

# Margins for strict inequalities
DELTA1 = 0.01    # (1)
DELTA2 = 0.01    # (2)
DELTA3 = 0.005   # (3)

# Safe / unsafe gap thresholds
D_SAFE   = 1.0
D_UNSAFE = 0.2

# Grid sizes
COARSE_N1 = 7
COARSE_N2 = 7
FINE_N1   = 9
FINE_N2   = 9


# ---------------------------------------------------------------------
# 1. Switched dynamics: two-car platoon
# ---------------------------------------------------------------------

def f1(x):
    """
    Mode 1: normal communication.
    x = [x1, x2] = [follower velocity, leader velocity]
    """
    x1, x2 = x
    x1_next = 0.01 * x2 + 0.9 * x1 - 0.02 * x1**2
    x2_next = 2.0 + 0.8 * x2 - 0.04 * x2**2
    return np.array([x1_next, x2_next])


def f2_safe(x):
    """
    Mode 2: safe communication breakdown, u1 = 0, u2 = 2.
    """
    x1, x2 = x
    x1_next = 0.9 * x1 - 0.02 * x1**2
    x2_next = 2.0 + 0.8 * x2 - 0.04 * x2**2
    return np.array([x1_next, x2_next])


MODES = [1, 2]


# ---------------------------------------------------------------------
# 2. Path-complete graph (Figure 5 of Anand et al.) with 0-based nodes
# ---------------------------------------------------------------------
# Nodes V = {0,1} representing v1, v2.
# Edges (u, sigma, v):
#   v1 --1--> v1   => (0,1,0)
#   v1 --1--> v2   => (0,1,1)
#   v1 --2--> v2   => (0,2,1)
#   v2 --2--> v1   => (1,2,0)

NODES = [0, 1]
EDGES = [
    (0, 1, 0),
    (0, 1, 1),
    (0, 2, 1),
    (1, 2, 0),
]


# ---------------------------------------------------------------------
# 3. Linear V_v(x) and closure C_v(x,x') = V_v(x') - V_v(x)
# ---------------------------------------------------------------------

NBASIS = 3  # [1, x1, x2]


def phi(x):
    """
    Linear basis for V_v(x) in R^2.

    x = [x1, x2]
    phi(x) = [1, x1, x2]
    """
    x1, x2 = x
    return np.array([1.0, x1, x2])


def C_expr(c_row, x, xp):
    """
    C_v(x, x') = V_v(x') - V_v(x) as a cvxpy expression.

    c_row: cvxpy Variable of shape (NBASIS,)
    x, xp: numpy arrays (2,) for state and next-state
    """
    return c_row @ (phi(xp) - phi(x))


def C_eval(c_row, x, xp):
    """
    Numeric evaluation of C_v(x, x') given coefficients c_row (numpy).
    """
    return float(c_row @ (phi(xp) - phi(x)))


# ---------------------------------------------------------------------
# 4. Grid construction
# ---------------------------------------------------------------------

def make_grid_optionA(n1=7, n2=7,
                      x1_range=(0.0, 2.5),
                      x2_range=(0.5, 3.0)):
    """
    Build a grid of states x = [x1,x2] with:
        x1 ∈ [x1_range]
        x2 ∈ [x2_range]
        gap = x2 - x1 >= 0
    """
    xs = []
    x1_vals = np.linspace(x1_range[0], x1_range[1], n1)
    x2_vals = np.linspace(x2_range[0], x2_range[1], n2)
    for x1 in x1_vals:
        for x2 in x2_vals:
            if x2 >= x1:  # enforce gap >= 0
                xs.append(np.array([x1, x2]))
    return np.array(xs), x1_vals, x2_vals


# ---------------------------------------------------------------------
# 5. Synthesis of PC-CC on a grid
# ---------------------------------------------------------------------

def synthesize_pccc():
    """
    Synthesize linear V_v(x) and closure C_v(x,x') = V_v(x') - V_v(x)
    satisfying inequalities (1),(2),(3) with margins on a finite grid.
    """
    # Decision variables: coefficients for V_v for each node v
    # c[v,:] is a 3-vector for V_v(x)
    c = cp.Variable((len(NODES), NBASIS))

    constraints = []

    # -----------------------------------------------------------------
    # Build coarse grids
    # -----------------------------------------------------------------
    X_grid, x1_vals, x2_vals = make_grid_optionA(
        n1=COARSE_N1, n2=COARSE_N2,
        x1_range=(0.0, 2.5),
        x2_range=(0.5, 3.0)
    )
    Xp_grid, _, _ = make_grid_optionA(
        n1=COARSE_N1, n2=COARSE_N2,
        x1_range=(0.0, 2.5),
        x2_range=(0.5, 3.0)
    )

    # Grid spacing (for potential Lipschitz arguments)
    if len(x1_vals) > 1:
        h_x1 = x1_vals[1] - x1_vals[0]
    else:
        h_x1 = 0.0
    if len(x2_vals) > 1:
        h_x2 = x2_vals[1] - x2_vals[0]
    else:
        h_x2 = 0.0
    print(f"Coarse grid spacing: h_x1={h_x1:.3f}, h_x2={h_x2:.3f}")

    # Initial states X0: slightly smaller band, with gap >= D_SAFE
    X0_coarse, _, _ = make_grid_optionA(
        n1=5, n2=5,
        x1_range=(0.0, 1.5),
        x2_range=(1.0, 2.5)
    )
    X0_grid = [x for x in X0_coarse if (x[1] - x[0]) >= D_SAFE]

    # Safe / unsafe states based on gap thresholds
    unsafe_points = [x for x in Xp_grid if (x[1] - x[0]) <= D_UNSAFE]
    safe_points   = [x for x in Xp_grid if (x[1] - x[0]) >= D_SAFE]

    if not unsafe_points:
        print("[WARN] No unsafe points in grid; consider adjusting D_UNSAFE.")
    if not safe_points:
        print("[WARN] No safe points in grid; consider adjusting D_SAFE.")
    if not X0_grid:
        print("[WARN] No initial points in X0_grid; consider adjusting X0 ranges.")
    X0_grid = np.array(X0_grid)

    # -----------------------------------------------------------------
    # (1) Transition boundedness with margin:
    #     C_v(x, f_sigma(x)) <= ZETA - DELTA1
    # -----------------------------------------------------------------
    for v in NODES:
        for x in X_grid:
            for sigma in MODES:
                xp = f1(x) if sigma == 1 else f2_safe(x)
                C_val = C_expr(c[v, :], x, xp)
                constraints.append(C_val <= ZETA - DELTA1)

    # -----------------------------------------------------------------
    # (2) Forward contraction along edges with margin:
    #     C_u(x,x') <= LAMBDA1*C_v(f_sigma(x),x') + (1-LAMBDA1)*ZETA - DELTA2
    # -----------------------------------------------------------------
    for (u, sigma, v) in EDGES:
        for x in X_grid:
            x_next = f1(x) if sigma == 1 else f2_safe(x)
            for xp in Xp_grid:
                C_u = C_expr(c[u, :], x, xp)
                C_v_next = C_expr(c[v, :], x_next, xp)
                rhs = LAMBDA1 * C_v_next + (1.0 - LAMBDA1) * ZETA - DELTA2
                constraints.append(C_u <= rhs)

    # -----------------------------------------------------------------
    # (3) Safety / well-foundedness with margin:
    #
    # For all x0 ∈ X0, x ∈ X_safe, x' ∈ X_unsafe:
    #
    #   C_v(x0,x) - C_v(x0,x')
    #       <= LAMBDA2*C_v(x0,x) + LAMBDA3*C_v(x,x')
    #          - (THETA + LAMBDA2*ZETA + LAMBDA3*ZETA) - DELTA3
    #
    # Equivalently, enforce lhs <= 0 with:
    #
    #   lhs = C_v(x0,x) - C_v(x0,x')
    #         - LAMBDA2*C_v(x0,x) - LAMBDA3*C_v(x,x')
    #         + (THETA + LAMBDA2*ZETA + LAMBDA3*ZETA) + DELTA3
    # -----------------------------------------------------------------
    if unsafe_points and safe_points and len(X0_grid) > 0:
        for v in NODES:
            for x0 in X0_grid:
                for x in safe_points:
                    for x_u in unsafe_points:
                        C_x0_x  = C_expr(c[v, :], x0, x)
                        C_x0_xu = C_expr(c[v, :], x0, x_u)
                        C_x_xu  = C_expr(c[v, :], x,  x_u)

                        lhs = (
                            C_x0_x - C_x0_xu
                            - LAMBDA2 * C_x0_x
                            - LAMBDA3 * C_x_xu
                            + (THETA + LAMBDA2 * ZETA + LAMBDA3 * ZETA)
                            + DELTA3
                        )
                        constraints.append(lhs <= 0.0)

    # -----------------------------------------------------------------
    # Normalization constraint to avoid the trivial solution:
    # Pick a reference pair (x_ref, xp_ref) and enforce C_0(x_ref,xp_ref) = -1.
    # -----------------------------------------------------------------
    x_ref = X_grid[0]
    xp_ref = Xp_grid[-1]
    C_ref = C_expr(c[0, :], x_ref, xp_ref)
    constraints.append(C_ref == -1.0)

    # -----------------------------------------------------------------
    # Objective: minimize coefficient norm (regularization)
    # -----------------------------------------------------------------
    obj = cp.Minimize(cp.sum_squares(c))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    print("Status:", prob.status)
    print("Optimal value:", prob.value)
    print("zeta =", ZETA)
    print("lambda1 =", LAMBDA1)
    print("lambda2 =", LAMBDA2)
    print("lambda3 =", LAMBDA3)
    print("theta =", THETA)
    print("delta1 =", DELTA1, "delta2 =", DELTA2, "delta3 =", DELTA3)
    print("c =")
    print(c.value)

    return {
        "status": prob.status,
        "opt_value": prob.value,
        "zeta": ZETA,
        "coeffs": c.value,
        "lambda1": LAMBDA1,
        "lambda2": LAMBDA2,
        "lambda3": LAMBDA3,
        "theta": THETA,
        "delta1": DELTA1,
        "delta2": DELTA2,
        "delta3": DELTA3,
        "d_safe": D_SAFE,
        "d_unsafe": D_UNSAFE,
        "coarse_h_x1": h_x1,
        "coarse_h_x2": h_x2,
    }


# ---------------------------------------------------------------------
# 6. Max violations on a finer grid (for all three inequalities)
# ---------------------------------------------------------------------

def max_violation(params):
    """
    Evaluate maximum violation of inequalities (1), (2), and (3) on a finer grid,
    using the synthesized coefficients.
    """
    c = params["coeffs"]
    zeta = params["zeta"]
    lambda1 = params["lambda1"]
    lambda2 = params["lambda2"]
    lambda3 = params["lambda3"]
    theta = params["theta"]
    delta1 = params["delta1"]
    delta2 = params["delta2"]
    delta3 = params["delta3"]
    d_safe = params["d_safe"]
    d_unsafe = params["d_unsafe"]

    # Finer grids
    X_grid, _, _ = make_grid_optionA(
        n1=FINE_N1, n2=FINE_N2,
        x1_range=(0.0, 2.5),
        x2_range=(0.5, 3.0)
    )
    Xp_grid, _, _ = make_grid_optionA(
        n1=FINE_N1, n2=FINE_N2,
        x1_range=(0.0, 2.5),
        x2_range=(0.5, 3.0)
    )

    # Finer X0
    X0_fine, _, _ = make_grid_optionA(
        n1=7, n2=7,
        x1_range=(0.0, 1.5),
        x2_range=(1.0, 2.5)
    )
    X0_grid = [x for x in X0_fine if (x[1] - x[0]) >= d_safe]
    X0_grid = np.array(X0_grid)

    unsafe_points = [x for x in Xp_grid if (x[1] - x[0]) <= d_unsafe]
    safe_points   = [x for x in Xp_grid if (x[1] - x[0]) >= d_safe]

    viol1 = 0.0
    viol2 = 0.0
    viol3 = 0.0

    # (1) Transition boundedness with margin
    for v in NODES:
        for x in X_grid:
            for sigma in MODES:
                xp = f1(x) if sigma == 1 else f2_safe(x)
                C_val = C_eval(c[v, :], x, xp)
                viol1 = max(viol1, C_val - (zeta - delta1))

    # (2) Forward contraction with margin
    for (u, sigma, v) in EDGES:
        for x in X_grid:
            x_next = f1(x) if sigma == 1 else f2_safe(x)
            for xp in Xp_grid:
                C_u = C_eval(c[u, :], x, xp)
                C_v_next = C_eval(c[v, :], x_next, xp)
                rhs = lambda1 * C_v_next + (1.0 - lambda1) * zeta - delta2
                lhs = C_u - rhs
                viol2 = max(viol2, lhs)

    # (3) Safety / well-foundedness with margin
    if unsafe_points and safe_points and len(X0_grid) > 0:
        for v in NODES:
            for x0 in X0_grid:
                for x in safe_points:
                    for x_u in unsafe_points:
                        C_x0_x  = C_eval(c[v, :], x0, x)
                        C_x0_xu = C_eval(c[v, :], x0, x_u)
                        C_x_xu  = C_eval(c[v, :], x,  x_u)

                        lhs = (
                            C_x0_x - C_x0_xu
                            - lambda2 * C_x0_x
                            - lambda3 * C_x_xu
                            + (theta + lambda2 * zeta + lambda3 * zeta)
                            + delta3
                        )
                        viol3 = max(viol3, lhs)

    return viol1, viol2, viol3


# ---------------------------------------------------------------------
# 7. Lipschitz constants for C_v in state-space (exact for linear)
# ---------------------------------------------------------------------

def lipschitz_C_for_node_linear(c_row):
    """
    For linear V(x) = c0 + c1 x1 + c2 x2, we have:

        grad V(x) = [c1, c2] (constant).

    Then C(x,x') = V(x') - V(x), so

        grad_x C = -grad V = [-c1, -c2]
        grad_x' C = grad V = [c1, c2]

    So the gradient wrt (x,x') ∈ R^4 is

        grad_C = [-c1, -c2, c1, c2]

    and its 2-norm is sqrt(2) * ||[c1, c2]||_2 (constant over domain).
    This is a global Lipschitz constant.
    """
    c0, c1, c2 = c_row
    g = np.array([c1, c2])
    return float(np.sqrt(2.0) * np.linalg.norm(g, 2))


def compute_lipschitz_constants(params):
    """
    Compute exact Lipschitz constants of C_v for each node for the linear case.
    """
    c = params["coeffs"]
    Ls = []
    for v in NODES:
        L_v = lipschitz_C_for_node_linear(c[v, :])
        Ls.append(L_v)
    return Ls


# ---------------------------------------------------------------------
# 8. Saving coefficients for later use (simulations, plotting)
# ---------------------------------------------------------------------

def save_coefficients(params, filename="pccc_platoon_coeffs_linear.npy"):
    """
    Save coefficients and parameters to a .npy file for later use.
    """
    out = {
        "coeffs": params["coeffs"],    # shape (2,3)
        "zeta": params["zeta"],
        "lambda1": params["lambda1"],
        "lambda2": params["lambda2"],
        "lambda3": params["lambda3"],
        "theta": params["theta"],
        "delta1": params["delta1"],
        "delta2": params["delta2"],
        "delta3": params["delta3"],
        "nodes": NODES,
        "edges": EDGES,
        "basis": "linear_state",
        "domain_option": "A",
        "x1_range": (0.0, 2.5),
        "x2_range": (0.5, 3.0),
        "d_safe": params["d_safe"],
        "d_unsafe": params["d_unsafe"],
        "coarse_h_x1": params["coarse_h_x1"],
        "coarse_h_x2": params["coarse_h_x2"],
    }
    np.save(filename, out, allow_pickle=True)
    print(f"\nSaved linear PC-CC coefficients and parameters to {os.path.abspath(filename)}")


# ---------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------

def main():
    params = synthesize_pccc()
    if params["status"] not in ("optimal", "optimal_inaccurate"):
        print("[WARN] Optimization did not return an optimal solution. Aborting checks.")
        return

    viol1, viol2, viol3 = max_violation(params)
    print("\nMax violation on finer grid:")
    print("  (1) Transition boundedness   :", viol1)
    print("  (2) Forward contraction      :", viol2)
    print("  (3) Safety / well-foundedness:", viol3)

    # Lipschitz constants of C_v in state-space (exact for linear case)
    Ls = compute_lipschitz_constants(params)
    print("\nExact Lipschitz constants of C_v (||grad||_2 over x,x'):")
    for v, L in zip(NODES, Ls):
        print(f"  Node v{v+1}: L_C = {L:.4f}")

    # Save coefficients to use in other scripts (simulation/plots)
    save_coefficients(params)


if __name__ == "__main__":
    main()
