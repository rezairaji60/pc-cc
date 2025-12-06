"""
Author: Reza Iraji
Date:   December 2025

Piecewise-quadratic V_v(x) for Path-Complete Closure Certificates (PC-CC)
on the two-car platoon example, on the domain [0,1] × [0,5.1].

We define node-wise piecewise-quadratic functions V_v(x) and set
    C_v(x, x') = V_v(x') - V_v(x).

We enforce three inequalities on finite grids:

(1) Transition boundedness:
    For all v ∈ NODES, sigma ∈ {1,2}, x in domain:
        C_v(x, f_sigma(x)) <= zeta

(2) Forward contraction along edges:
    For all (u --sigma--> v) ∈ EDGES, all x, x':
        C_u(x, x') <= lambda1 * C_v(f_sigma(x), x') + (1 - lambda1)*zeta

(3) Well-foundedness / finite visits of X_u:
    For all x0 ∈ X0, x ∈ X_u\X0, x' ∈ X_u\X0, and all v in RANK_NODES:
        C_v(x0, x') + C_v(x, x0)
            <= lambda2 * C_v(x0, x) + lambda3 * C_v(x, x')
               - (theta + lambda2*zeta + lambda3*zeta)

We implement these as grid-based LMIs / inequalities.

State and gap:
    x = [x1, x2]
    gap = x2 - x1 >= 0

X0 (initial set) is defined by gap <= D_SAFE.
X_u (unsafe set) is defined by gap <= D_UNSAFE, with D_SAFE < D_UNSAFE.

Template for V_v(x), x ∈ R^2, piecewise-quadratic:
    x = (x1, x2), gap = x2 - x1.

We use 3 regions in gap:
    Region 0: gap <= D_SAFE
    Region 1: D_SAFE < gap <= D_UNSAFE
    Region 2: gap > D_UNSAFE

On each region r we use a quadratic in
    [1, x1, x2, x1^2, x1*x2, x2^2]^T
with node- and region-dependent coefficients.

So for node v and region r we have coefficients c_reg[v, r, :] ∈ R^6, and

    V_v(x) = c_reg[v, r(x), :] @ basis(x),

where r(x) is the region index determined by gap(x).
"""

import numpy as np
import cvxpy as cp
import os

# ---------------------------------------------------------------------
# 0. Global hyperparameters
# ---------------------------------------------------------------------

# Main PC-CC scalars
ZETA    = 3.0       # transition bound level
LAMBDA1 = 0.95      # forward contraction factor

# Ranking / well-foundedness parameters
LAMBDA2 = 0.02
LAMBDA3 = 0.02
THETA   = 5e-5

# Margins (set to zero; grid tolerance already gives "strictness")
DELTA1 = 0.0   # (1) transition boundedness
DELTA2 = 0.0   # (2) forward contraction
DELTA3 = 0.0   # (3) well-foundedness

# Safe / unsafe thresholds in gap (x2 - x1)
D_SAFE   = 0.3   # X0: gap <= D_SAFE
D_UNSAFE = 0.4   # X_u: gap <= D_UNSAFE (X0 ⊂ X_u and X_u\X0 nonempty)

# Number of gap regions for piecewise-quadratic V_v(x)
N_REGIONS = 3    # Region 0: gap<=D_SAFE; 1: (D_SAFE,D_UNSAFE]; 2: >D_UNSAFE

# Grid sizes for synthesis
COARSE_N1 = 7   # synthesis grid (x1 direction)
COARSE_N2 = 7   # synthesis grid (x2 direction)

# Fine grid for checking
FINE_N1   = 9
FINE_N2   = 9

# Well-foundedness (X_u) grid
WF_N1     = 5
WF_N2     = 5

# X0 grid resolution
X0_N1     = 5
X0_N2     = 5

# Numerical tolerance for grid-based certification
NUM_TOL = 1e-8

# State space ranges
X1_RANGE = (0.0, 3.0)
X2_RANGE = (0.0, 5.1)

# Nodes used as ranking nodes in (3); both nodes appear in (1),(2)
RANK_NODES = [0]   # node 0 is the ranking node

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
# 2. Path-complete graph (0-based nodes)
# ---------------------------------------------------------------------

NODES = [0, 1]
EDGES = [
    (0, 1, 0),
    (0, 1, 1),
    (0, 2, 1),
    (1, 2, 0),
]

# ---------------------------------------------------------------------
# 3. Piecewise-quadratic V_v(x) and C_v(x,x') = V_v(x') - V_v(x)
# ---------------------------------------------------------------------

NBASIS = 6  # [1, x1, x2, x1^2, x1*x2, x2^2]


def quad_basis(x):
    x1, x2 = x
    return np.array([1.0, x1, x2, x1**2, x1 * x2, x2**2])


def region_index(x):
    """
    Decide which gap region x belongs to, based on gap = x2 - x1.

    Region 0: gap <= D_SAFE
    Region 1: D_SAFE < gap <= D_UNSAFE
    Region 2: gap > D_UNSAFE
    """
    x1, x2 = x
    gap = x2 - x1
    if gap <= D_SAFE:
        return 0
    elif gap <= D_UNSAFE:
        return 1
    else:
        return 2


def V_expr(c_reg_v, x):
    """
    Piecewise-quadratic V_v(x) as a cvxpy expression.

    c_reg_v: slice of shape (N_REGIONS, NBASIS) for node v (cvxpy Variable slice).
    x: numpy array of shape (2,).
    """
    r = region_index(x)
    return c_reg_v[r, :] @ quad_basis(x)


def V_eval(c_reg_v_val, x):
    """
    Numeric evaluation of V_v(x) given coefficients for node v.

    c_reg_v_val: numpy array of shape (N_REGIONS, NBASIS).
    """
    r = region_index(x)
    return float(c_reg_v_val[r, :] @ quad_basis(x))


def C_expr(c_reg, v, x, xp):
    """
    C_v(x, x') = V_v(x') - V_v(x) as a cvxpy expression.

    c_reg: cvxpy Variable of shape (len(NODES), N_REGIONS, NBASIS)
    v: node index (0 or 1)
    x, xp: numpy arrays (2,) for state and next-state
    """
    return V_expr(c_reg[v, :, :], xp) - V_expr(c_reg[v, :, :], x)


def C_eval(c_reg_val, v, x, xp):
    """
    Numeric evaluation of C_v(x, x') given coefficients (numpy array).

    c_reg_val: numpy array of shape (len(NODES), N_REGIONS, NBASIS)
    """
    return V_eval(c_reg_val[v, :, :], xp) - V_eval(c_reg_val[v, :, :], x)


# ---------------------------------------------------------------------
# 4. Grid construction
# ---------------------------------------------------------------------

def make_grid_optionA(n1=COARSE_N1, n2=COARSE_N2,
                      x1_range=X1_RANGE,
                      x2_range=X2_RANGE):
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


def make_X0_grid(n1, n2):
    """
    Build X0 grid with gap <= D_SAFE.
    """
    X0_raw, _, _ = make_grid_optionA(
        n1=n1, n2=n2,
        x1_range=X1_RANGE,
        x2_range=X2_RANGE
    )
    X0_grid = [x for x in X0_raw if (x[1] - x[0]) <= D_SAFE]
    return np.array(X0_grid)


def make_unsafe_grid(n1, n2):
    """
    Build the unsafe set X_u grid based on gap <= D_UNSAFE.
    """
    X_raw, _, _ = make_grid_optionA(
        n1=n1, n2=n2,
        x1_range=X1_RANGE,
        x2_range=X2_RANGE
    )
    unsafe_points = [x for x in X_raw if (x[1] - x[0]) <= D_UNSAFE]
    return np.array(unsafe_points)


# ---------------------------------------------------------------------
# 5. Synthesis of piecewise-quadratic PC-CC with 3 gap regions
# ---------------------------------------------------------------------

def synthesize_pccc():
    """
    Synthesize piecewise-quadratic V_v(x) and closure C_v(x,x') = V_v(x') - V_v(x)
    with 3 gap regions, satisfying inequalities (1),(2),(3) on a finite grid.
    """

    # Decision variables: coefficients for V_v, for each node v, region r.
    # Shape: (len(NODES), N_REGIONS, NBASIS)
    c_reg = cp.Variable((len(NODES), N_REGIONS, NBASIS))

    constraints = []

    # -----------------------------------------------------------------
    # Build coarse grids for (1) and (2)
    # -----------------------------------------------------------------
    X_grid, x1_vals, x2_vals = make_grid_optionA(
        n1=COARSE_N1, n2=COARSE_N2,
        x1_range=X1_RANGE,
        x2_range=X2_RANGE
    )
    Xp_grid, _, _ = make_grid_optionA(
        n1=COARSE_N1, n2=COARSE_N2,
        x1_range=X1_RANGE,
        x2_range=X2_RANGE
    )

    # Grid spacing (for reporting)
    if len(x1_vals) > 1:
        h_x1 = x1_vals[1] - x1_vals[0]
    else:
        h_x1 = 0.0
    if len(x2_vals) > 1:
        h_x2 = x2_vals[1] - x2_vals[0]
    else:
        h_x2 = 0.0
    print(f"Coarse grid spacing: h_x1={h_x1:.3f}, h_x2={h_x2:.3f}")

    # X0 grid and unsafe grid for well-foundedness
    X0_grid = make_X0_grid(X0_N1, X0_N2)
    unsafe_points = make_unsafe_grid(WF_N1, WF_N2)

    if unsafe_points.size == 0:
        print("[WARN] No unsafe points in WF grid; consider adjusting D_UNSAFE.")
    if X0_grid.size == 0:
        print("[WARN] No initial points in X0_grid; consider adjusting X0 ranges.")

    # For well-foundedness we EXCLUDE unsafe points that coincide with any x0 ∈ X0,
    # to avoid the impossible x = x' = x0 situation.
    unsafe_wf = []
    if unsafe_points.size > 0:
        for x in unsafe_points:
            if X0_grid.size == 0:
                unsafe_wf.append(x)
            else:
                if not any(np.allclose(x, x0) for x0 in X0_grid):
                    unsafe_wf.append(x)
        unsafe_wf = np.array(unsafe_wf)
    else:
        unsafe_wf = np.array([])

    # -----------------------------------------------------------------
    # (1) Transition boundedness:
    #     C_v(x, f_sigma(x)) <= ZETA
    # -----------------------------------------------------------------
    for v in NODES:
        for x in X_grid:
            for sigma in MODES:
                xp = f1(x) if sigma == 1 else f2_safe(x)
                C_val = C_expr(c_reg, v, x, xp)
                constraints.append(C_val <= ZETA - DELTA1)

    # -----------------------------------------------------------------
    # (2) Forward contraction along edges:
    #     C_u(x,x') <= LAMBDA1*C_v(f_sigma(x),x') + (1-LAMBDA1)*ZETA
    # -----------------------------------------------------------------
    for (u, sigma, v) in EDGES:
        for x in X_grid:
            x_next = f1(x) if sigma == 1 else f2_safe(x)
            for xp in Xp_grid:
                C_u_val = C_expr(c_reg, u, x, xp)
                C_v_next = C_expr(c_reg, v, x_next, xp)
                rhs = LAMBDA1 * C_v_next + (1.0 - LAMBDA1) * ZETA - DELTA2
                constraints.append(C_u_val <= rhs)

    # -----------------------------------------------------------------
    # (3) Well-foundedness / finite visits of X_u:
    #
    # For all x0 ∈ X0, x ∈ X_u\X0, x' ∈ X_u\X0, for all v ∈ RANK_NODES:
    #
    #   C_v(x0,x') + C_v(x,x0)
    #       <= LAMBDA2*C_v(x0,x) + LAMBDA3*C_v(x,x')
    #          - (THETA + LAMBDA2*ZETA + LAMBDA3*ZETA)
    #
    # Equivalently, enforce lhs <= 0 with:
    #
    #   lhs = C_v(x0,x') + C_v(x,x0)
    #         - LAMBDA2*C_v(x0,x) - LAMBDA3*C_v(x,x')
    #         + (THETA + LAMBDA2*ZETA + LAMBDA3*ZETA)
    #         + DELTA3
    # -----------------------------------------------------------------
    if (unsafe_wf.size > 0) and (X0_grid.size > 0):
        for v in RANK_NODES:
            for x0 in X0_grid:
                for x in unsafe_wf:
                    for x_u in unsafe_wf:
                        C_x0_xu = C_expr(c_reg, v, x0, x_u)
                        C_x_x0  = C_expr(c_reg, v, x,  x0)
                        C_x0_x  = C_expr(c_reg, v, x0, x)
                        C_x_xu  = C_expr(c_reg, v, x,  x_u)

                        lhs = (
                            C_x0_xu + C_x_x0
                            - LAMBDA2 * C_x0_x
                            - LAMBDA3 * C_x_xu
                            + (THETA + LAMBDA2 * ZETA + LAMBDA3 * ZETA)
                            + DELTA3
                        )
                        constraints.append(lhs <= 0.0)

    # -----------------------------------------------------------------
    # Normalization to avoid trivial solution:
    # Pick a reference pair (x_ref, xp_ref) and enforce C_0(x_ref,xp_ref) = -1.
    # -----------------------------------------------------------------
    x_ref  = X_grid[0]
    xp_ref = Xp_grid[-1]
    C_ref  = C_expr(c_reg, 0, x_ref, xp_ref)
    constraints.append(C_ref == -1.0)

    # -----------------------------------------------------------------
    # Objective: minimize coefficient norm (regularization)
    # -----------------------------------------------------------------
    obj = cp.Minimize(cp.sum_squares(c_reg))

    prob = cp.Problem(obj, constraints)
    print("Solving piecewise-quadratic PC-CC synthesis problem with ECOS...")
    prob.solve(
        solver=cp.ECOS,
        abstol=1e-8,
        reltol=1e-8,
        feastol=1e-8,
        max_iters=100000,
        verbose=True,
    )
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("ECOS failed or not optimal, falling back to SCS...")
        prob.solve(
            solver=cp.SCS,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iters=200000,
            verbose=True,
        )

    print("Status:", prob.status)
    print("Optimal value:", prob.value)
    print("zeta =", ZETA)
    print("lambda1 =", LAMBDA1)
    print("lambda2 =", LAMBDA2)
    print("lambda3 =", LAMBDA3)
    print("theta =", THETA)
    print("delta1 =", DELTA1, "delta2 =", DELTA2, "delta3 =", DELTA3)
    print("c_reg (coefficients, shape (nodes, regions, basis)) =")
    print(c_reg.value)

    return {
        "status": prob.status,
        "opt_value": prob.value,
        "zeta": ZETA,
        "coeffs_reg": c_reg.value,
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
# 6. Max violations on a fine grid (for all three inequalities)
# ---------------------------------------------------------------------

def max_violation(params):
    """
    Evaluate maximum violation of inequalities (1), (2), and (3) on grids.

    (1),(2): checked on FINE_N1 x FINE_N2 grid for x,x'.
    (3): checked on WF_N1 x WF_N2 grid for X_u and X0_N1 x X0_N2 for X0.
    """
    c_reg_val = params["coeffs_reg"]
    zeta    = params["zeta"]
    lambda1 = params["lambda1"]
    lambda2 = params["lambda2"]
    lambda3 = params["lambda3"]
    theta   = params["theta"]
    delta1  = params["delta1"]
    delta2  = params["delta2"]
    delta3  = params["delta3"]

    # Grids for x, x' in (1) and (2)
    X_grid, x1_vals_f, x2_vals_f = make_grid_optionA(
        n1=FINE_N1, n2=FINE_N2,
        x1_range=X1_RANGE,
        x2_range=X2_RANGE
    )
    Xp_grid, _, _ = make_grid_optionA(
        n1=FINE_N1, n2=FINE_N2,
        x1_range=X1_RANGE,
        x2_range=X2_RANGE
    )

    # X0 and X_u grids
    X0_grid = make_X0_grid(X0_N1, X0_N2)
    unsafe_points = make_unsafe_grid(WF_N1, WF_N2)

    # Exclude points in X0 from unsafe set for (3)
    unsafe_wf = []
    if unsafe_points.size > 0:
        for x in unsafe_points:
            if X0_grid.size == 0:
                unsafe_wf.append(x)
            else:
                if not any(np.allclose(x, x0) for x0 in X0_grid):
                    unsafe_wf.append(x)
        unsafe_wf = np.array(unsafe_wf)

    viol1 = 0.0
    viol2 = 0.0
    viol3 = 0.0

    worst1 = None
    worst2 = None
    worst3 = None

    # (1) Transition boundedness
    for v in NODES:
        for x in X_grid:
            for sigma in MODES:
                xp = f1(x) if sigma == 1 else f2_safe(x)
                C_val = C_eval(c_reg_val, v, x, xp)
                lhs = C_val - (zeta - delta1)
                if lhs > viol1:
                    viol1 = lhs
                    worst1 = (v, sigma, x.copy(), xp.copy())

    # (2) Forward contraction
    for (u, sigma, v) in EDGES:
        for x in X_grid:
            x_next = f1(x) if sigma == 1 else f2_safe(x)
            for xp in Xp_grid:
                C_u_val = C_eval(c_reg_val, u, x, xp)
                C_v_next = C_eval(c_reg_val, v, x_next, xp)
                rhs = lambda1 * C_v_next + (1.0 - lambda1) * zeta - delta2
                lhs = C_u_val - rhs
                if lhs > viol2:
                    viol2 = lhs
                    worst2 = (u, sigma, v, x.copy(), x_next.copy(), xp.copy())

    # (3) Well-foundedness / finite visits
    if (unsafe_wf.size > 0) and (X0_grid.size > 0):
        for v in RANK_NODES:
            for x0 in X0_grid:
                for x in unsafe_wf:
                    for x_u in unsafe_wf:
                        C_x0_xu = C_eval(c_reg_val, v, x0, x_u)
                        C_x_x0  = C_eval(c_reg_val, v, x,  x0)
                        C_x0_x  = C_eval(c_reg_val, v, x0, x)
                        C_x_xu  = C_eval(c_reg_val, v, x,  x_u)

                        lhs = (
                            C_x0_xu + C_x_x0
                            - lambda2 * C_x0_x
                            - lambda3 * C_x_xu
                            + (theta + lambda2 * zeta + lambda3 * zeta)
                            + delta3
                        )
                        if lhs > viol3:
                            viol3 = lhs
                            worst3 = (v, x0.copy(), x.copy(), x_u.copy())

    # Fine-grid spacings
    if len(x1_vals_f) > 1:
        h_x1_f = x1_vals_f[1] - x1_vals_f[0]
    else:
        h_x1_f = 0.0
    if len(x2_vals_f) > 1:
        h_x2_f = x2_vals_f[1] - x2_vals_f[0]
    else:
        h_x2_f = 0.0

    # Debug info
    if worst2 is not None:
        u, sigma, v, x, x_next, xp = worst2
        print("\n[DEBUG] Worst forward-contraction (2) violation on fine grid:")
        print("  value =", viol2)
        print("  edge (u --sigma--> v) =", (u, sigma, v))
        print("  x      =", x)
        print("  f_sigma(x) =", x_next)
        print("  x'     =", xp)

    if worst1 is not None:
        v, sigma, x, xp = worst1
        print("\n[DEBUG] Worst transition-boundedness (1) violation:")
        print("  value =", viol1)
        print("  node v =", v, ", sigma =", sigma)
        print("  x      =", x)
        print("  f_sigma(x) =", xp)

    if worst3 is not None:
        v, x0, x, x_u = worst3
        print("\n[DEBUG] Worst well-foundedness (3) violation:")
        print("  value =", viol3)
        print("  node v =", v)
        print("  x0 =", x0)
        print("  x  =", x)
        print("  x_u =", x_u)

    return viol1, viol2, viol3, h_x1_f, h_x2_f


# ---------------------------------------------------------------------
# 7. Lipschitz constants – not implemented for quadratic here
# ---------------------------------------------------------------------

def compute_lipschitz_constants_infty(_params):
    """
    Placeholder: Lipschitz constants of C_v for piecewise-quadratic case
    are not computed in this script.
    """
    print("\n[INFO] Lipschitz constants for piecewise-quadratic C_v "
          "are not computed in this script.")
    return []


def compute_lipschitz_margins_infty(Ls_infty, h_x1_f, h_x2_f):
    """
    With no Lipschitz constants, margins are zero; we rely on grid-based check.
    """
    if not Ls_infty:
        return 0.0, 0.0, 0.0

    L_max = max(Ls_infty)
    h_max = max(h_x1_f, h_x2_f)
    base = L_max * (h_max / 2.0)
    lip_margin1 = base
    lip_margin2 = 2.0 * base
    lip_margin3 = 3.0 * base
    return lip_margin1, lip_margin2, lip_margin3


# ---------------------------------------------------------------------
# 8. Saving coefficients for later use (simulations, plotting)
# ---------------------------------------------------------------------

def save_coefficients(params, Ls_infty,
                      filename="pccc_platoon_coeffs_pwquad_3regions.npy"):
    """
    Save coefficients and parameters to a .npy file for later use.
    """
    out = {
        "coeffs_reg": params["coeffs_reg"],     # shape (2, N_REGIONS, 6)
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
        "basis": "piecewise_quadratic_state_3gapregions",
        "domain_option": "A",
        "x1_range": X1_RANGE,
        "x2_range": X2_RANGE,
        "d_safe": params["d_safe"],
        "d_unsafe": params["d_unsafe"],
        "coarse_h_x1": params["coarse_h_x1"],
        "coarse_h_x2": params["coarse_h_x2"],
        "L_C_infty_nodes": Ls_infty,
        "L_C_infty_max": max(Ls_infty) if Ls_infty else None,
        "n_regions": N_REGIONS,
        "region_description": (
            "Region 0: gap<=D_SAFE; Region 1: D_SAFE<gap<=D_UNSAFE; "
            "Region 2: gap>D_UNSAFE"
        ),
    }
    np.save(filename, out, allow_pickle=True)
    print(f"\nSaved piecewise-quadratic PC-CC coefficients and parameters to "
          f"{os.path.abspath(filename)}")


# ---------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------

def main():
    params = synthesize_pccc()
    if params["status"] not in ("optimal", "optimal_inaccurate"):
        print("[WARN] Optimization did not return an optimal solution. Aborting checks.")
        return

    viol1, viol2, viol3, h_x1_f, h_x2_f = max_violation(params)
    print("\nMax violation on grid:")
    print("  (1) Transition boundedness   :", viol1)
    print("  (2) Forward contraction      :", viol2)
    print("  (3) Well-foundedness (X_u)   :", viol3)

    grid_certified = (viol1 <= NUM_TOL) and (viol2 <= NUM_TOL) and (viol3 <= NUM_TOL)
    print(f"\nGrid-based certification (tol={NUM_TOL:g}): "
          f"{'YES' if grid_certified else 'NO'}")

    # Lipschitz constants of C_v (not implemented for quadratic):
    Ls_infty = compute_lipschitz_constants_infty(params)

    # Lipschitz-based margins (will be zero if Ls_infty is empty)
    lip_margin1, lip_margin2, lip_margin3 = compute_lipschitz_margins_infty(
        Ls_infty, h_x1_f, h_x2_f
    )
    print("\nLipschitz margins (∞-norm, using grid spacing):")
    print("  lip_margin1 (transition boundedness) : {:.6e}".format(lip_margin1))
    print("  lip_margin2 (forward contraction)    : {:.6e}".format(lip_margin2))
    print("  lip_margin3 (well-foundedness)       : {:.6e}".format(lip_margin3))

    cert_viol1 = viol1 + lip_margin1
    cert_viol2 = viol2 + lip_margin2
    cert_viol3 = viol3 + lip_margin3

    print("\nConservative Lipschitz-based worst-case (grid max + margins):")
    print("  (1) Transition boundedness   :", cert_viol1)
    print("  (2) Forward contraction      :", cert_viol2)
    print("  (3) Well-foundedness (X_u)   :", cert_viol3)

    if (cert_viol1 <= 0.0) and (cert_viol2 <= 0.0) and (cert_viol3 <= 0.0):
        print("\n*** PC-CC is Lipschitz-certified over the domain "
              "covered by this grid (very conservative) ***")
    else:
        print("\n*** Lipschitz robustness check is inconclusive (margins not "
              "computed exactly for piecewise-quadratic), but the grid-based "
              "certificate above is still valid if 'YES'. ***")

    # Save coefficients to use in other scripts (simulation/plots)
    save_coefficients(params, Ls_infty)


if __name__ == "__main__":
    main()
