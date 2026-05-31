"""
Author: Reza Iraji
Date:   December 2025

Simulation and plotting for Path-Complete Closure Certificates (PC-CC)
on the two-car platoon example, using the *piecewise-quadratic* PC-CC
synthesized in pccc_synth_platoon.py.

Figures produced (for paper / journal case study):

1) Separate figures:
   - State-space trajectories (x1, x2) with domain, unsafe band (red),
     safe band (green).
   - Gap evolution g(t) = x2(t) - x1(t) with unsafe/safe shaded regions.
   - Closure values C_{v_t}(x_t, x_{t+1}) with threshold zeta and
     shaded safe/unsafe regions.

2) Combined 3-panel figure:
   - Left: state trajectories.
   - Middle: gap evolution.
   - Right: closure values.

Run from repo root:

    python -m sim.pccc_sim_platoon
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 0. Load PW-quadratic PC-CC parameters
# ---------------------------------------------------------------------

def load_pccc_pwquad(filename="pccc_platoon_coeffs_pwquad_3regions.npy"):
    """
    Load the synthesized piecewise-quadratic PC-CC coefficients and parameters.
    Expects the file to live one level above this script (repo root).

    Required keys in the npy (from the synthesis script):
      - 'coeffs_reg' : shape (num_nodes, n_regions, 6)
      - 'zeta', 'lambda1', 'lambda2', 'lambda3', 'theta'
      - 'd_safe', 'd_unsafe'
      - 'x1_range', 'x2_range'
      - 'nodes', 'edges', 'n_regions', ...
    """
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", filename)
    data = np.load(path, allow_pickle=True).item()

    if "coeffs_reg" not in data:
        raise KeyError(
            "Expected key 'coeffs_reg' in pw-quad npy file. "
            f"Available keys: {list(data.keys())}"
        )

    return data


# ---------------------------------------------------------------------
# 1. Platoon dynamics (same as in synthesis)
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


def f_sigma(x, sigma):
    return f1(x) if sigma == 1 else f2_safe(x)


# ---------------------------------------------------------------------
# 2. Path-complete graph structure (loaded from file)
# ---------------------------------------------------------------------

# These will be filled in main() from the loaded npy
NODES = None
EDGES = None


def next_node(v_curr, sigma):
    """
    Deterministic node update along the path-complete graph:
    given current node v_curr and mode sigma, pick the FIRST edge
    (v_curr, sigma, v_next) from EDGES. If none found, stay in v_curr.
    """
    global EDGES
    for (u, s, v_next) in EDGES:
        if u == v_curr and s == sigma:
            return v_next
    return v_curr  # fallback (should not happen if graph is path-complete)


# ---------------------------------------------------------------------
# 3. Piecewise-quadratic V_v,r(x) and closure C_v(x,x')
# ---------------------------------------------------------------------

def phi_quad(x):
    """
    Quadratic basis for V_{v,r}(x):
        x = [x1, x2]
        phi(x) = [1, x1, x2, x1^2, x1*x2, x2^2]
    """
    x1, x2 = x
    return np.array([1.0, x1, x2, x1**2, x1 * x2, x2**2])


def region_index_gap(x, d_safe, d_unsafe):
    """
    Region partition exactly as in synthesis:

      gap = x2 - x1

      Region 0: gap <= d_safe
      Region 1: d_safe < gap <= d_unsafe
      Region 2: gap > d_unsafe

    Assumes n_regions = 3; if more are stored, you can generalize here.
    """
    x1, x2 = x
    gap = x2 - x1
    if gap <= d_safe:
        return 0
    elif gap <= d_unsafe:
        return 1
    else:
        return 2


def V_eval_pw(c_reg, v, x, d_safe, d_unsafe):
    """
    Evaluate V_{v,r(x)}(x) = c_{v,r}^T phi_quad(x), using the gap-based region.
      c_reg : shape (num_nodes, n_regions, 6)
    """
    r = region_index_gap(x, d_safe, d_unsafe)
    return float(c_reg[v, r, :] @ phi_quad(x))


def C_eval_pw(c_reg, v, x, xp, d_safe, d_unsafe):
    """
    Evaluate closure C_v(x,x') = V_{v,r(x)}(x') - V_{v,r(x)}(x),
    using the *same* region index r(x) for x and xp (matches synthesis).
    """
    r = region_index_gap(x, d_safe, d_unsafe)
    return float(c_reg[v, r, :] @ phi_quad(xp) -
                 c_reg[v, r, :] @ phi_quad(x))


# ---------------------------------------------------------------------
# 4. Simulation of trajectories under various switching patterns
# ---------------------------------------------------------------------

def simulate_trajectory(x0, T, switching_rule):
    """
    Simulate a trajectory of length T (T transitions, T+1 states).

    Inputs:
        x0 : np.array of shape (2,)
        T  : int, horizon
        switching_rule : function f(t, x_t) -> sigma in {1,2}

    Returns:
        X      : np.array of shape (T+1, 2), states
        sigmas : np.array of shape (T,), modes
    """
    X = np.zeros((T + 1, 2))
    sigmas = np.zeros(T, dtype=int)

    X[0, :] = x0
    for t in range(T):
        sigma = switching_rule(t, X[t, :])
        sigmas[t] = sigma
        X[t + 1, :] = f_sigma(X[t, :], sigma)

    return X, sigmas


def simulate_with_node_tracking_pwquad(
    x0,
    T,
    switching_rule,
    c_reg,
    d_safe,
    d_unsafe,
    node0=0,
):
    """
    Simulate trajectory and also track the graph node v_t,
    and closure values C_{v_t}(x_t, x_{t+1}).

    Inputs:
        x0 : initial state
        T  : horizon
        switching_rule : function f(t, x_t) -> sigma
        c_reg : np.array, shape (num_nodes, n_regions, basis_dim)
        d_safe, d_unsafe : gap thresholds
        node0 : initial node index (0 or 1)

    Returns:
        X        : (T+1,2)
        sigmas   : (T,)
        v_path   : (T+1,) node indices
        C_values : (T,)   C_{v_t}(x_t, x_{t+1})
        V_values : (T+1,) V_{v_t}(x_t)
    """
    global NODES
    X, sigmas = simulate_trajectory(x0, T, switching_rule)
    num_nodes, num_regions, _ = c_reg.shape

    v_path = np.zeros(T + 1, dtype=int)
    v_path[0] = node0

    C_values = np.zeros(T)
    V_values = np.zeros(T + 1)

    # Evaluate V at t=0
    V_values[0] = V_eval_pw(c_reg, v_path[0], X[0, :], d_safe, d_unsafe)

    for t in range(T):
        sigma = sigmas[t]
        v_curr = v_path[t]
        v_next = next_node(v_curr, sigma)

        C_values[t] = C_eval_pw(c_reg, v_curr, X[t, :], X[t + 1, :],
                                d_safe, d_unsafe)
        V_values[t + 1] = V_eval_pw(c_reg, v_next, X[t + 1, :],
                                    d_safe, d_unsafe)

        v_path[t + 1] = v_next

    return X, sigmas, v_path, C_values, V_values


# ---------------------------------------------------------------------
# 5. Switching rules for experiments
# ---------------------------------------------------------------------

def switching_always1(t, x):
    return 1


def switching_always2(t, x):
    return 2


def switching_periodic_1212(t, x):
    return 1 if t % 2 == 0 else 2


def switching_random(p=0.5):
    rng = np.random.default_rng()

    def rule(t, x):
        return 1 if rng.random() < p else 2

    return rule


# ---------------------------------------------------------------------
# 6. Plotting utilities
# ---------------------------------------------------------------------

def _dedup_legend(ax):
    """Helper to remove duplicate legend entries (by label)."""
    handles, labels = ax.get_legend_handles_labels()
    # Filter out empty labels
    pairs = [(h, l) for h, l in zip(handles, labels) if l]
    if not pairs:
        return
    by_label = {}
    for h, l in pairs:
        by_label[l] = h
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        framealpha=0.9,
        fontsize=8,
    )


def plot_state_trajectories(pl_params, trajectories, title_suffix="", ax=None):
    """
    trajectories: list of dicts with keys:
        'X'       : (T+1,2)
        'label'   : string
        'color'   : matplotlib color
    pl_params: dict from loaded data (zeta, d_safe, d_unsafe, domain ranges)

    We draw:
      - certified domain [x1_min,x1_max] x [x2_min,x2_max],
      - unsafe band (red): g = x2 - x1 <= d_unsafe,
      - safe band (green): g >= d_safe,
      - diagonal g=0.
    """
    d_safe = pl_params["d_safe"]
    d_unsafe = pl_params["d_unsafe"]

    x1_min_dom, x1_max_dom = pl_params["x1_range"]
    x2_min_dom, x2_max_dom = pl_params["x2_range"]

    # Plot frame for state trajectories (use domain ranges)
    x1_band_min, x1_band_max = x1_min_dom, x1_max_dom
    x2_band_min, x2_band_max = x2_min_dom, x2_max_dom

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True
    else:
        fig = ax.figure

    ax.set_xlim(x1_band_min, x1_band_max)
    ax.set_ylim(x2_band_min, x2_band_max)

    # Unsafe band: gap <= d_unsafe => x2 <= x1 + d_unsafe
    x1_line = np.linspace(x1_band_min, x1_band_max, 400)
    x2_unsafe_top = np.minimum(x1_line + d_unsafe, x2_band_max)

    ax.fill_between(
        x1_line,
        x1_line,            # x2 = x1 (gap = 0)
        x2_unsafe_top,      # x2 = x1 + d_unsafe (clipped)
        color="red",
        alpha=0.2,
        label=r"unsafe band ($g \leq d_{\mathrm{unsafe}}$)",
    )

    # Safe band: gap >= d_safe => x2 >= x1 + d_safe
    x2_safe_bottom = np.minimum(x1_line + d_safe, x2_band_max)
    #ax.fill_between(
    #    x1_line,
    #    x2_safe_bottom,     # x2 = x1 + d_safe
    #    x2_band_max,        # up to domain max
    #    color="green",
    #    alpha=0.15,
    #    label=r"safe band",
    #)

    # Plot diagonal (g = 0) over domain
    ax.plot(x1_line, x1_line, "k--", linewidth=1, label=r"$g = 0$")

    # Certified domain box outline
    ax.plot(
        [x1_min_dom, x1_max_dom, x1_max_dom, x1_min_dom, x1_min_dom],
        [x2_min_dom, x2_min_dom, x2_max_dom, x2_max_dom, x2_min_dom],
        "k-", linewidth=1, alpha=0.8, label="certified domain"
    )

    # Plot trajectories
    for traj in trajectories:
        X = traj["X"]
        ax.plot(
            X[:, 0],
            X[:, 1],
            color=traj.get("color", "blue"),
            linewidth=1.5,
            label=traj.get("label", None),
            alpha=0.9,
        )
        ax.plot(
            X[0, 0],
            X[0, 1],
            "o",
            color=traj.get("color", "blue"),
            markersize=4,
        )

    ax.set_xlabel(r"$x_1$ (follower velocity)")
    ax.set_ylabel(r"$x_2$ (leader velocity)")
    ax.set_title("Platoon state trajectories" + title_suffix)
    ax.grid(True)
    _dedup_legend(ax)

    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_gap_trajectories(pl_params, trajectories, title_suffix="", ax=None):
    """
    trajectories: list of dicts with keys:
        'X'     : (T+1,2)
        'label' : string
        'color' : matplotlib color

    Adds shaded unsafe and safe gap regions:
        - unsafe: 0 <= g <= d_unsafe
        - safe:   g >= d_safe

    g_max_dom = x2_max - x1_min (max possible gap in certified domain).
    """
    d_safe = pl_params["d_safe"]
    d_unsafe = pl_params["d_unsafe"]
    x1_min_dom, x1_max_dom = pl_params["x1_range"]
    x2_min_dom, x2_max_dom = pl_params["x2_range"]

    # Max gap allowed by the certified domain
    g_max_dom = x2_max_dom - x1_min_dom

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    # Precompute gap trajectories, clipped to the domain-based max gap
    gap_series = []
    for traj in trajectories:
        X = traj["X"]
        g_raw = X[:, 1] - X[:, 0]
        # Clip *for plotting* to [0, g_max_dom]
        g = np.clip(g_raw, 0.0, g_max_dom)
        gap_series.append((g, traj))

    if gap_series:
        T_plus_1 = gap_series[0][0].shape[0]
    else:
        T_plus_1 = 0
    t = np.arange(T_plus_1)

    # Shaded unsafe band (horizontal)
    ax.axhspan(
        0.0,
        min(d_unsafe, g_max_dom),
        color="red",
        alpha=0.08,
        label=r"unsafe gap region ($g \leq d_{\mathrm{unsafe}}$)",
    )

    # Shaded safe band (horizontal)
    #ax.axhspan(
    #    min(d_safe, g_max_dom),
    #    g_max_dom,
    #    color="green",
    #    alpha=0.05,
    #    label=r"safe gap region",
    #)

    # Plot trajectories (clipped gaps)
    for g, traj in gap_series:
        ax.plot(
            t,
            g,
            color=traj.get("color", "blue"),
            linewidth=1.5,
            label=traj.get("label", None),
            alpha=0.9,
        )

    # Reference lines for d_safe, d_unsafe (clipped to the domain gap)
    #ax.axhline(
    #    min(d_safe, g_max_dom),
    #    linestyle="--",
    #    color="green",
    #    linewidth=1,
    #    label=r"$d_{\mathrm{safe}}$",
    #)
    ax.axhline(
        min(d_unsafe, g_max_dom),
        linestyle="--",
        color="red",
        linewidth=1,
        label=r"$d_{\mathrm{unsafe}}$",
    )

    ax.set_xlim(0, T_plus_1 - 1 if T_plus_1 > 0 else 1)
    ax.set_ylim(0.0, g_max_dom)

    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"gap $g(t) = x_2(t) - x_1(t)$")
    ax.set_title("Gap evolution along trajectories" + title_suffix)
    ax.grid(True)
    _dedup_legend(ax)

    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_closure_trajectories(pl_params, trajectories, title_suffix="", ax=None):
    """
    trajectories: list of dicts with keys:
        'C_values' : (T,)
        'label'    : str
        'color'    : matplotlib color

    We add shaded regions:
      - green: C <= zeta
      - red:   C >  zeta
    """
    zeta = pl_params["zeta"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    # Collect min/max C over all trajectories for scaling and shading
    all_C = []
    for traj in trajectories:
        C_vals = traj["C_values"]
        if C_vals.size > 0:
            all_C.append(C_vals)
    if all_C:
        all_C_concat = np.concatenate(all_C)
        C_min = float(np.min(all_C_concat))
        C_max = float(np.max(all_C_concat))
    else:
        C_min, C_max = 0.0, zeta

    # Choose y-limits that include 0, zeta, and observed range
    y_min = min(C_min, 0.0, zeta - abs(zeta) * 0.2)
    y_max = max(C_max, zeta * 1.2)

    # Shaded safe region: C <= zeta
    ax.axhspan(
        y_min,
        zeta,
        color="green",
        alpha=0.05,
        label=r"closure region $C \leq \zeta$",
    )

    # Shaded unsafe region: C > zeta
    ax.axhspan(
        zeta,
        y_max,
        color="red",
        alpha=0.05,
        label=r"closure region $C > \zeta$",
    )

    # Plot trajectories
    for traj in trajectories:
        C_vals = traj["C_values"]
        t = np.arange(len(C_vals))
        ax.plot(
            t,
            C_vals,
            color=traj.get("color", "blue"),
            linewidth=1.5,
            label=traj.get("label", None),
            alpha=0.9,
        )

    ax.axhline(
        zeta,
        linestyle="--",
        color="black",
        linewidth=1,
        label=r"$\zeta$",
    )

    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"$C_{v_t}(x_t, x_{t+1})$")
    ax.set_title("Closure values along trajectories" + title_suffix)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    _dedup_legend(ax)

    if created_fig:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# 7. Main experimental routine
# ---------------------------------------------------------------------

def main():
    global NODES, EDGES

    # -----------------------------------------------------------------
    # Load PW-quadratic PC-CC data
    # -----------------------------------------------------------------
    data = load_pccc_pwquad()
    c_reg = np.array(data["coeffs_reg"])  # shape (num_nodes, n_regions, 6)
    zeta = float(data["zeta"])
    lambda1 = float(data["lambda1"])
    d_safe = float(data["d_safe"])
    d_unsafe = float(data["d_unsafe"])
    x1_range = tuple(data["x1_range"])
    x2_range = tuple(data["x2_range"])

    # Graph info
    NODES = list(data["nodes"])
    EDGES = [tuple(e) for e in data["edges"]]

    num_nodes, num_regions, basis_dim = c_reg.shape

    print("Loaded PW-quadratic PC-CC:")
    print("  c_reg shape =", c_reg.shape,
          "(num_nodes, num_regions, basis_dim)")
    print("  zeta =", zeta, "  lambda1 =", lambda1)
    print("  domain x1_range =", x1_range, " x2_range =", x2_range)
    print("  d_safe =", d_safe, "  d_unsafe =", d_unsafe)
    print("  num_nodes =", num_nodes,
          "num_regions =", num_regions,
          "basis_dim =", basis_dim)
    print("  nodes =", NODES)
    print("  edges =", EDGES)

    # Plotting params
    pl_params = {
        "zeta": zeta,
        "d_safe": d_safe,
        "d_unsafe": d_unsafe,
        "x1_range": x1_range,      # certified operating region used in synthesis
        "x2_range": x2_range,
    }

    x1_min, x1_max = x1_range
    x2_min, x2_max = x2_range

    # Output directory for figures (journal-ready PDFs)
    here = os.path.dirname(__file__)
    figs_dir = os.path.join(here, "..", "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Simulation setup: safe initial conditions (inside operating region)
    # with gap >= d_safe so they lie in the "green band"
    # -----------------------------------------------------------------
    x0_list = [
        np.array([0.5, 0.5 + d_safe]),
        np.array([1.0, 1.0 + d_safe]),
        np.array([0.2, 0.2 + d_safe + 0.5]),
    ]
    # Clip to domain if needed
    x0_list = [
        np.array([
            np.clip(x0[0], x1_min, x1_max),
            np.clip(x0[1], x2_min, x2_max)
        ]) for x0 in x0_list
    ]

    T = 30  # simulation horizon

    # Define some switching patterns
    switching_rules = [
        ("always mode 1", switching_always1, "tab:blue"),
        ("always mode 2", switching_always2, "tab:orange"),
        ("periodic 1-2", switching_periodic_1212, "tab:green"),
        ("random p=0.5", switching_random(0.5), "tab:red"),
    ]

    # Simulate all combinations of x0 and switching rules (safe starts)
    state_trajs = []
    gap_trajs = []
    closure_trajs = []

    for i, x0 in enumerate(x0_list):
        for name, rule, color in switching_rules:
            label = f"x0={x0}, {name}"
            X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking_pwquad(
                x0, T, rule, c_reg, d_safe, d_unsafe, node0=0
            )

            state_trajs.append({
                "X": X,
                "label": label,
                "color": color,
            })
            gap_trajs.append({
                "X": X,
                "label": label,
                "color": color,
            })
            closure_trajs.append({
                "C_values": C_vals,
                "label": label,
                "color": color,
            })

    # -----------------------------------------------------------------
    # Additional example: start inside X_u (red unsafe band)
    # -----------------------------------------------------------------
    # Choose an x0 so that gap g = x2 - x1 is strictly inside the unsafe band.
    # We use half of d_unsafe for a clear "red start".
    gap_u = 0.5 * d_unsafe
    x1_u = (x1_min + x1_max) / 2.0
    if x1_u + gap_u > x2_max:
        x1_u = x2_max - gap_u
    x2_u = x1_u + gap_u
    x1_u = max(x1_u, x1_min)
    x2_u = max(x2_u, x2_min)
    x0_unsafe = np.array([x1_u, x2_u])

    print("Unsafe initial condition (in red band): x0_unsafe =", x0_unsafe,
          "gap =", x0_unsafe[1] - x0_unsafe[0], "d_unsafe =", d_unsafe)

    unsafe_state_trajs = []
    unsafe_gap_trajs = []
    unsafe_closure_trajs = []

    for name, rule, color in switching_rules:
        label = f"x0 in X_u, {name}"
        X_u, sigmas_u, v_path_u, C_vals_u, V_vals_u = simulate_with_node_tracking_pwquad(
            x0_unsafe, T, rule, c_reg, d_safe, d_unsafe, node0=0
        )
        unsafe_state_trajs.append({
            "X": X_u,
            "label": label,
            "color": color,
        })
        unsafe_gap_trajs.append({
            "X": X_u,
            "label": label,
            "color": color,
        })
        unsafe_closure_trajs.append({
            "C_values": C_vals_u,
            "label": label,
            "color": color,
        })

    # Merge safe and unsafe trajectories for the main figures
    all_state_trajs = state_trajs + unsafe_state_trajs
    all_gap_trajs = gap_trajs + unsafe_gap_trajs
    all_closure_trajs = closure_trajs + unsafe_closure_trajs

    # -----------------------------------------------------------------
    # Individual figures
    # -----------------------------------------------------------------
    fig_states, ax_states = plot_state_trajectories(
        pl_params, all_state_trajs, title_suffix=" (PW-quadratic PC-CC)"
    )
    fig_states.savefig(
        os.path.join(figs_dir, "platoon_state_trajectories_pwquad.pdf"),
        bbox_inches="tight"
    )

    fig_gap, ax_gap = plot_gap_trajectories(
        pl_params, all_gap_trajs, title_suffix=" (PW-quadratic PC-CC)"
    )
    fig_gap.savefig(
        os.path.join(figs_dir, "platoon_gap_trajectories_pwquad.pdf"),
        bbox_inches="tight"
    )

    fig_closure, ax_closure = plot_closure_trajectories(
        pl_params, all_closure_trajs, title_suffix=" (PW-quadratic PC-CC)"
    )
    fig_closure.savefig(
        os.path.join(figs_dir, "platoon_closure_trajectories_pwquad.pdf"),
        bbox_inches="tight"
    )

    # -----------------------------------------------------------------
    # Combined 3-panel figure for journal:
    # single x0 in X_safe, multiple switchings
    # -----------------------------------------------------------------
    x0_rep = x0_list[0].copy()  # representative safe initial condition
    rep_state_trajs = []
    rep_gap_trajs = []
    rep_closure_trajs = []

    for name, rule, color in switching_rules:
        label = name  # cleaner labels for journal figure
        X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking_pwquad(
            x0_rep, T, rule, c_reg, d_safe, d_unsafe, node0=0
        )
        rep_state_trajs.append({
            "X": X,
            "label": label,
            "color": color,
        })
        rep_gap_trajs.append({
            "X": X,
            "label": label,
            "color": color,
        })
        rep_closure_trajs.append({
            "C_values": C_vals,
            "label": label,
            "color": color,
        })

    fig_comb, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Left: state-space trajectories
    plot_state_trajectories(
        pl_params, rep_state_trajs, title_suffix="", ax=axes[0]
    )
    axes[0].set_title("State trajectories")

    # Middle: gap evolution with shaded bands
    plot_gap_trajectories(
        pl_params, rep_gap_trajs, title_suffix="", ax=axes[1]
    )
    axes[1].set_title("Gap evolution")

    # Right: closure values vs time
    plot_closure_trajectories(
        pl_params, rep_closure_trajs, title_suffix="", ax=axes[2]
    )
    axes[2].set_title("Closure values")

    fig_comb.suptitle(
        "Two-car platoon with PW-quadratic Path-Complete Closure Certificates",
        y=1.02,
        fontsize=12
    )
    fig_comb.tight_layout()
    fig_comb.savefig(
        os.path.join(figs_dir, "platoon_combined_pwquad.pdf"),
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
