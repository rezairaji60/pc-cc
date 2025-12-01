"""
Author: Reza Iraji
Date:   November 2025
File:   pccc_sim_platoon.py

Simulation and plotting for Path-Complete Closure Certificates (PC-CC)
on the two-car platoon example, using the *linear* PC-CC synthesized
in pccc_synth_platoon.py.

Figures produced (for paper / journal case study):

1) Separate figures:
   - State-space trajectories (x1, x2) with domain, unsafe band, safe band.
   - Gap evolution g(t) = x2(t) - x1(t) with unsafe/safe shaded regions.
   - Closure values C_{v_t}(x_t, x_{t+1}) with threshold zeta.

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
# 0. Load linear PC-CC parameters
# ---------------------------------------------------------------------

def load_pccc_linear(filename="pccc_platoon_coeffs_linear.npy"):
    """
    Load the synthesized linear PC-CC coefficients and parameters.
    Expects the file to live one level above this script (repo root).
    """
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", filename)
    data = np.load(path, allow_pickle=True).item()
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
# 2. Path-complete graph structure (same as in synthesis)
# ---------------------------------------------------------------------

NODES = [0, 1]
EDGES = [
    (0, 1, 0),  # v1 --1--> v1
    (0, 1, 1),  # v1 --1--> v2
    (0, 2, 1),  # v1 --2--> v2
    (1, 2, 0),  # v2 --2--> v1
]


# ---------------------------------------------------------------------
# 3. Linear V_v(x) and closure C_v(x,x') = V_v(x') - V_v(x)
# ---------------------------------------------------------------------

def phi(x):
    """
    Linear basis for V_v(x):
        x = [x1, x2]
        phi(x) = [1, x1, x2]
    """
    x1, x2 = x
    return np.array([1.0, x1, x2])


def V_eval(c_v, x):
    """
    Evaluate V_v(x) = c_v^T phi(x).
    c_v is a length-3 numpy vector.
    """
    return float(c_v @ phi(x))


def C_eval(c_v, x, xp):
    """
    Evaluate C_v(x,x') = V_v(x') - V_v(x).
    """
    return V_eval(c_v, xp) - V_eval(c_v, x)


def next_node(v_curr, sigma):
    """
    Simple deterministic node update along the path-complete graph:
    given current node v_curr and mode sigma, pick the FIRST edge
    (v_curr, sigma, v_next) from EDGES. If none found, stay in v_curr.
    """
    for (u, s, v_next) in EDGES:
        if u == v_curr and s == sigma:
            return v_next
    return v_curr  # fallback (should not happen if graph is path-complete)


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


def simulate_with_node_tracking(x0, T, switching_rule, c, node0=0):
    """
    Simulate trajectory and also track the graph node v_t,
    and closure values C_{v_t}(x_t, x_{t+1}).

    Inputs:
        x0 : initial state
        T  : horizon
        switching_rule : function f(t, x_t) -> sigma
        c   : np.array of shape (2,3), PC-CC coefficients (for nodes v1,v2)
        node0 : initial node index (0 or 1)

    Returns:
        X        : (T+1,2)
        sigmas   : (T,)
        v_path   : (T+1,) node indices
        C_values : (T,)   C_{v_t}(x_t, x_{t+1})
        V_values : (T+1,) V_{v_t}(x_t)
    """
    X, sigmas = simulate_trajectory(x0, T, switching_rule)
    v_path = np.zeros(T + 1, dtype=int)
    v_path[0] = node0

    C_values = np.zeros(T)
    V_values = np.zeros(T + 1)

    # Evaluate V at t=0
    V_values[0] = V_eval(c[v_path[0], :], X[0, :])

    for t in range(T):
        sigma = sigmas[t]
        v_next = next_node(v_path[t], sigma)

        C_values[t] = C_eval(c[v_path[t], :], X[t, :], X[t + 1, :])
        V_values[t + 1] = V_eval(c[v_next, :], X[t + 1, :])

        v_path[t + 1] = v_next

    return X, sigmas, v_path, C_values, V_values


# ---------------------------------------------------------------------
# 5. Define some switching rules for experiments
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
#    (axis-aware so we can reuse for combined figure)
# ---------------------------------------------------------------------

def _dedup_legend(ax):
    """Helper to remove duplicate legend entries (by label) and
    place the legend in the bottom-right of the axes."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower right",   # bottom-right inside the axes
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

    If ax is None, creates a new figure; otherwise draws on the given axis.
    Returns (fig, ax).
    """
    d_safe = pl_params["d_safe"]
    d_unsafe = pl_params["d_unsafe"]
    x1_min, x1_max = pl_params["x1_range"]
    x2_min, x2_max = pl_params["x2_range"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True
    else:
        fig = ax.figure

    # Plot domain box
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    # Unsafe band: gap <= d_unsafe => x2 <= x1 + d_unsafe
    x1_line = np.linspace(x1_min, x1_max, 200)
    ax.fill_between(
        x1_line,
        x1_line,                # x2 = x1 (gap = 0)
        x1_line + d_unsafe,     # x2 = x1 + d_unsafe
        color="red",
        alpha=0.2,
        label=r"unsafe band ($g \leq d_{\mathrm{unsafe}}$)",
    )

    # Safe band: gap >= d_safe => x2 >= x1 + d_safe
    ax.fill_between(
        x1_line,
        x1_line + d_safe,
        x2_max,
        color="green",
        alpha=0.15,
        label=r"safe band ($g \geq d_{\mathrm{safe}}$)",
    )

    # Plot diagonal (gap=0)
    ax.plot(x1_line, x1_line, "k--", linewidth=1, label=r"$g = 0$")

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
    """
    d_safe = pl_params["d_safe"]
    d_unsafe = pl_params["d_unsafe"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    # Precompute gap trajectories to get a good y-range
    gap_series = []
    max_g = 0.0
    for traj in trajectories:
        X = traj["X"]
        g = X[:, 1] - X[:, 0]
        gap_series.append((g, traj))
        max_g = max(max_g, np.max(g))

    # Avoid degenerate scaling
    max_g = max(max_g, d_safe * 1.1, d_unsafe * 1.1)

    T_plus_1 = gap_series[0][0].shape[0] if gap_series else 0
    t = np.arange(T_plus_1)

    # Shaded unsafe band (horizontal)
    ax.axhspan(
        0.0,
        d_unsafe,
        color="red",
        alpha=0.08,
        label=r"unsafe gap region ($g \leq d_{\mathrm{unsafe}}$)",
    )

    # Shaded safe band (horizontal)
    ax.axhspan(
        d_safe,
        max_g,
        color="green",
        alpha=0.05,
        label=r"safe gap region ($g \geq d_{\mathrm{safe}}$)",
    )

    # Plot trajectories
    for g, traj in gap_series:
        ax.plot(
            t,
            g,
            color=traj.get("color", "blue"),
            linewidth=1.5,
            label=traj.get("label", None),
            alpha=0.9,
        )

    ax.axhline(
        d_safe,
        linestyle="--",
        color="green",
        linewidth=1,
        label=r"$d_{\mathrm{safe}}$",
    )
    ax.axhline(
        d_unsafe,
        linestyle="--",
        color="red",
        linewidth=1,
        label=r"$d_{\mathrm{unsafe}}$",
    )

    ax.set_xlim(0, T_plus_1 - 1 if T_plus_1 > 0 else 1)
    ax.set_ylim(0.0, max_g * 1.05)

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
    """
    zeta = pl_params["zeta"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

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
    ax.grid(True)
    _dedup_legend(ax)

    if created_fig:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# 7. Main experimental routine
# ---------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------
    # Load PC-CC data
    # -----------------------------------------------------------------
    data = load_pccc_linear()
    c = data["coeffs"]        # shape (2,3)
    zeta = data["zeta"]
    lambda1 = data["lambda1"]
    print("Loaded linear PC-CC:")
    print("  coeffs c =\n", c)
    print("  zeta =", zeta, "  lambda1 =", lambda1)

    # For convenience
    pl_params = {
        "zeta": zeta,
        "d_safe": data["d_safe"],
        "d_unsafe": data["d_unsafe"],
        "x1_range": data["x1_range"],
        "x2_range": data["x2_range"],
    }

    d_unsafe = pl_params["d_unsafe"]
    x1_min, x1_max = pl_params["x1_range"]
    x2_min, x2_max = pl_params["x2_range"]

    # Output directory for figures (journal-ready PDFs)
    here = os.path.dirname(__file__)
    figs_dir = os.path.join(here, "..", "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Simulation setup: safe initial conditions
    # -----------------------------------------------------------------
    x0_list = [
        np.array([0.5, 2.0]),
        np.array([1.0, 2.5]),
        np.array([0.2, 1.5]),
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
            X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking(
                x0, T, rule, c, node0=0
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
    # Additional persistence example: start inside X_u (red unsafe band)
    # -----------------------------------------------------------------
    # Choose an x0 so that gap g = x2 - x1 is strictly inside the unsafe band.
    # We use half of d_unsafe for a clear "red start" that leaves the band later.
    gap_u = 0.5 * d_unsafe
    # pick x1 in the middle of the admissible range, then adjust if needed
    x1_u = (x1_min + x1_max) / 2.0
    if x1_u + gap_u > x2_max:
        x1_u = x2_max - gap_u
    x2_u = x1_u + gap_u
    # Ensure we respect lower bounds as well
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
        X_u, sigmas_u, v_path_u, C_vals_u, V_vals_u = simulate_with_node_tracking(
            x0_unsafe, T, rule, c, node0=0
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
    # Individual figures (diagnostics + paper/supp material)
    # -----------------------------------------------------------------
    fig_states, ax_states = plot_state_trajectories(
        pl_params, all_state_trajs, title_suffix=" (linear PC-CC)"
    )
    fig_states.savefig(
        os.path.join(figs_dir, "platoon_state_trajectories_linear.pdf"),
        bbox_inches="tight"
    )

    fig_gap, ax_gap = plot_gap_trajectories(
        pl_params, all_gap_trajs, title_suffix=" (linear PC-CC)"
    )
    fig_gap.savefig(
        os.path.join(figs_dir, "platoon_gap_trajectories_linear.pdf"),
        bbox_inches="tight"
    )

    fig_closure, ax_closure = plot_closure_trajectories(
        pl_params, all_closure_trajs, title_suffix=" (linear PC-CC)"
    )
    fig_closure.savefig(
        os.path.join(figs_dir, "platoon_closure_trajectories_linear.pdf"),
        bbox_inches="tight"
    )

    # -----------------------------------------------------------------
    # Combined 3-panel figure for journal: single x0 in X_safe, multiple switchings
    # -----------------------------------------------------------------
    x0_rep = np.array([0.5, 2.0])
    rep_state_trajs = []
    rep_gap_trajs = []
    rep_closure_trajs = []

    for name, rule, color in switching_rules:
        label = name  # cleaner labels for journal figure
        X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking(
            x0_rep, T, rule, c, node0=0
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
        "Two-car platoon with linear Path-Complete Closure Certificates",
        y=1.02,
        fontsize=12
    )
    fig_comb.tight_layout()
    fig_comb.savefig(
        os.path.join(figs_dir, "platoon_combined_linear.pdf"),
        bbox_inches="tight"
    )

    # Show figures interactively if desired
    plt.show()


if __name__ == "__main__":
    main()
