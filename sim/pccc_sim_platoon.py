"""
Author: Reza Iraji
Date:   November 2025
File:   pccc_sim_platoon.py

Simulation and plotting for Path-Complete Closure Certificates (PC-CC)
on the two-car platoon example, using the *linear* PC-CC synthesized
in pccc_synth_platoon.py.

Figures produced:

1) Separate figures:
   - State-space trajectories (x1, x2) with domain, unsafe band, safe band.
   - Gap evolution g(t) = x2(t) - x1(t) with unsafe/safe shaded regions.
   - Closure values C_{v_t}(x_t, x_{t+1}) with threshold zeta.

2) Combined 3-panel figures:
   - Safe initial condition: standard example.
   - Unsafe initial condition: persistence illustration (finite visits).

All legends are placed externally (to the right) for better visibility.
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
    Deterministic node update along the path-complete graph:
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
    Simulate trajectory and track graph node v_t and closure values.

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

    V_values[0] = V_eval(c[v_path[0], :], X[0, :])

    for t in range(T):
        sigma = sigmas[t]
        v_next = next_node(v_path[t], sigma)

        C_values[t] = C_eval(c[v_path[t], :], X[t, :], X[t + 1, :])
        V_values[t + 1] = V_eval(c[v_next, :], X[t + 1, :])

        v_path[t + 1] = v_next

    return X, sigmas, v_path, C_values, V_values


# ---------------------------------------------------------------------
# 5. Switching rules
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
# 6. Plotting utilities (with external legends)
# ---------------------------------------------------------------------

def _dedup_legend(ax, outside=True):
    """
    Helper to remove duplicate legend entries (by label) and place the
    legend externally to the right for better visibility.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        if outside:
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                framealpha=0.9,
            )
        else:
            ax.legend(by_label.values(), by_label.keys(), loc="best")


def plot_state_trajectories(pl_params, trajectories, title_suffix="", ax=None):
    """
    trajectories: list of dicts with keys:
        'X'       : (T+1,2)
        'label'   : string
        'color'   : matplotlib color
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

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    x1_line = np.linspace(x1_min, x1_max, 200)

    # Unsafe band: gap <= d_unsafe => x2 <= x1 + d_unsafe
    ax.fill_between(
        x1_line,
        x1_line,                # g = 0
        x1_line + d_unsafe,
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

    # Diagonal g = 0
    ax.plot(x1_line, x1_line, "k--", linewidth=1, label=r"$g = 0$")

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
    _dedup_legend(ax, outside=True)

    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_gap_trajectories(pl_params, trajectories, title_suffix="", ax=None):
    """
    trajectories: list of dicts with keys:
        'X'     : (T+1,2)
        'label' : string
        'color' : matplotlib color
    """
    d_safe = pl_params["d_safe"]
    d_unsafe = pl_params["d_unsafe"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    gap_series = []
    max_g = 0.0
    for traj in trajectories:
        X = traj["X"]
        g = X[:, 1] - X[:, 0]
        gap_series.append((g, traj))
        max_g = max(max_g, np.max(g))

    max_g = max(max_g, d_safe * 1.1, d_unsafe * 1.1)
    T_plus_1 = gap_series[0][0].shape[0] if gap_series else 0
    t = np.arange(T_plus_1)

    ax.axhspan(
        0.0,
        d_unsafe,
        color="red",
        alpha=0.08,
        label=r"unsafe gap region ($g \leq d_{\mathrm{unsafe}}$)",
    )

    ax.axhspan(
        d_safe,
        max_g,
        color="green",
        alpha=0.05,
        label=r"safe gap region ($g \geq d_{\mathrm{safe}}$)",
    )

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
    _dedup_legend(ax, outside=True)

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
    _dedup_legend(ax, outside=True)

    if created_fig:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# 7. Main experimental routine
# ---------------------------------------------------------------------

def main():
    # Load PC-CC data
    data = load_pccc_linear()
    c = data["coeffs"]
    zeta = data["zeta"]
    lambda1 = data["lambda1"]
    print("Loaded linear PC-CC:")
    print("  coeffs c =\n", c)
    print("  zeta =", zeta, "  lambda1 =", lambda1)

    pl_params = {
        "zeta": zeta,
        "d_safe": data["d_safe"],
        "d_unsafe": data["d_unsafe"],
        "x1_range": data["x1_range"],
        "x2_range": data["x2_range"],
    }

    here = os.path.dirname(__file__)
    figs_dir = os.path.join(here, "..", "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Simulation setup: safe and unsafe initial conditions
    # -----------------------------------------------------------------
    # Safe initial conditions (g >= d_safe in synthesis domain)
    x0_list_safe = [
        np.array([0.5, 2.0]),
        np.array([1.0, 2.5]),
        np.array([0.2, 1.5]),
    ]

    # Unsafe initial conditions (g <= d_unsafe) to illustrate persistence
    # These satisfy 0 <= x2 - x1 <= d_unsafe
    x0_list_unsafe = [
        np.array([1.0, 1.15]),   # gap = 0.15
        np.array([0.8, 0.95]),   # gap = 0.15
    ]

    T = 30  # simulation horizon

    switching_rules = [
        ("always mode 1", switching_always1, "tab:blue"),
        ("always mode 2", switching_always2, "tab:orange"),
        ("periodic 1-2", switching_periodic_1212, "tab:green"),
        ("random p=0.5", switching_random(0.5), "tab:red"),
    ]

    state_trajs = []
    gap_trajs = []
    closure_trajs = []

    # Safe initial conditions
    for x0 in x0_list_safe:
        for name, rule, color in switching_rules:
            label = f"safe x0={x0}, {name}"
            X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking(
                x0, T, rule, c, node0=0
            )
            state_trajs.append({"X": X, "label": label, "color": color})
            gap_trajs.append({"X": X, "label": label, "color": color})
            closure_trajs.append({"C_values": C_vals, "label": label, "color": color})

    # Unsafe initial conditions (persistence example)
    for x0 in x0_list_unsafe:
        for name, rule, color in switching_rules:
            label = f"unsafe x0={x0}, {name}"
            X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking(
                x0, T, rule, c, node0=0
            )
            state_trajs.append({"X": X, "label": label, "color": color})
            gap_trajs.append({"X": X, "label": label, "color": color})
            closure_trajs.append({"C_values": C_vals, "label": label, "color": color})

    # -----------------------------------------------------------------
    # Individual figures (all trajectories together)
    # -----------------------------------------------------------------
    fig_states, ax_states = plot_state_trajectories(
        pl_params, state_trajs, title_suffix=" (linear PC-CC)"
    )
    fig_states.savefig(
        os.path.join(figs_dir, "platoon_state_trajectories_linear.pdf"),
        bbox_inches="tight"
    )

    fig_gap, ax_gap = plot_gap_trajectories(
        pl_params, gap_trajs, title_suffix=" (linear PC-CC)"
    )
    fig_gap.savefig(
        os.path.join(figs_dir, "platoon_gap_trajectories_linear.pdf"),
        bbox_inches="tight"
    )

    fig_closure, ax_closure = plot_closure_trajectories(
        pl_params, closure_trajs, title_suffix=" (linear PC-CC)"
    )
    fig_closure.savefig(
        os.path.join(figs_dir, "platoon_closure_trajectories_linear.pdf"),
        bbox_inches="tight"
    )

    # -----------------------------------------------------------------
    # Combined 3-panel figure: representative safe initial condition
    # -----------------------------------------------------------------
    x0_rep_safe = np.array([0.5, 2.0])
    rep_state_trajs_safe = []
    rep_gap_trajs_safe = []
    rep_closure_trajs_safe = []

    for name, rule, color in switching_rules:
        label = name
        X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking(
            x0_rep_safe, T, rule, c, node0=0
        )
        rep_state_trajs_safe.append({"X": X, "label": label, "color": color})
        rep_gap_trajs_safe.append({"X": X, "label": label, "color": color})
        rep_closure_trajs_safe.append({"C_values": C_vals, "label": label, "color": color})

    fig_comb_safe, axes_safe = plt.subplots(1, 3, figsize=(16, 4))

    plot_state_trajectories(pl_params, rep_state_trajs_safe, title_suffix="", ax=axes_safe[0])
    axes_safe[0].set_title("State trajectories (safe $x_0$)")

    plot_gap_trajectories(pl_params, rep_gap_trajs_safe, title_suffix="", ax=axes_safe[1])
    axes_safe[1].set_title("Gap evolution (safe $x_0$)")

    plot_closure_trajectories(pl_params, rep_closure_trajs_safe, title_suffix="", ax=axes_safe[2])
    axes_safe[2].set_title("Closure values (safe $x_0$)")

    fig_comb_safe.suptitle(
        "Two-car platoon with linear Path-Complete Closure Certificates (safe initial condition)",
        y=1.05,
        fontsize=12,
    )
    fig_comb_safe.tight_layout()
    fig_comb_safe.savefig(
        os.path.join(figs_dir, "platoon_combined_linear_safe.pdf"),
        bbox_inches="tight"
    )

    # -----------------------------------------------------------------
    # Combined 3-panel figure: representative unsafe initial condition
    # (persistence: finitely many visits to unsafe band)
    # -----------------------------------------------------------------
    x0_rep_unsafe = x0_list_unsafe[0]
    rep_state_trajs_unsafe = []
    rep_gap_trajs_unsafe = []
    rep_closure_trajs_unsafe = []

    for name, rule, color in switching_rules:
        label = name
        X, sigmas, v_path, C_vals, V_vals = simulate_with_node_tracking(
            x0_rep_unsafe, T, rule, c, node0=0
        )
        rep_state_trajs_unsafe.append({"X": X, "label": label, "color": color})
        rep_gap_trajs_unsafe.append({"X": X, "label": label, "color": color})
        rep_closure_trajs_unsafe.append({"C_values": C_vals, "label": label, "color": color})

    fig_comb_unsafe, axes_unsafe = plt.subplots(1, 3, figsize=(16, 4))

    plot_state_trajectories(pl_params, rep_state_trajs_unsafe, title_suffix="", ax=axes_unsafe[0])
    axes_unsafe[0].set_title("State trajectories (unsafe $x_0$)")

    plot_gap_trajectories(pl_params, rep_gap_trajs_unsafe, title_suffix="", ax=axes_unsafe[1])
    axes_unsafe[1].set_title("Gap evolution (unsafe $x_0$)")

    plot_closure_trajectories(pl_params, rep_closure_trajs_unsafe, title_suffix="", ax=axes_unsafe[2])
    axes_unsafe[2].set_title("Closure values (unsafe $x_0$)")

    fig_comb_unsafe.suptitle(
        "Two-car platoon with linear Path-Complete Closure Certificates (unsafe initial condition, persistence)",
        y=1.05,
        fontsize=12,
    )
    fig_comb_unsafe.tight_layout()
    fig_comb_unsafe.savefig(
        os.path.join(figs_dir, "platoon_combined_linear_unsafe.pdf"),
        bbox_inches="tight"
    )

    # Show figures interactively (optional)
    plt.show()


if __name__ == "__main__":
    main()
