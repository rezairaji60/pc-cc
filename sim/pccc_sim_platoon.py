"""
pccc_sim_platoon.py

Simulation and plotting for Path-Complete Closure Certificates (PC-CC)
on the two-car platoon example, using the *linear* PC-CC synthesized
in pccc_synth_platoon_linear.py.

Figures produced (for paper case study):

1) State-space trajectories (x1, x2) overlaid with:
   - domain,
   - unsafe band (gap <= d_unsafe),
   - safe band (gap >= d_safe).

2) Gap evolution g(t) = x2(t) - x1(t) for several switching patterns.

3) Closure values along trajectories:
   C_{v_t}(x_t, x_{t+1}) vs time, with the threshold zeta.

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
# ---------------------------------------------------------------------

def plot_state_trajectories(pl_params, trajectories, title_suffix=""):
    """
    trajectories: list of dicts with keys:
        'X'       : (T+1,2)
        'label'   : string
        'color'   : matplotlib color
    pl_params: dict from loaded data (zeta, d_safe, d_unsafe, domain ranges)
    """
    d_safe = pl_params["d_safe"]
    d_unsafe = pl_params["d_unsafe"]
    x1_min, x1_max = pl_params["x1_range"]
    x2_min, x2_max = pl_params["x2_range"]

    fig, ax = plt.subplots(figsize=(6, 5))

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
        label="unsafe band (gap ≤ d_unsafe)",
    )

    # Safe band: gap >= d_safe => x2 >= x1 + d_safe
    ax.fill_between(
        x1_line,
        x1_line + d_safe,
        x2_max,
        color="green",
        alpha=0.15,
        label="safe band (gap ≥ d_safe)",
    )

    # Plot diagonal (gap=0)
    ax.plot(x1_line, x1_line, "k--", linewidth=1, label="gap = 0")

    # Plot trajectories
    for traj in trajectories:
        X = traj["X"]
        ax.plot(X[:, 0], X[:, 1], color=traj.get("color", "blue"),
                linewidth=1.5, label=traj.get("label", None), alpha=0.9)
        ax.plot(X[0, 0], X[0, 1], "o", color=traj.get("color", "blue"),
                markersize=4)

    ax.set_xlabel(r"$x_1$ (follower velocity)")
    ax.set_ylabel(r"$x_2$ (leader velocity)")
    ax.set_title("Platoon state trajectories" + title_suffix)
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()


def plot_gap_trajectories(pl_params, trajectories, title_suffix=""):
    """
    trajectories: list of dicts with keys:
        'X'     : (T+1,2)
        'label' : string
        'color' : matplotlib color
    """
    d_safe = pl_params["d_safe"]
    d_unsafe = pl_params["d_unsafe"]

    fig, ax = plt.subplots(figsize=(6, 4))

    for traj in trajectories:
        X = traj["X"]
        g = X[:, 1] - X[:, 0]
        ax.plot(g, color=traj.get("color", "blue"),
                linewidth=1.5, label=traj.get("label", None), alpha=0.9)

    ax.axhline(d_safe, linestyle="--", color="green", linewidth=1,
               label=r"$d_{\mathrm{safe}}$")
    ax.axhline(d_unsafe, linestyle="--", color="red", linewidth=1,
               label=r"$d_{\mathrm{unsafe}}$")

    ax.set_xlabel("time step t")
    ax.set_ylabel(r"gap $x_2(t) - x_1(t)$")
    ax.set_title("Gap evolution along trajectories" + title_suffix)
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()


def plot_closure_trajectories(pl_params, trajectories, title_suffix=""):
    """
    trajectories: list of dicts with keys:
        'C_values' : (T,)
        'label'    : str
        'color'    : matplotlib color
    """
    zeta = pl_params["zeta"]

    fig, ax = plt.subplots(figsize=(6, 4))

    for traj in trajectories:
        C_vals = traj["C_values"]
        ax.plot(C_vals, color=traj.get("color", "blue"),
                linewidth=1.5, label=traj.get("label", None), alpha=0.9)

    ax.axhline(zeta, linestyle="--", color="black", linewidth=1,
               label=r"$\zeta$")
    ax.set_xlabel("time step t")
    ax.set_ylabel(r"$C_{v_t}(x_t, x_{t+1})$")
    ax.set_title("Closure values along trajectories" + title_suffix)
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()


# ---------------------------------------------------------------------
# 7. Main experimental routine
# ---------------------------------------------------------------------

def main():
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

    # Initial conditions for experiments
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

    # Simulate
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

    # Plot figures
    plot_state_trajectories(pl_params, state_trajs,
                            title_suffix=" (linear PC-CC)")
    plot_gap_trajectories(pl_params, gap_trajs,
                          title_suffix=" (linear PC-CC)")
    plot_closure_trajectories(pl_params, closure_trajs,
                              title_suffix=" (linear PC-CC)")

    plt.show()


if __name__ == "__main__":
    main()
