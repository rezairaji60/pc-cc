"""
Simulation and plotting for the nonlinear platoon case study using the
polynomial PC-CC

    C_{p,q}(x,y) = (y2 - y1) - 0.40001.

This script replaces the legacy PW-quadratic/zeta plotting script.  It does
not need a coefficient .npy file for plotting, because the final feasible
polynomial certificate is explicit. It should be described in the paper as a
linear feasible point of the quadratic SOS/TSSOS template.

Run from repository root, for example:

    python sim/pccc_sim_platoon_polynomial_pccc.py

It writes:

    Figs/platoon_combined_polynomial_pccc.pdf
    Figs/platoon_state_polynomial_pccc.pdf
    Figs/platoon_gap_polynomial_pccc.pdf
    Figs/platoon_closure_polynomial_pccc.pdf
"""

from __future__ import annotations

import os
import random as py_random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Problem data
# ---------------------------------------------------------------------

D_VF = 0.4
CERT_OFFSET = 0.40001
X1_RANGE = (0.0, 3.0)
X2_RANGE = (0.0, 5.1)
T_HORIZON = 30
RANDOM_SEED = 7


@dataclass
class Trajectory:
    X: np.ndarray
    sigmas: np.ndarray
    label: str
    color: str


def gap(x: np.ndarray) -> float:
    return float(x[1] - x[0])


def polynomial_certificate(x: np.ndarray, y: np.ndarray) -> float:
    """C_{p,q}(x,y) = g(y) - 0.40001, independent of p,q."""
    return gap(y) - CERT_OFFSET


# ---------------------------------------------------------------------
# 2. Platoon dynamics
# ---------------------------------------------------------------------


def f1(x: np.ndarray) -> np.ndarray:
    x1, x2 = x
    return np.array([
        0.01 * x2 + 0.9 * x1 - 0.02 * x1**2,
        2.0 + 0.8 * x2 - 0.04 * x2**2,
    ])


def f2(x: np.ndarray) -> np.ndarray:
    x1, x2 = x
    return np.array([
        0.9 * x1 - 0.02 * x1**2,
        2.0 + 0.8 * x2 - 0.04 * x2**2,
    ])


def f_sigma(x: np.ndarray, sigma: int) -> np.ndarray:
    if sigma == 1:
        return f1(x)
    if sigma == 2:
        return f2(x)
    raise ValueError(f"mode must be 1 or 2, got {sigma}")


# ---------------------------------------------------------------------
# 3. Switching rules
# ---------------------------------------------------------------------


def switching_always1(t: int, x: np.ndarray) -> int:
    return 1


def switching_always2(t: int, x: np.ndarray) -> int:
    return 2


def switching_periodic_1212(t: int, x: np.ndarray) -> int:
    return 1 if t % 2 == 0 else 2


def switching_random(p: float = 0.5, seed: int = RANDOM_SEED) -> Callable[[int, np.ndarray], int]:
    """Deterministic random switching using Python stdlib only.

    Avoids numpy.random because some Windows Application Control policies
    block NumPy's compiled random DLLs such as _pcg64.
    """
    rng = py_random.Random(seed)

    def rule(t: int, x: np.ndarray) -> int:
        return 1 if rng.random() < p else 2

    return rule


# ---------------------------------------------------------------------
# 4. Simulation
# ---------------------------------------------------------------------


def simulate_trajectory(x0: np.ndarray, T: int, rule: Callable[[int, np.ndarray], int]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((T + 1, 2), dtype=float)
    sigmas = np.zeros(T, dtype=int)
    X[0] = x0
    for t in range(T):
        sigmas[t] = rule(t, X[t])
        X[t + 1] = f_sigma(X[t], int(sigmas[t]))
    return X, sigmas


def make_trajectories() -> List[Trajectory]:
    # Representative initial condition used in the paper: g(x0)=0.3.
    x0 = np.array([0.5, 0.8], dtype=float)
    rules = [
        ("always mode 1", switching_always1, "tab:blue"),
        ("always mode 2", switching_always2, "tab:orange"),
        ("periodic 1-2", switching_periodic_1212, "tab:green"),
        ("random p=0.5", switching_random(0.5, RANDOM_SEED), "tab:red"),
    ]
    trajectories: List[Trajectory] = []
    for label, rule, color in rules:
        X, sigmas = simulate_trajectory(x0, T_HORIZON, rule)
        trajectories.append(Trajectory(X=X, sigmas=sigmas, label=label, color=color))
    return trajectories


# ---------------------------------------------------------------------
# 5. Plot helpers
# ---------------------------------------------------------------------


def _dedup_legend(ax, loc: str = "best") -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label: Dict[str, object] = {}
    for h, l in zip(handles, labels):
        if l and l not in by_label:
            by_label[l] = h
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc=loc, framealpha=0.9, fontsize=8)


def plot_state_trajectories(trajectories: List[Trajectory], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    x1_min, x1_max = X1_RANGE
    x2_min, x2_max = X2_RANGE
    x1_line = np.linspace(x1_min, x1_max, 400)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    # Certified domain outline.
    ax.plot(
        [x1_min, x1_max, x1_max, x1_min, x1_min],
        [x2_min, x2_min, x2_max, x2_max, x2_min],
        "k-", linewidth=1, alpha=0.7, label="certified domain",
    )

    # Physical lower boundary g=0 and finite-visit band 0 <= g <= 0.4.
    ax.plot(x1_line, x1_line, "k--", linewidth=1, label=r"$g=0$")
    ax.fill_between(
        x1_line,
        x1_line,
        np.minimum(x1_line + D_VF, x2_max),
        color="red",
        alpha=0.16,
        label=r"finite-visit set ($g\leq0.4$)",
    )

    for traj in trajectories:
        X = traj.X
        ax.plot(X[:, 0], X[:, 1], color=traj.color, linewidth=1.7, label=traj.label)
        ax.plot(X[0, 0], X[0, 1], "o", color=traj.color, markersize=4)

    ax.set_xlabel(r"$x_1$ (follower velocity)")
    ax.set_ylabel(r"$x_2$ (leader velocity)")
    ax.set_title("State trajectories")
    ax.grid(True, alpha=0.5)
    _dedup_legend(ax, loc="lower right")
    return ax


def plot_gap_trajectories(trajectories: List[Trajectory], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    for traj in trajectories:
        gvals = traj.X[:, 1] - traj.X[:, 0]
        t = np.arange(len(gvals))
        ax.plot(t, gvals, color=traj.color, linewidth=1.7, label=traj.label)

    ax.axhspan(0.0, D_VF, color="red", alpha=0.10, label=r"finite-visit region ($g\leq0.4$)")
    ax.axhline(D_VF, linestyle="--", color="red", linewidth=1, label=r"$0.4$")
    ax.set_xlim(0, T_HORIZON)
    ax.set_ylim(0, X2_RANGE[1] - X1_RANGE[0])
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"gap $g(t)=x_2(t)-x_1(t)$")
    ax.set_title("Gap evolution")
    ax.grid(True, alpha=0.5)
    _dedup_legend(ax, loc="lower right")
    return ax


def plot_certificate_values(trajectories: List[Trajectory], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    all_vals = []
    for traj in trajectories:
        C_vals = np.array([
            polynomial_certificate(traj.X[t], traj.X[t + 1])
            for t in range(len(traj.X) - 1)
        ])
        all_vals.append(C_vals)
        t = np.arange(len(C_vals))
        ax.plot(t, C_vals, color=traj.color, linewidth=1.7, label=traj.label)

    all_concat = np.concatenate(all_vals) if all_vals else np.array([0.0])
    y_min = min(-0.05, float(np.min(all_concat)) - 0.1)
    y_max = max(1.5, float(np.max(all_concat)) + 0.1)

    ax.axhspan(0.0, y_max, color="green", alpha=0.07, label=r"certified one-step region ($C\geq0$)")
    if y_min < 0:
        ax.axhspan(y_min, 0.0, color="red", alpha=0.07, label=r"$C<0$")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1, label=r"$0$")

    ax.set_xlim(0, T_HORIZON - 1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"$C(x(t),x(t+1))=g(x(t+1))-0.40001$")
    ax.set_title("Polynomial PC-CC values")
    ax.grid(True, alpha=0.5)
    _dedup_legend(ax, loc="lower right")
    return ax


# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------


def main() -> None:
    trajectories = make_trajectories()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Supports both repo root/sim and direct sandbox execution.
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    figs_dir = os.path.join(repo_root, "Figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Individual figures.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_state_trajectories(trajectories, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "platoon_state_polynomial_pccc.pdf"), bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_gap_trajectories(trajectories, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "platoon_gap_polynomial_pccc.pdf"), bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_certificate_values(trajectories, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "platoon_closure_polynomial_pccc.pdf"), bbox_inches="tight")

    # Combined figure.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_state_trajectories(trajectories, ax=axes[0])
    plot_gap_trajectories(trajectories, ax=axes[1])
    plot_certificate_values(trajectories, ax=axes[2])
    fig.suptitle("Two-car platoon with polynomial PC-CC (SOS/TSSOS template)", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "platoon_combined_polynomial_pccc.pdf"), bbox_inches="tight")
    print("Saved figures to", figs_dir)


if __name__ == "__main__":
    main()
