import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Dynamics: two-car platoon (discrete-time, polynomial)
# ---------------------------------------------------------

def f1(x):
    """
    Mode 1: normal communication.
    x[0] = x1 (follower velocity)
    x[1] = x2 (leader  velocity)
    """
    x1, x2 = x
    x1_next = 0.01 * x2 + 0.9 * x1 - 0.02 * x1**2
    x2_next = 2.0 + 0.8 * x2 - 0.04 * x2**2
    return np.array([x1_next, x2_next])


def f2_safe(x):
    """
    Mode 2 (safe case): communication breakdown, u1 = 0, u2 = 2.
    """
    x1, x2 = x
    x1_next = 0.9 * x1 - 0.02 * x1**2
    x2_next = 2.0 + 0.8 * x2 - 0.04 * x2**2
    return np.array([x1_next, x2_next])


def f2_unsafe(x):
    """
    Mode 2 (unsafe case): communication breakdown, u1 = 0, u2 = 0.
    """
    x1, x2 = x
    x1_next = 0.9 * x1 - 0.02 * x1**2
    x2_next = 0.8 * x2 - 0.04 * x2**2
    return np.array([x1_next, x2_next])


# ---------------------------------------------------------
# Simulation utilities
# ---------------------------------------------------------

def sample_initial(rng):
    """
    Sample x0 in X0:
      x1 in [0, 3]
      gap = x2 - x1 in [1, 2]
      => x2 in [x1 + 1, x1 + 2]
    """
    x1 = rng.uniform(0.0, 3.0)
    gap = rng.uniform(1.0, 2.0)
    x2 = x1 + gap
    return np.array([x1, x2])


def simulate_system(system_type="safe", n_traj=100, T=20, p_mode2=0.2, seed=0):
    """
    Simulate many trajectories under random switching.

    Parameters
    ----------
    system_type : {"safe", "unsafe"}
        Whether to use the safe or unsafe definition of mode 2.
    n_traj : int
        Number of trajectories.
    T : int
        Time horizon (number of steps).
    p_mode2 : float
        Probability of being in mode 2 at each time step.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    gaps_all = []      # store x2 - x1 for all trajectories
    switches_all = []  # store chosen modes (1 or 2)

    for _ in range(n_traj):
        x = sample_initial(rng)
        gaps = []
        sigmas = []

        for t in range(T + 1):
            gap = x[1] - x[0]
            gaps.append(gap)

            if t == T:
                break

            # Random switching
            if rng.random() < p_mode2:
                sigma = 2
            else:
                sigma = 1

            sigmas.append(sigma)

            if sigma == 1:
                x = f1(x)
            else:
                if system_type == "safe":
                    x = f2_safe(x)
                else:
                    x = f2_unsafe(x)

        gaps_all.append(np.array(gaps))
        switches_all.append(np.array(sigmas))

    return np.array(gaps_all), switches_all


def plot_gaps(gaps, unsafe_threshold=0.2, title="", save_path=None):
    """
    Plot x2 - x1 vs time for many trajectories.
    """
    T = gaps.shape[1] - 1
    t = np.arange(T + 1)

    plt.figure(figsize=(6, 4))
    for k in range(gaps.shape[0]):
        plt.plot(t, gaps[k, :], linewidth=0.8, alpha=0.7)

    plt.axhline(unsafe_threshold, linestyle="--", linewidth=1.5,
                label=f"unsafe threshold = {unsafe_threshold}")
    plt.xlabel("time step t")
    plt.ylabel(r"$x_2(t) - x_1(t)$")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    # Show for interactive use (you can comment out in non-interactive runs)
    plt.show()


def summarize_gaps(gaps, unsafe_threshold=0.2):
    """
    Return summary statistics: min/mean gap and fraction unsafe.
    """
    min_gap = float(gaps.min())
    mean_gap = float(gaps.mean())
    frac_unsafe = float(np.mean(gaps < unsafe_threshold))
    return min_gap, mean_gap, frac_unsafe


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------

def main():
    # Parameters
    N_TRAJ = 100
    T = 20
    P_MODE2 = 0.2
    THRESH = 0.2

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # -------- Safe system --------
    gaps_safe, switches_safe = simulate_system(
        system_type="safe",
        n_traj=N_TRAJ,
        T=T,
        p_mode2=P_MODE2,
        seed=0,
    )
    min_safe, mean_safe, frac_unsafe_safe = summarize_gaps(gaps_safe, THRESH)
    print("SAFE SYSTEM:")
    print(f"  min gap              = {min_safe:.3f}")
    print(f"  mean gap             = {mean_safe:.3f}")
    print(f"  frac(gap < {THRESH}) = {frac_unsafe_safe:.4f}")

    plot_gaps(
        gaps_safe,
        unsafe_threshold=THRESH,
        title="Safe system: random switching",
        save_path=out_dir / "platoon_safe.png",
    )

    # -------- Unsafe system --------
    gaps_unsafe, switches_unsafe = simulate_system(
        system_type="unsafe",
        n_traj=N_TRAJ,
        T=T,
        p_mode2=P_MODE2,
        seed=1,
    )
    min_uns, mean_uns, frac_unsafe_uns = summarize_gaps(gaps_unsafe, THRESH)
    print("\nUNSAFE SYSTEM:")
    print(f"  min gap              = {min_uns:.3f}")
    print(f"  mean gap             = {mean_uns:.3f}")
    print(f"  frac(gap < {THRESH}) = {frac_unsafe_uns:.4f}")

    plot_gaps(
        gaps_unsafe,
        unsafe_threshold=THRESH,
        title="Unsafe system: random switching",
        save_path=out_dir / "platoon_unsafe.png",
    )


if __name__ == "__main__":
    main()
