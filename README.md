# pc-cc

Simulation code and examples for **Path-Complete Closure Certificates (PC-CCs)** applied to nonlinear switched systems.

This repository currently includes:

- A two-car platoon switched system example with two modes (normal and communication breakdown).
- Simulation scripts to generate trajectories under random switching.
- Basic analysis of safety with respect to a velocity-gap constraint.

---

## Structure

```text
pc-cc/
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
└── sim/
    ├── __init__.py
    └── platoon_sim.py   # Two-car platoon switched system simulation
