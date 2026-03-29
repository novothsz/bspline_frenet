# B-Spline Frenet Formation Control

Distributed motion planning for multi-vehicle formation control using B-splines in a moving Serret-Frenet frame. Vehicles navigate in formation through obstacles, coordinating via ADMM (Alternating Direction Method of Multipliers).

Based on the ACC paper: *"Distributed Motion Planning for Navigation in Closed Formation Among Obstacles"*.

## Setup

Requires Python 3.10+.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

### Dependencies

| Package    | Purpose                                      |
|------------|----------------------------------------------|
| casadi     | Symbolic NLP construction + IPOPT solver      |
| numpy      | Numerical arrays                              |
| scipy      | Sparse matrices, linear algebra               |
| matplotlib | Plotting trajectories and formation snapshots  |

No commercial solvers required. The old Gurobi and autograd dependencies have been removed.

## Running

```bash
uv run python run_example.py
```

This runs 4 vehicles through 100 MPC steps with randomly generated obstacles. Typical output:

```
Building NLP solvers...
Solvers built.
Step 0/100: 2.240s
Step 1/100: 0.942s
...
Completed 100 steps
Average iteration time: 0.642s
Vehicle 0: 88/100 succeeded
```

On first run, the Frenet path splines are fitted and cached to `coeffs.pickle`. Subsequent runs load from cache.

## Configuration

All hyperparameters live in `src/config.py` as a single `Config` dataclass. Override defaults by passing a modified config to `run()`:

```python
from src.config import Config
from run_example import run

config = Config(
    n_vehicles=6,
    t_step=0.02,          # larger step = fewer MPC iterations, faster
    knot_intervals=3,     # fewer knots = smaller NLP, faster but less accurate
    rho=100.0,            # ADMM penalty parameter
    n_admm_iterations=2,  # more ADMM iterations = better consensus, slower
)
group, times = run(config)
```

### Key parameters

| Parameter            | Default | Description                                           |
|----------------------|---------|-------------------------------------------------------|
| `n_vehicles`         | 4       | Number of vehicles in the formation                   |
| `t_step`             | 0.01    | MPC time step (fraction of total horizon)             |
| `t_window_size`      | 0.12    | Prediction horizon length                             |
| `knot_intervals`     | 5       | B-spline knot intervals (controls trajectory freedom) |
| `rho`                | 50.0    | ADMM consensus penalty weight                         |
| `rho_input`          | 200.0   | Acceleration cost weight                              |
| `radius`             | 0.08    | Vehicle collision radius                              |
| `n_waypoints`        | 5       | Intermediate waypoints from the warm-starter          |
| `n_admm_iterations`  | 1       | ADMM iterations per MPC step                          |
| `ipopt_max_iter`     | 10000   | Max IPOPT solver iterations                           |

## Architecture

```
src/
├── config.py                      # Config dataclass with all hyperparameters
├── bspline/                       # B-spline math (from OMG-tools, cleaned)
│   ├── basis.py                   #   BSplineBasis: Cox-de Boor evaluation, knot ops
│   ├── spline.py                  #   BSpline: arithmetic, derivative, integral
│   └── operations.py              #   shift, crop, extrapolate, definite_integral, make_basis()
├── frenet/                        # Reference path and coordinate transforms
│   ├── path.py                    #   FrenetPath: path definition, inertial<->Frenet transforms
│   └── spline_fitter.py           #   SplineFitter: fit B-splines to data via IPOPT
├── optimization/                  # CasADi/IPOPT NLP construction
│   ├── nlp_builder.py             #   NLPBuilder: accumulate variables, constraints, cost
│   ├── admm.py                    #   ADMMState: primal/dual/consensus variable storage
│   ├── x_problem.py               #   XProblem: trajectory optimization NLP (per-vehicle)
│   └── z_problem.py               #   ZProblem: formation consensus NLP (per-vehicle)
├── formation/                     # Multi-vehicle coordination
│   ├── vehicle.py                 #   Vehicle: state, spline shifting, solver calls
│   ├── group.py                   #   Group: ADMM loop, data exchange, obstacle generation
│   ├── obstacle.py                #   Obstacle: polygon corners as Frenet-frame splines
│   └── warm_start.py              #   FormationWarmStarter: analytical waypoint generation
└── visualization/
    └── plotting.py                # Trajectory and formation plots
```

## How the algorithm works

### 1. Reference path (Frenet frame)

A sinusoidal reference path is defined in the inertial frame:

```
x(tau) = tau
y(tau) = sin(tau / (pi/2))
```

The moving Serret-Frenet frame travels along this path, parameterized by `t in [0, 1]`. All vehicle positions are expressed in Frenet coordinates `(p, q)` relative to this frame, where `p` is along the path and `q` is perpendicular.

### 2. B-spline parameterization

Vehicle trajectories are represented as B-splines with `knot_intervals` knot spans on `[0, 1]`. This is a cubic B-spline by default (`state_degree=3`), giving `knot_intervals + state_degree = 8` coefficients per dimension. Each vehicle has 3 dimensions: `p`, `q`, and `phi` (formation rotation angle).

The B-spline parameterization converts the continuous trajectory optimization into a finite-dimensional NLP over the spline coefficients.

### 3. Obstacle avoidance (separating hyperplane)

Obstacles are convex polygons with corners that move in the Frenet frame over time (since the frame itself is moving). Collision avoidance uses the separating hyperplane theorem:

- A hyperplane `a^T x = b` is placed between each vehicle and each obstacle
- **Constraint 1**: the vehicle must be on one side: `a^T p_vehicle - b <= -radius`
- **Constraint 2**: all obstacle corners must be on the other side: `a^T p_corner - b - d >= 0`
- **Constraint 3**: the normal is unit-bounded: `||a||^2 <= 1`

The hyperplane parameters `(a, b, d)` are themselves B-splines, optimized jointly with the trajectory.

### 4. Distributed optimization (ADMM)

Vehicles don't solve one giant centralized problem. Instead, each vehicle solves its own trajectory independently, then they coordinate through ADMM consensus:

```
For each MPC step:
  1. X-update:  Each vehicle optimizes its trajectory (parallel)
  2. Exchange:  Vehicles broadcast their trajectories to neighbours
  3. Z-update:  Each vehicle optimizes consensus variables (formation shape)
  4. Lambda:    Dual variable update
  5. Exchange:  Vehicles broadcast consensus variables
```

The **x-update** (trajectory optimization) minimizes acceleration while satisfying:
- Initial position/velocity constraints
- Waypoint constraints from the warm-starter
- Obstacle avoidance via separating hyperplanes
- ADMM consensus cost pulling toward the agreed formation

The **z-update** (formation consensus) enforces:
- Formation shape: cross-product constraint ensures relative positions match the reference formation (up to rotation and scaling)
- Mean position at origin: the formation centroid stays on the Frenet path
- Rotation agreement: all vehicles share the same `phi` angle

### 5. Warm-starting (replaces DFG)

Before each MPC step, the `FormationWarmStarter` generates intermediate waypoints that guide the formation around obstacles. It:

1. Scans the time window for collisions between the formation and obstacle danger zones
2. For each collision interval, computes **critical rotation angles** analytically (the exact angles that place a vehicle on an obstacle edge)
3. Tests these angles combined with scaling factors to find a collision-free formation configuration
4. Assigns the resulting waypoints as soft constraints in the x-update

This replaces the original DFG (Dynamic Formation Generator) which used a brute-force grid search over 432 rotation/scaling candidates.

### 6. MPC loop

The full simulation advances in steps of `t_step`:

```
For each step:
  1. Warm-starter generates waypoints for the current time window
  2. ADMM solve (x-update -> exchange -> z-update -> lambda -> exchange)
  3. Extract new initial state from the solution at t = t_step
  4. Shift all spline coefficients forward (warm-start next step)
  5. Advance the time window
```

Spline shifting uses extrapolation + cropping + basis transformation to reuse the previous solution as a warm start for the next step.

## Extending

### Changing the reference path

Edit `src/frenet/path.py`. Modify `fx(tau)`, `fy(tau)` and their analytical derivatives `fx_d`, `fy_d`, `fx_dd`, `fy_dd`. Delete `coeffs.pickle` to force re-fitting.

### Adding more obstacles

Modify `Group.generate_obstacles()` in `src/formation/group.py`. Each obstacle is a list of `[x, y]` corners in the inertial frame. The `Obstacle` class automatically fits spline trajectories in the Frenet frame.

### Custom formation shapes

The formation shape is defined by the initial/final vehicle positions. Change the ellipse parameters in `Group._ellipse_positions()` or pass custom positions directly via `Vehicle.set_position()`.
