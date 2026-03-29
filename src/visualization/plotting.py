"""Visualization for formation trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

from ..bspline import BSpline, make_basis


def plot_formation_snapshot(ax, group, step, t):
    """Plot a single snapshot of the formation."""
    fp = group.frenet_path
    fp.plot_path(ax, t)

    # Plot obstacles
    for obs in group.vehicles[0].obstacles:
        corners = []
        for corner_splines in obs.corners_spline:
            x = float(corner_splines[0](t))
            y = float(corner_splines[1](t))
            ix, iy = fp.frenet_to_inertial(x, y, t)
            corners.append([ix, iy])
        corners.append(corners[0])
        poly = Polygon(corners, closed=True, fill=True,
                       fc=(0.7, 0.7, 0.7, 0.5), ec='gray', lw=1, zorder=1)
        ax.add_patch(poly)

    # Plot vehicles
    config = group.config
    basis = make_basis(config.state_degree, config.knot_intervals)
    colors = ['blue', 'red', 'green', 'orange']

    for i, v in enumerate(group.vehicles):
        if step < len(v.history['y']):
            coeffs = v.history['y'][step]
            p_spline = BSpline(basis, np.array(coeffs[:len(basis)]))
            q_spline = BSpline(basis, np.array(coeffs[len(basis):2 * len(basis)]))

            # Plot trajectory
            t_pts = np.linspace(0, 1, 50)
            ps = [float(p_spline(tp)) for tp in t_pts]
            qs = [float(q_spline(tp)) for tp in t_pts]
            xs, ys = [], []
            for p, q in zip(ps, qs):
                ix, iy = fp.frenet_to_inertial(p, q, t)
                xs.append(ix)
                ys.append(iy)
            color = colors[i % len(colors)]
            ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1)

            # Plot current position
            p0, q0 = float(p_spline(0)), float(q_spline(0))
            ix0, iy0 = fp.frenet_to_inertial(p0, q0, t)
            circle = plt.Circle((ix0, iy0), config.radius,
                                color=color, fill=True, alpha=0.7, zorder=5)
            ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_title(f'Step {step}')


def plot_frenet_view(group, save_path=None):
    """Plot vehicle trajectories in the Frenet frame."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    config = group.config
    basis = make_basis(config.state_degree, config.knot_intervals)
    colors = ['blue', 'red', 'green', 'orange']

    for i, v in enumerate(group.vehicles):
        ps, qs = [], []
        for step, y in enumerate(v.history['y']):
            p_spline = BSpline(basis, np.array(y[:len(basis)]))
            q_spline = BSpline(basis, np.array(y[len(basis):2 * len(basis)]))
            ps.append(float(p_spline(0)))
            qs.append(float(q_spline(0)))

        color = colors[i % len(colors)]
        ax1.plot(ps, label=f'Vehicle {i}', color=color)
        ax2.plot(qs, color=color)

    ax1.set_ylabel('p (along path)')
    ax1.set_title('Vehicle positions in Frenet frame')
    ax1.legend()
    ax2.set_ylabel('q (across path)')
    ax2.set_xlabel('Step')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
