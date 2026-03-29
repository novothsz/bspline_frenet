"""Visualization for formation trajectories.

Renders per-frame PNGs and assembles them into a GIF, similar to the
original plot_moovie_frames implementation but using the new data structures.
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from ..bspline import BSpline, make_basis


def render_movie(group, output_path='formation.gif', fps=15, dpi=150, follow_camera=True):
    """Render the full simulation as an animated GIF.

    Args:
        group: Group instance with completed simulation history.
        output_path: Output file path (.gif or directory for .png frames).
        fps: Frames per second for GIF.
        dpi: Resolution of each frame.
        follow_camera: If True, camera follows the Frenet frame along the path.
    """
    config = group.config
    fp = group.frenet_path
    basis = make_basis(config.state_degree, config.knot_intervals)
    n_steps = len(group.vehicles[0].history['y'])

    # Determine if we output a GIF or individual frames
    output_dir = None
    if output_path.endswith('.gif'):
        frames = []
    else:
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for step in range(n_steps):
        ax.clear()
        t_start = group.vehicles[0].history['t_start'][step]
        t_end = group.vehicles[0].history['t_end'][step]

        # Draw the reference path
        _draw_path(ax, fp)

        # Draw obstacles (in inertial frame at current time)
        _draw_obstacles(ax, group.vehicles[0].obstacles, fp, t_start)

        # Draw each vehicle's trajectory and drone
        for v in group.vehicles:
            _draw_vehicle_trajectory(ax, v, step, basis, fp, config)
            _draw_drone(ax, v, step, basis, fp, config)

        # Camera
        if follow_camera:
            cx = float(fp.fx_spline(t_start))
            cy = float(fp.fy_spline(t_start))
            ax.set_xlim(cx - 2, cx + 6)
            ax.set_ylim(cy - 2.5, cy + 2.5)
        else:
            ax.set_xlim(fp.tau_0 * 1.1, fp.tau_f * 1.1)
            ax.set_ylim(-2.5, 2.5)

        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        fig.tight_layout(pad=0.5)

        if output_dir:
            fig.savefig(os.path.join(output_dir, f'{step:04d}.png'), dpi=dpi)
        else:
            # Render to in-memory image for GIF
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frames.append(buf.copy())

    plt.close(fig)

    if not output_dir:
        _save_gif(frames, output_path, fps)

    print(f'Saved {n_steps} frames to {output_path}')


def plot_frenet_view(group, save_path=None):
    """Plot vehicle trajectories in the Frenet frame (p and q over time)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    config = group.config
    basis = make_basis(config.state_degree, config.knot_intervals)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for i, v in enumerate(group.vehicles):
        ps, qs = [], []
        for y in v.history['y']:
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
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_path(ax, fp):
    """Draw the sinusoidal reference path."""
    t = np.linspace(0, 1, 200)
    fx = [fp.fx(fp.t_to_tau(t_)) for t_ in t]
    fy = [fp.fy(fp.t_to_tau(t_)) for t_ in t]
    ax.plot(fx, fy, linestyle=':', color='gray', lw=1, zorder=1, label='Path')


def _draw_obstacles(ax, obstacles, fp, t):
    """Draw obstacles in inertial frame at time t."""
    for obs in obstacles:
        # Actual obstacle
        corners_inertial = []
        for corner_splines in obs.corners_spline:
            p_val = float(corner_splines[0](t))
            q_val = float(corner_splines[1](t))
            ix, iy = fp.frenet_to_inertial(p_val, q_val, t)
            corners_inertial.append([ix, iy])
        poly = Polygon(corners_inertial, closed=True, fill=True,
                       fc=(0, 0, 0, 0.1), ec=(0, 0, 0, 1), lw=1, zorder=1)
        ax.add_patch(poly)


def _draw_vehicle_trajectory(ax, vehicle, step, basis, fp, config):
    """Draw a vehicle's predicted trajectory for one time step."""
    y = vehicle.history['y'][step]
    t_start = vehicle.history['t_start'][step]
    t_end = vehicle.history['t_end'][step]

    nc = len(basis)
    p_spline = BSpline(basis, np.array(y[:nc]))
    q_spline = BSpline(basis, np.array(y[nc:2 * nc]))

    n_pts = 50
    # Index separating "committed" part (next t_step) from prediction
    if t_end > t_start:
        frac = config.t_step / (t_end - t_start)
        split_idx = max(1, int(n_pts * frac))
    else:
        split_idx = n_pts

    xs, ys = [], []
    for t_pq, t_frenet in zip(np.linspace(0, 1, n_pts),
                               np.linspace(t_start, t_end, n_pts)):
        p_val = float(p_spline(t_pq))
        q_val = float(q_spline(t_pq))
        ix, iy = fp.frenet_to_inertial(p_val, q_val, t_frenet)
        xs.append(ix)
        ys.append(iy)

    # Committed portion in black, rest in light blue
    ax.plot(xs[:split_idx], ys[:split_idx], c='k', lw=0.8, alpha=1, zorder=5)
    ax.plot(xs[split_idx:], ys[split_idx:], c='cornflowerblue', lw=0.8, alpha=0.5, zorder=3)


def _draw_drone(ax, vehicle, step, basis, fp, config):
    """Draw a quadrotor drone icon at the vehicle's current position."""
    y = vehicle.history['y'][step]
    t_start = vehicle.history['t_start'][step]

    nc = len(basis)
    p_spline = BSpline(basis, np.array(y[:nc]))
    q_spline = BSpline(basis, np.array(y[nc:2 * nc]))

    p0 = float(p_spline(0))
    q0 = float(q_spline(0))
    x0, y0 = fp.frenet_to_inertial(p0, q0, t_start)

    # Heading: from path tangent + vehicle velocity
    dp = float(p_spline.derivative()(0))
    dq = float(q_spline.derivative()(0))
    fx_d = float(fp.fx_d_spline(t_start))
    fy_d = float(fp.fy_d_spline(t_start))
    theta = math.atan2(fy_d + dq, fx_d + dp)

    r = config.radius * 0.5
    arm = r * 2
    rotor_r = arm * 0.3

    # Four arms at 45-degree offsets
    for k in range(4):
        angle = theta + math.pi / 4 * (1 + 2 * k)
        ex = x0 + arm * math.cos(angle)
        ey = y0 + arm * math.sin(angle)
        ax.add_line(Line2D([x0, ex], [y0, ey], color='k', lw=0.8, zorder=8))
        ax.add_patch(plt.Circle((ex, ey), rotor_r, color='k', alpha=0.5, zorder=10))

    # Body rectangle
    rect_half = r / 2
    rx = x0 + (-rect_half) * math.cos(theta) - (-rect_half) * math.sin(theta)
    ry = y0 + (-rect_half) * math.sin(theta) + (-rect_half) * math.cos(theta)
    rect = patches.Rectangle(
        (rx, ry), r, r,
        angle=math.degrees(theta),
        linewidth=1, edgecolor='k', facecolor='k', zorder=9)
    ax.add_patch(rect)


def _save_gif(frames, path, fps):
    """Save a list of RGBA numpy arrays as an animated GIF using matplotlib."""
    # Use Pillow if available, otherwise fall back to imageio
    try:
        from PIL import Image
        images = [Image.fromarray(f) for f in frames]
        duration = int(1000 / fps)
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=duration, loop=0, optimize=True)
    except ImportError:
        try:
            import imageio
            imageio.mimsave(path, frames, fps=fps)
        except ImportError:
            # Last resort: save individual PNGs
            out_dir = path.replace('.gif', '_frames')
            os.makedirs(out_dir, exist_ok=True)
            for i, f in enumerate(frames):
                plt.imsave(os.path.join(out_dir, f'{i:04d}.png'), f)
            print(f'Neither Pillow nor imageio found. Saved {len(frames)} PNGs to {out_dir}/')
