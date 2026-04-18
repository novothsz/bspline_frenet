import matplotlib.pyplot as plt
import numpy as np


def plot_frenet_view(group):
    fig, ax = group.figures["figures"]

    for i in range(group.stage):
        ax.clear()
        for vehicle in group.vehicles:
            ax = vehicle.visualize_x_problem(ax, i)

        fig.savefig(group.cwd + "/figures/frenet_view_" + "{:0>2d}".format(i) + ".png", dpi=200)


def plot_moovie_frames(group, n_frames, iternum: int = 0, seed=""):
    fig, ax = group.figures["figures"]
    ax.clear()
    frame_num = 0
    horizon_num = 0

    for _ in np.linspace(0, 1, n_frames):
        for i in range(len(group.vehicles)):
            t_start, ax = group.vehicles[i].plot_moovie_frames_mooving_horizon(ax, horizon_num)
        horizon_num += 1

        ax.set_xlim(group.border_x[0] * 1.2, group.border_x[1] * 1.2)
        ax.set_ylim(group.border_y[0] * 1.2, group.border_y[1] * 1.2)
        ax.set_xlim(group.vehicles[0].fp.fx_spline(t_start)[0][0] - 2, group.vehicles[0].fp.fx_spline(t_start)[0][0] + 2 * 3)
        ax.set_ylim(group.vehicles[0].fp.fy_spline(t_start)[0][0] - 2.5, group.vehicles[0].fp.fy_spline(t_start)[0][0] + 2.5)
        ax.set_aspect("equal", adjustable="box")
        plt.axis("off")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        fig.savefig(group.cwd + "/video/" + "{:0>2d}".format(frame_num) + ".png", dpi=200)
        ax.clear()
        frame_num += 1

    return group
