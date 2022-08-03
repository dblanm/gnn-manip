# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#
import matplotlib
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

matplotlib.use('Agg')

plt.style.use('seaborn-white')
rcParams['figure.figsize'] = 11.7, 8.27

rcParams['figure.dpi'] = 600
rcParams['figure.subplot.left'] = 0.12
rcParams['figure.subplot.right'] = 0.95
rcParams['figure.subplot.bottom'] = 0.1
rcParams['figure.subplot.top'] = 0.95
rcParams['axes.grid'] = False
rcParams['grid.linestyle'] = ":"
rcParams['xtick.major.bottom'] = True
rcParams['xtick.bottom'] = True
rcParams['ytick.left'] = True
rcParams['ytick.minor.visible'] = True
rcParams['ytick.minor.size'] = 0.5
rcParams['ytick.minor.width'] = 0.4
rcParams['ytick.major.left'] = True
# Font
rcParams['text.usetex'] = True
rcParams['ps.fonttype'] = 42
# Lines
rcParams['lines.linewidth'] = 3
rcParams['lines.markersize'] = 6

font_value = 24

rcParams['font.size'] = font_value
rcParams['axes.titleweight'] = "bold"
rcParams['legend.title_fontsize'] = 100


def plot_2d(ax, nodes, senders, receivers, color, label,
            x_axis=0, y_axis=1, z_axis=2, marker_size=0.5, line_width=0.3):

    if color == "darkred":
        alpha_color = "lightcoral"
    elif color == "darkgreen":
        alpha_color = "limegreen"
    elif color == "darkorange":
        alpha_color = "gold"
    elif color == "blue":
        alpha_color = "aqua"
    else:
        alpha_color = color

    x_senders = nodes[senders]
    x_receivers = nodes[receivers]

    for i in range(0, senders.shape[0]):  # Node i
        ax.plot3D([x_senders[i, x_axis], x_receivers[i, x_axis]], [x_senders[i, y_axis], x_receivers[i, y_axis]],
                  [x_senders[i, z_axis], x_receivers[i, z_axis]], marker='None', color=alpha_color, ls='-',
                  linewidth=line_width, zorder=1)

    ax.scatter(nodes[:, x_axis], nodes[:, y_axis], nodes[:, z_axis],
               s=marker_size, marker='o', color=color, zorder=5)


def plot_single_graph(nodes, edge_index, title):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nodes = nodes.numpy()
    senders = edge_index[0, :].numpy()
    receivers = edge_index[1, :].numpy()
    color = "blue"

    plot_2d(ax, nodes, senders, receivers, color=color, label="Graph")

    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.1, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.set_zlim(-0.001, 1.0)
    ax.view_init(elev=42., azim=-106)
    plt.legend()
    # plt.show()

    plt.savefig(title)

    plt.close(fig)


# Following added to plot 3D coffee datasets with granular and rigid-body particles
def plot_coffee_2d(ax, nodes, coffee_nodes, rigid_nodes, senders, receivers, color, label,
                   x_axis=0, y_axis=1, z_axis=2, marker_size=0.5, line_width=0.3):

    if color == "darkred":
        alpha_color = "lightcoral"
    elif color == "darkgreen":
        alpha_color = "yellowgreen"
    elif color == "darkorange":
        alpha_color = "gold"
    elif color == "blue":
        alpha_color = "aqua"
    elif color == "black":
        alpha_color = "dimgray"
    elif color == "dimgray":
        alpha_color =  "silver"
    elif color == "slategrey":
        alpha_color = "lightsteelblue"
    else:
        alpha_color = color

    color_coffee = "darkorange"

    x_senders = nodes[senders]
    x_receivers = nodes[receivers]

    for i in range(0, senders.shape[0]):  # Node i
        ax.plot3D([x_senders[i, x_axis], x_receivers[i, x_axis]], [x_senders[i, y_axis], x_receivers[i, y_axis]],
                  [x_senders[i, z_axis], x_receivers[i, z_axis]], marker='None', color=alpha_color, ls='-',
                  linewidth=line_width, zorder=1)

    ax.scatter(coffee_nodes[:, x_axis], coffee_nodes[:, y_axis], coffee_nodes[:, z_axis],
               s=marker_size, marker='o', color=color_coffee, zorder=5)

    ax.scatter(rigid_nodes[:, x_axis], rigid_nodes[:, y_axis], rigid_nodes[:, z_axis],
               s=marker_size, marker='o', color=color, zorder=5)


def plot_coffeee_graph(nodes, coffee_nodes, rigid_nodes, edge_index, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nodes = nodes.numpy()
    senders = edge_index[0, :].numpy()
    receivers = edge_index[1, :].numpy()
    color = "slategrey"
    plot_coffee_2d(ax, nodes, coffee_nodes, rigid_nodes, senders, receivers, color=color, label="Graph",
                   x_axis=2, y_axis=0, z_axis=1, marker_size=1.0, line_width=0.5)

    ax.grid(False)

    ax.set_xlim(0.65, 0.4)
    ax.set_ylim(0.4, 0.65)
    ax.set_zlim(0.1, 0.35)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_axis_off()
    plt.savefig(title)

    plt.close(fig)


def plot_trajectory(traj_rot0, traj_ty0, traj_rot1, traj_ty1, title='',
                    save_name='test_trajectory'):

    fig, ax = plt.subplots(2, 1)
    x = np.arange(traj_rot0.shape[0])
    ax[0].set_title(title)
    ax[0].plot(x, traj_rot0[:], label='rot groundtruth', color='r')
    ax[0].plot(x, traj_rot1, label='rot test', color='b')
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel('Degrees')
    ax[1].plot(x, traj_ty0[:], label='ty groundtruth', color='r')
    ax[1].plot(x, traj_ty1, label='ty test', color='b')
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel('Translation (m)')
    ax[1].set_xlabel('Timesteps')

    plt.savefig(save_name + '.png')

    plt.close(fig)


def plot_multiple_nodes(coffee_particles, rigid_particles, title, desired=None):
    # Plot YZ
    fig, ax4 = plt.subplots()
    ax4.scatter(coffee_particles[:, 2], coffee_particles[:, 1], color='darkorange', zorder=4, label='Sand Particles')
    ax4.scatter(rigid_particles[:, 2], rigid_particles[:, 1], color='grey', zorder=3, label='Rigid Particles')
    if desired is not None:
        ax4.scatter(desired[:, 2], desired[:, 1], color='darkkhaki', zorder=2, label='Desired')
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 0.8)
    ax4.legend(frameon=True)
    ax4.axhline(y=0.1,xmin=0.4, xmax=0.6)
    ax4.axvline(x=0.4, ymin=0.1, ymax=0.3)
    ax4.axvline(x=0.6, ymin=0.1, ymax=0.3)
    plt.savefig(title+'particles_YZ.png', dpi=100)
    plt.close(fig)