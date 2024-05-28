import matplotlib.pyplot as plt
import numpy as np
from compas_fab.utilities import LazyLoader

pybullet = LazyLoader('pybullet', globals(), 'pybullet')


def plot_assembly_env(env, fig=None, ax=None):
    """
    Plot an assembly environment in 2d.

    Args:
        env: Either AssemblyEnv or AssemblyGym. If env is of type AssemblyGym, also the targets are plotted.
    """

    if hasattr(env, "assembly_env"):
        assembly_env = env.assembly_env
        gym_env = env
    else:
        assembly_env = env
        gym_env = None

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # plot ground
    ax.axhspan(-0.01, 0, color='grey')

    # plot obstacles
    for i, b in enumerate(assembly_env.obstacles):
        vertices = np.array([*b.vertices_2d])
        plt.fill(vertices[:, 0], vertices[:, 1], '+', edgecolor='k', facecolor='tab:red')
        ax.text(b.position[0], b.position[2], i, ha="center", va="center", color="w")

    # plot blocks
    for i, b in enumerate(assembly_env.blocks):
        frozen = i == assembly_env.frozen_block_index
        color = "tab:orange" if frozen else "tab:blue"
        vertices = np.array([*b.vertices_2d])
        plt.fill(vertices[:, 0], vertices[:, 1], '+', edgecolor='k', facecolor=color)
        ax.text(b.position[0], b.position[2], i, ha="center", va="center", color="w")

    # plot targets
    if gym_env is not None:
        for t in gym_env.targets:
            plt.scatter(t[0], t[2], marker="*", s=100, color="tab:green")

    ax.axis('equal')
    # ax.set_xlim(assembly_env.bounds[0][0], assembly_env.bounds[1][0])
    # ax.set_ylim(assembly_env.bounds[0][2], assembly_env.bounds[1][2])
    return fig, ax


def render_assembly_env(width=512, height=512, fov=120, near=0.02, far=10,
                  target=(0, 0, 0), distance=0.5, yaw=0, pitch=-40, roll=0, fig=None, ax=None):
    """
    Render an assembly environment in 3d through pybullet.
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    if ax is None:
        ax = fig.add_subplot(111)

    rgb_array = get_rgb_array(width=width, height=height, fov=fov, near=near, far=far, target=target, distance=distance,
                               yaw=yaw, pitch=pitch, roll=roll)

    ax.imshow(rgb_array)
    # hide the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def get_rgb_array(width=512, height=512, fov=120, near=0.02, far=10,
                  target=(0, 0, 0), distance=0.5, yaw=0, pitch=-40, roll=0):
    aspect = width / height
    view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=2
    )

    proj_matrix = pybullet.computeProjectionMatrixFOV(fov, aspect, near, far)

    images = pybullet.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
    )

    rgb_array = np.reshape(images[2], (height, width, 4)) * 1. / 255.
    return rgb_array[:, :, :3]


def render_blocks_2d(blocks, xlim, ylim, img_size=(512,512)):
    image = np.zeros(img_size, dtype=bool)
    # Note: Y axis is reversed to make image plot work without reversing the axis
    X, Y = np.meshgrid(np.linspace(*xlim, img_size[0]), np.linspace(ylim[1], ylim[0], img_size[1]))
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    for block in blocks:
        image = image | block.contains_2d(positions).reshape(img_size)

    return image


def plot_block_movements(initial_states, final_states, bounds, fig=None, ax=None):
    """
    Plot the movements of blocks in 2D.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Ensure the plot uses the same bounds as the assembly environment
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][2], bounds[1][2])

    # Plot initial and final positions and draw arrows with indices
    for index, ((initial_pos, _), (final_pos, _)) in enumerate(zip(initial_states, final_states)):
        # Plot initial position
        # ax.scatter(initial_pos[0], initial_pos[1], c='blue', label='Initial' if index == 0 else "", marker='o')

        # Plot final position
        # ax.scatter(final_pos[0], final_pos[1], c='red', label='Final' if index == 0 else "", marker='x')

        # Draw an arrow from initial to final position
        ax.arrow(initial_pos[0], initial_pos[2], final_pos[0] - initial_pos[0], final_pos[2] - initial_pos[2],
                    head_width=0.01, head_length=0.02, fc='green', ec='green')

        # Add an index label near the start of the arrow
        ax.text(initial_pos[0], initial_pos[1], f'{index}', color='purple', fontsize=9, ha='right', va='top')

    # Update legend to show only one entry per label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    return fig, ax