import matplotlib.pyplot as plt
import numpy as np
from compas_fab.utilities import LazyLoader


from assembly_gym.utils.geometry import block_vertices_2d

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
        ax.fill(vertices[:, 0], vertices[:, 1], '+', edgecolor='k', facecolor='tab:red')
        ax.text(b.position[0], b.position[2], i, ha="center", va="center", color="w")

    # plot blocks
    for i, b in enumerate(assembly_env.blocks):
        frozen = i == assembly_env.frozen_block_index
        color = "tab:orange" if frozen else "tab:blue"
        vertices = np.array([*b.vertices_2d])
        ax.fill(vertices[:, 0], vertices[:, 1], '+', edgecolor='k', facecolor=color)
        ax.text(b.position[0], b.position[2], i, ha="center", va="center", color="w")

    # plot targets
    if gym_env is not None:
        for t in gym_env.targets:
            ax.scatter(t[0], t[2], marker="*", s=100, color="tab:green")

    # ax.axis('equal')
    #ax.set_xlim(assembly_env.bounds[0][0], assembly_env.bounds[1][0])
    #ax.set_ylim(assembly_env.bounds[0][2], assembly_env.bounds[1][2])
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


def render_blocks_2d(blocks, xlim, ylim, img_size=(512, 512)):
    image = np.zeros(img_size, dtype=bool)
    # Note: Y axis is reversed to make image plot work without reversing the axis
    X, Y = np.meshgrid(np.linspace(*xlim, img_size[0]), np.linspace(ylim[1], ylim[0], img_size[1]))
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    
    print("blocks:", blocks)
    print("positions shape:", positions.shape)

    for block in blocks:
        block_mask = block.contains_2d(positions).reshape(img_size)
        image = image | block_mask
        print(f"block_mask for block {block}:")
        print(block_mask)
    
    print("hello")
    plt.imshow(image, cmap='gray', origin='upper', extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rendered Blocks 2D')
    plt.show()
    return image


def plot_block_movements(initial_states, final_states, bounds=None, fig=None, ax=None):
    """
    Plot the movements of blocks in 2D.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Ensure the plot uses the same bounds as the assembly environment
    if bounds is not None:
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


def plot_cra_assembly(assembly, fig=None, ax=None, plot_forces=True, force_scale=1.0, plot_edges=False, graph=None):
    """
    Plot the CRA assembly in 2D with forces.
    """
    if graph is None:
        graph = assembly.graph
    if fig is None:
        fig, ax = plt.subplots(1)

    # plot blocks
    for i, node in graph.node.items():
        block = node['block']
        
        facecolor = 'tab:blue'
        if i == -1:
            facecolor = 'gray'
        elif node.get('is_support'):
            facecolor = 'tab:orange'

        vertices = np.array([*block_vertices_2d(block)])
        ax.fill(vertices[:, 0], vertices[:, 1], '+', edgecolor='k', facecolor=facecolor)

    # plot nodes
    for node in graph.nodes():
        point = assembly.node_point(node)
        ax.plot(point[0], point[2], 'o', color='tab:red')

    # plot edges
    if plot_edges:
        for edge in graph.edges():
            u, v = edge
            point_u = assembly.node_point(u)
            point_v = assembly.node_point(v)
            ax.plot([point_u[0], point_v[0]], [point_u[2], point_v[2]], 'k--', linewidth=1)

    # plot interfaces
    for interface in assembly.interfaces():
        points = [p for p in interface.points if p[1] > 0]
        points = np.array(points)
        if len(points) == 2:
            ax.plot(points[:, 0], points[:, 2], 'k-' ,linewidth=4)
        else:
            ax.scatter(points[:, 0], points[:, 2], c='k', s=10)
            print("Warning: interface with more than 2 points")

    # plot forces
    if plot_forces:
        for edge in graph.edges():
            interfaces = graph.edge_attribute(edge, "interfaces")
            for interface in interfaces:
                frame = interface.frame

                n = len(interface.points)
                for i in range(n):
                    # plot point
                    point = interface.points[i]
                    if point[1] < 0:
                        continue

                    ax.plot(point[0], point[2], 'o', color='tab:green')

                    force = interface.forces[i]

                    force_vector = [force['c_u'], force['c_v'], force['c_np'] - force['c_nn']]
                    # to world coordinates
                    force_vector = frame.to_world_coordinates(force_vector) - frame.point
                    ax.arrow(point[0], point[2], -force_scale * force_vector[0], -force_scale * force_vector[2], color='tab:green')

    ax.axis('equal')
    return fig, ax