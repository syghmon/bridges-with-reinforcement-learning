import gym

from compas.geometry import distance_point_point
from gym import spaces
import numpy as np

from assembly_gym.envs.assembly_env import AssemblyEnv, Block, Shape
from assembly_gym.utils import align_frames_2d, distance_box_point
from assembly_gym.utils.rendering import plot_assembly_env, get_rgb_array


def sparse_reward(gym_env, info=None):
    num_targets_reached = info['num_targets_reached']
    if gym_env.assembly_env.state_info['collision'] or not gym_env.assembly_env.state_info['stable']:
        return -1
    elif num_targets_reached > 0:
        return num_targets_reached
    return 0


def bridge_setup(H=0.45, num_stories=1):
    trapezoid = Shape(urdf_file='../assembly_gym/shapes/trapezoid.urdf', name="trapezoid", n_effective_faces=4)
    shapes = [trapezoid]
    targets = [ (0.525 , 0, num_stories * H + H/2), ]
    obstacles = [(targets[0][0], 0., i*H + H/2) for i in range(num_stories)]

    return dict(shapes=shapes, obstacles=obstacles, targets=targets)


def tower_setup(num_targets = 3, x_min = 0.2, x_max = 0.8, z_min = 0, z_max = 0.2):
    cube = Shape(urdf_file='../assembly_gym/shapes/cube.urdf', name="cube")
    shapes = [cube]
    # We can only place the cubes on top of each other:
    targets = [(np.random.uniform(x_min, x_max), 0, np.random.uniform(z_min, z_max)) for _ in range(num_targets)]
    #obstacles = [[targets[0][0], 0, 0.02]]
    obstacles = []
    return dict(shapes=shapes, obstacles=obstacles, targets=targets)


def hard_tower_setup():
    trapezoid = Shape(urdf_file='../assembly_gym/shapes/trapezoid.urdf', name="trapezoid", n_effective_faces=4)
    cube = Shape(urdf_file='../assembly_gym/shapes/cube.urdf', name="cube", n_effective_faces=1)
    shapes = [trapezoid, cube]
    targets = [[0.2, 0, 0], [0.2, 0, 0.3]]
    obstacles = [[0.2, 0, 0.1]]
    return dict(shapes=shapes, obstacles=obstacles, targets=targets)

def image_features(observation, xlim=(0,1), ylim=(0, 1), width=512, height=512):
    return np.stack([
        render_blocks(observation['blocks'], xlim=xlim, ylim=ylim, width=width, height=height),
        render_blocks(observation['obstacle_blocks'], xlim=xlim, ylim=ylim, width=width, height=height),
    ])

def binary_features(observation):
    return np.array([
        observation['stable'],
        observation['collision'], # ToDo obstacle vs block collision
        observation['collision_block'],
        observation['collision_obstacle'],
        observation['collision_floor'],
        observation['collision_boundary'],
    ])

def task_features(observation, xlim=(0,1), ylim=(0, 1), width=512, height=512):
    return render_blocks(observation['target_blocks'], xlim=xlim, ylim=ylim, width=width, height=height)

def contains(block, points):
    contains = np.ones(len(points), dtype=bool)

    for i in range(block.num_faces_2d()):
        frame = block.get_face_frame_2d(i)

        offset = np.array([frame.point[0], frame.point[2]])
        normal = np.array([frame.normal[0], frame.normal[2]])

        contains = contains & (np.dot(points - offset, normal) <= 0)

    return contains

def render_blocks(blocks, xlim, ylim, width=512, height=512):
    image = np.zeros((width, height), dtype=bool)
    X, Y = np.meshgrid(np.linspace(*xlim, image.shape[0]), np.linspace(*ylim, image.shape[1]))
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    for block in blocks:
        image = image | contains(block, positions).reshape(image.shape)

    return image

def iterate_actions(gym, ground_pos_values, offset_values=None, max_angle_rad=np.pi/4, max_blocks_per_face=1):
    """
    This is a generator for all possible actions in the assembly environment.
    It will skip actions that are not feasible, e.g. because the block would be placed at an angle that is too steep
    or faces that are already occupied.
    """
    # generator for all possible actions
    if offset_values is None:
        offset_values = [0.]

    # iterate target shapes and faces up to symmetries
    for target_shape_index in range(len(gym.shapes)):
        target_shape = gym.shapes[target_shape_index]
        for target_face in range(0, target_shape.num_faces_2d(), target_shape.num_faces_2d() // target_shape.n_effective_faces):

            # place new block on ground
            for offset_x in ground_pos_values:
                yield (-1, 0, target_shape_index, target_face, offset_x)

            # place new block on existing block
            for block_index in range(len(gym.assembly_env.blocks)):
                block = gym.assembly_env.blocks[block_index]

                for block_face in range(block.num_faces_2d()):
                    # print(block_index, block_face)
                    face_frame = block.get_face_frame_2d(block_face)

                    # skip faces where the angle w.r.t. to the horizontal plane is too large
                    angle = np.arccos(face_frame.normal[2])
                    if max_angle_rad is not None and angle > max_angle_rad:
                        # print(f"skipping because of angle {angle}")
                        continue

                    # skip occupied faces
                    if max_blocks_per_face and len(gym.block_graph.get((block_index, block_face), tuple())) >= max_blocks_per_face:
                        # print(f"skipping because of max_blocks_per_face")
                        continue

                    for offset_x in offset_values:
                        yield (block_index, block_face, target_shape_index, target_face, offset_x)


class AssemblyGym(gym.Env):
    # gym metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, reward_fct, shapes=None, obstacles=None, targets=None, render_mode=None, assembly_env=None, restrict_2d=False):
        self.target_blocks = []
        self.blocks = []
        self.reward_fct = reward_fct

        self.render_mode = render_mode
        self.restrict_2d = restrict_2d
        self.observation_space = None
        self.action_space = None
        self.action_history = None
        self.block_graph = None

        if not restrict_2d:
            # currently only 2d mode is tested
            raise NotImplementedError

        if assembly_env is None:
            assembly_env = AssemblyEnv(render=render_mode == 'human')
        self.assembly_env = assembly_env

        self.reset(shapes, obstacles, targets)  # Set obstacles and targets

    def terminated(self, assembly_env):
        return (not assembly_env.state_info['stable'] or
                assembly_env.state_info['collision'] or
                self.all_targets_reached())

    @property
    def num_targets(self):
        return len(self.targets)

    @property
    def num_obstacles(self):
        return len(self.obstacles)

    def distance_to_targets(self):
        if len(self.assembly_env.blocks) == 0:
            return self.num_targets * [np.inf]
        min_distances = []
        for target in self.targets:
            min_distances.append(min([distance_box_point(block.bounding_box, target) for block in self.assembly_env.blocks]))
        return min_distances

    def all_targets_reached(self):
        for target in self.targets:
            # check if target has been reached
            if not any([block.bounding_box.contains(target) for block in self.assembly_env.blocks]):
                return False
        return True

    def _num_targets_reached(self, new_block):
        n = 0
        for target in self.targets:
            # check if target has been reached
            if new_block.bounding_box.contains(target):
                n += 1
        return n

    def _update_action_and_obs_space(self):
        if self.restrict_2d:

            # For now, we assume shapes with 4 faces
            for shape in self.shapes:
                if not shape.num_faces_2d() == 4:
                    print(f"Warning: shape {shape.urdf_file} has {shape.num_faces_2d()} 2d faces, but 4 are assumed in the definition of the action space.")

            # Ideally, we want to conditionally restrict the action spaces depending on the shapes
            self.action_space = spaces.Tuple((spaces.Discrete(len(self.assembly_env.blocks) + 1, start=-1),
                                              spaces.Discrete(4),
                                              spaces.Discrete(len(self.shapes)),
                                              spaces.Discrete(4),
                                              spaces.Box(-1, 1, shape=(2,))))

        self.observation_space = spaces.Dict( {
            'blocks': spaces.Tuple(self.num_step * [self.action_space]),
            # 'targets': spaces.Tuple(len(self.targets) * [spaces.Box(-1, 1, shape=(3,))]),
            # 'obstacles': spaces.Tuple(len(self.obstacles) * [spaces.Box(-1, 1, shape=(3,))]),
            'unstable': spaces.Discrete(2),
            'collision': spaces.Discrete(2),
        })

    def _get_obs(self):
        return {
            'blocks': self.blocks,
            'targets': self.targets,
            'target_blocks': self.target_blocks,
            'obstacles': self.obstacles,
            'obstacle_blocks': self.assembly_env.obstacles,
            'stable': int(self.assembly_env.state_info['stable']),
            'collision': int(self.assembly_env.state_info['collision']),
            'collision_block': bool(self.assembly_env.state_info['collision_info']['blocks']),
            'collision_obstacle': bool(self.assembly_env.state_info['collision_info']['obstacles']),
            'collision_floor': bool(self.assembly_env.state_info['collision_info']['floor']),
            'collision_boundary': bool(self.assembly_env.state_info['collision_info']['bounding_box']),
        }

    def _get_info(self):
        return {
            'obstacles': self.obstacles,
            'targets': self.targets,
            'stable': self.assembly_env.state_info['stable'],
            'collision': self.assembly_env.state_info['collision'],
            'collision_info': self.assembly_env.state_info['collision_info'],
            'distance_to_targets': self.distance_to_targets(),
        }

    def _get_block_index(self, blocks, target_shape, target_shape_index):
        if target_shape == -1:
            return -1
        shape_count = 0
        additional_blocks = 0
        for b in blocks:
            if b[-1] == target_shape:
                shape_count += 1
                if shape_count > target_shape_index: # denotes the time when we reach the current chosen block in hte block list
                    break
            else:
                additional_blocks += 1
        target_shape_index += additional_blocks
        return target_shape_index

    def step(self, action, action_space='relative'):
        if action_space == 'relative':
            block_index, block_face, shape, shape_face, offset_x = action
            offset_y = 0

        elif action_space == 'relative_shape':
            target_shape, target_shape_index, block_face, shape, shape_face, offset_x, offset_y = action
            block_index = self._get_block_index(self.blocks, target_shape, target_shape_index)
        
        # compute new block orientation by aligning the selected frames
        if block_index == -1:
            block_frame = self.assembly_env.get_floor_frame()
        else:
            block_frame = self.assembly_env.blocks[block_index].get_face_frame_2d(block_face)

        shape_frame = self.shapes[shape].get_face_frame_2d(shape_face)

        offset = [offset_x, 0, offset_y]
        position, rotation = align_frames_2d(block_frame, shape_frame, offset)

        # create and add block to environment
        new_block = Block(self.shapes[shape], position, rotation.quaternion)
        self.assembly_env.add_block(new_block)

        # update history and spaces
        self.action_history.append(action)
        self.blocks.append(list(position) + list(rotation.quaternion) + [shape])

        new_block_index = len(self.assembly_env.blocks) - 1
        if not (block_index, block_face) in self.block_graph:
            self.block_graph[(block_index, block_face)] = []
        self.block_graph[(block_index, block_face)].append((new_block_index, shape_face))
        self.block_graph[(new_block_index, shape_face)] = [(block_index, block_face)]

        self._update_action_and_obs_space()

        # evaluate state and reward
        terminated = self.terminated(self.assembly_env)
        info = self._get_info()
        info["new_block_pos"] = position
        info["num_targets_reached"] = self._num_targets_reached(new_block)
        reward = self.reward_fct(self, info)
        observation = self._get_obs()
 
        return observation, reward, terminated, info

    def reset(self, shapes=None, obstacles=None, targets=None):
        self.assembly_env.reset()
        self.action_history = []
        self.blocks = []
        self.block_graph = {(-1, 0) : []}

        if shapes is not None:
            self.shapes = shapes

        if obstacles is not None:
            self.obstacles = obstacles

        if targets is not None:
            self.targets = targets

        self._update_action_and_obs_space()


        cube = Shape(urdf_file='../assembly_gym/shapes/cube.urdf')
        for position in self.obstacles:
            self.assembly_env.add_obstacle(Block(shape=cube, position=position))

        self.target_blocks = [Block(shape=cube, position=target) for target in self.targets]

        observation = self._get_obs()
        info = self._get_info()

        return observation

    @property
    def num_step(self):
        return len(self.action_history)

    def render(self):
        if self.render_mode == 'human':
            raise NotImplementedError
        elif self.render_mode == 'rgb_array':
            return get_rgb_array(self.assembly_env.client)

    def close(self):
        self.assembly_env.disconnect_client()
