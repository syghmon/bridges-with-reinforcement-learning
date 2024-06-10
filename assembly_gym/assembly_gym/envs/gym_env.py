from collections import namedtuple
import numpy as np
import gymnasium as gym
from compas.geometry import Quaternion

from assembly_gym.envs.assembly_env import AssemblyEnv, Block, Shape
from assembly_gym.utils import align_frames_2d, distance_box_point
from assembly_gym.utils.rendering import get_rgb_array


def sparse_reward(gym_env, obs, info):
    if gym_env.assembly_env.state_info['collision'] or not gym_env.assembly_env.state_info['stable']:
        return -1

    num_targets_reached = len(obs['targets_reached'])
    if not gym_env.all_targets_reached(): 
        return -1 + num_targets_reached

    return num_targets_reached

def cliff_setup(H=0.04, num_obstacles=5, rescale=1.):
    # Define the shape
    trapezoid = Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")
    trapezoid.scale(rescale)
    shapes = [trapezoid]

    # Define the targets and obstacles
    # In this setup, obstacles are stacked horizontally along the x-axis
    x_start = 0.1  # starting x position for the obstacles
    targets = [(x_start + num_obstacles * H, 0, H / 2)]  # target is at the end of the horizontal stack
    obstacles = [(x_start + i * H, 0, H / 2) for i in range(num_obstacles)]

    return dict(shapes=shapes, obstacles=obstacles, targets=targets)



def bridge_setup(H=0.04, num_stories=1, rescale=1.):
    trapezoid = Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")
    trapezoid.scale(rescale)
    shapes = [trapezoid]
    targets = [ (0.5 , 0, num_stories * H + H/2) ]
    obstacles = [(targets[0][0], 0., i*H + H/2) for i in range(num_stories)]
    #targets = [ (0.5 , 0, np.random.uniform(0.07, 0.12)) ]
    #obstacles = [(targets[0][0], 0., 0.02)]

    return dict(shapes=shapes, obstacles=obstacles, targets=targets)


def tower_setup(num_targets = 3, targets=None):
    if targets is None:
        x_min = 0.3
        x_max = 0.7
        z_min = 0.
        z_max = 0.3

        targets = [(np.random.uniform(x_min, x_max), 0, np.random.uniform(z_min, z_max)) for _ in range(num_targets)]
    cube = Shape(urdf_file='shapes/cube.urdf', name="cube", receiving_faces_2d=[3], target_faces_2d=[1])
    rectangle = Shape(urdf_file='shapes/block.urdf', name="rectangle", receiving_faces_2d=[3], target_faces_2d=[0])
    shapes = [cube, rectangle]
    obstacles = []
    #obstacles = [[targets[0][0], 0, 0.02]]
    return dict(shapes=shapes, obstacles=obstacles, targets=targets)


def hard_tower_setup():
    trapezoid = Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")
    cube = Shape(urdf_file='shapes/cube.urdf', name="cube", receiving_faces_2d=[0], target_faces_2d=[2])
    shapes = [trapezoid, cube]
    targets = [[0.2, 0, 0], [0.2, 0, 0.3]]
    obstacles = [[0.2, 0, 0.1]]
    return dict(shapes=shapes, obstacles=obstacles, targets=targets)


def connecting_setup(): # Similar as the connecting setup in Deepmind's paper
    rectangle = Shape(urdf_file='shapes/block.urdf', name="rectangle", receiving_faces_2d=[3], target_faces_2d=[0])
    cube = Shape(urdf_file='shapes/cube.urdf', name="cube", receiving_faces_2d=[3], target_faces_2d=[1])
    shapes = [rectangle, cube]
    targets = [[np.random.uniform(0.4, 0.6), 0, 0.175], [np.random.uniform(0.4, 0.6), 0, 0.175], [np.random.uniform(0.4, 0.6), 0, 0.175]]
    obstacles = [[np.random.uniform(0.4, 0.47), 0, np.random.uniform(0.025, 0.125)], [np.random.uniform(0.53, 0.6), 0, np.random.uniform(0.025, 0.125)]]
    #obstacles = [[x_target + np.random.uniform(-0.03, 0.03), 0, 0.025]]

    return dict(shapes=shapes, obstacles=obstacles, targets=targets)


Action = namedtuple('Action', ('target_block', 'target_face', 'shape', 'face', 'offset_x', 'offset_y'))


class AssemblyGym(gym.Env):
    # gym metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, reward_fct, shapes=None, obstacles=None, targets=None, render_mode=None, assembly_env=None, restrict_2d=False, max_steps=None):
        self.blocks = []
        self.shapes = []
        self.obstacles = []
        self.targets = []
        self.reward_fct = reward_fct

        self.render_mode = render_mode
        self.restrict_2d = restrict_2d
        self.observation_space = None
        self.action_space = None
        self.action_history = None
        self.block_graph = None
        self.max_steps = max_steps

        if not restrict_2d:
            # currently only 2d mode is tested
            raise NotImplementedError

        if assembly_env is None:
            assembly_env = AssemblyEnv(render=render_mode == 'human')
        self.assembly_env = assembly_env

        self.reset(shapes, obstacles, targets)  # Set obstacles and targets

    def terminated(self, assembly_env):
        terminated = not assembly_env.state_info['stable'] or assembly_env.state_info['collision'] or self.all_targets_reached()
        truncated = self.max_steps and len(self.blocks) >= self.max_steps
        return terminated, truncated

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

    def _update_targets(self, new_block):
        targets_reached = []
        for target in self.targets_remaining:
            if new_block.bounding_box.contains_point(target):
                self.targets_reached.append(target)
                self.targets_remaining.remove(target)
        return targets_reached

    def all_targets_reached(self):
        return len(self.targets_remaining) == 0

    def _update_action_and_obs_space(self):
        # ToDo: Our action and observation space is dynamic and we need to update it after each step
        self.action_space = None
        self.observation_space = None


    def _get_obs(self):
        # TODO: Do we really need to deepcopy these?
        return {
            'blocks': self.blocks,
            'stable': int(self.assembly_env.state_info['stable']),
            'collision': int(self.assembly_env.state_info['collision']),
            'collision_block': bool(self.assembly_env.state_info['collision_info']['blocks']),
            'collision_obstacle': bool(self.assembly_env.state_info['collision_info']['obstacles']),
            'collision_floor': bool(self.assembly_env.state_info['collision_info']['floor']),
            'collision_boundary': bool(self.assembly_env.state_info['collision_info']['bounding_box']),
            'frozen_block' : self.assembly_env.frozen_block_index,
            'obstacles': self.obstacles,
            'obstacle_blocks': self.assembly_env.obstacles,
            'targets': self.targets,
            'targets_remaining' : self.targets_remaining,
            'targets_reached' : self.targets_reached,
            'distance_to_targets': self.distance_to_targets(),
        }

    def _get_info(self):
        return {
            'blocks_initial_state' : self.assembly_env.state_info.get('pybullet_initial_state'),
            'blocks_final_state' : self.assembly_env.state_info.get('pybullet_final_state'),
        }
    
    # TODO: Previous implementation for refernce, we can do this computation outside the environment to keep the inferface simple
    # def _get_block_index(self, blocks, target_shape, target_shape_index):
    #     if target_shape == -1:
    #         return -1
    #     shape_count = 0
    #     additional_blocks = 0
    #     for b in blocks:
    #         if b[-1] == target_shape:
    #             shape_count += 1
    #             if shape_count > target_shape_index: # denotes the time when we reach the current chosen block in hte block list
    #                 break
    #         else:
    #             additional_blocks += 1
    #     target_shape_index += additional_blocks
    #     return target_shape_index

    # TODO: Currently we have create_block() below, but if needed we can have this function to get the position and orientation of the block
    # def action_to_position(self, action):
    #     target_shape, target_shape_index, block_face, shape, shape_face, offset_x, offset_y = action
    #     # block_index = self._get_block_index(self.blocks, target_shape, target_shape_index)

    #     # compute new block orientation by aligning the selected frames
    #     if block_index == -1:
    #         block_frame = self.assembly_env.get_floor_frame()
    #     else:
    #         block_frame = self.assembly_env.blocks[block_index].get_face_frame_2d(block_face)

    #     shape_frame = self.shapes[shape].get_face_frame_2d(shape_face)

    #     offset = [offset_x, 0, offset_y]
    #     position, rotation = align_frames_2d(block_frame, shape_frame, offset)

    def create_block(self, action : Action):
        # compute new block orientation by aligning the selected frames
        if action.target_block == -1:
            block_frame = self.assembly_env.get_floor_frame()
        else:
            block_frame = self.assembly_env.blocks[action.target_block].get_face_frame_2d(action.target_face)

        shape_frame = self.shapes[action.shape].get_face_frame_2d(action.face)

        offset = [action.offset_x, 0, action.offset_y]
        position, rotation = align_frames_2d(block_frame, shape_frame, offset)
        new_block = Block(self.shapes[action.shape], position=position, orientation=rotation.quaternion)
        return new_block

    def step(self, action : Action):
        # create and add block to environment
        new_block = self.create_block(action)
        self.assembly_env.add_block(new_block)

        # update history and spaces
        self.action_history.append(action)
        self.blocks.append(new_block)

        # update block graph
        new_block_index = len(self.assembly_env.blocks) - 1
        if not (action.target_block, action.target_face) in self.block_graph:
            self.block_graph[(action.target_block, action.target_face)] = []
        self.block_graph[(action.target_block, action.target_face)].append((new_block_index, action.face))
        self.block_graph[(new_block_index, action.face)] = [(action.target_block, action.target_face)]

        self.assembly_env.unfreeze_block()
        self.assembly_env.freeze_block(len(self.assembly_env.blocks) - 1)
        
        self._update_targets(new_block)
        self._update_action_and_obs_space()

        # evaluate state and reward
        terminated, truncated = self.terminated(self.assembly_env)
        info = self._get_info()
        observation = self._get_obs()
        reward = self.reward_fct(self, observation, info)
 
        return observation, reward, terminated, truncated, info

    def reset(self, shapes=None, obstacles=None, targets=None, blocks=None):
        self.assembly_env.reset()
        self.action_history = []
        self.blocks = []
        self.block_graph = {(-1, 0) : []}
        self.targets_reached = []

        if shapes is not None:
            self.shapes = shapes

        if obstacles is not None:
            self.obstacles = obstacles

        if targets is not None:
            self.targets = targets

        if blocks is not None:
            self.blocks = blocks

        self.targets_remaining = self.targets.copy()
        self._update_action_and_obs_space()

        small_cube = Shape(urdf_file='shapes/small_cube.urdf')

        for b in self.blocks:
            new_block = Block(self.shapes[b[-1]], b[:3], Quaternion(*b[3:7]))
            self.assembly_env.add_block(new_block)

        for position in self.obstacles:
            self.assembly_env.add_obstacle(Block(shape=small_cube, position=position))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

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
