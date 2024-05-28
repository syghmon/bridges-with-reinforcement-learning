
import torch
import numpy as np
from assembly_gym.envs.gym_env import Action, AssemblyGym


def generate_actions(gym : AssemblyGym, x_discr_ground, offset_values=None, max_angle_rad=np.pi/2-0.01, max_blocks_per_face=1, include_frozen=False, x_block_offset=None):
    """
    This is a generator for all possible actions in the assembly environment.
    It will skip actions that are not feasible, e.g. because the block would be placed at an angle that is too steep
    or faces that are already occupied.
    """
    # generator for all possible actions
    if offset_values is None:
        offset_values = [0.]

    # iterate target shapes and faces up to symmetries
    for shape_index in range(len(gym.shapes)):
        shape = gym.shapes[shape_index]
        for face in shape.target_faces_2d:

            # place new block on ground
            for offset_x in x_discr_ground:
                if include_frozen:
                    raise NotImplementedError
                else:
                    yield Action(-1, 0, shape_index, face, offset_x, offset_y=0.)

            # place new block on existing block
            for target_block in range(len(gym.assembly_env.blocks)):
                block = gym.assembly_env.blocks[target_block]

                for target_face in block.receiving_faces_2d:
                    # print(block_index, block_face)
                    face_frame = block.get_face_frame_2d(target_face)

                    # skip faces where the angle w.r.t. to the horizontal plane is too large
                    angle = np.arccos(face_frame.normal[2])
                    if max_angle_rad is not None and angle > max_angle_rad:
                        # print(f"skipping because of angle {angle}")
                        continue

                    # skip occupied faces
                    if max_blocks_per_face and len(gym.block_graph.get((target_block, target_face), tuple())) >= max_blocks_per_face:
                        # print(f"skipping because of max_blocks_per_face")
                        continue

                    for offset_x in offset_values:
                        if include_frozen:
                            raise NotImplementedError
                        else:
                            yield Action(target_block, target_face, shape_index, face, offset_x, offset_y=0.)


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


def filter_actions(available_actions, action_features, block_features, obstacle_features):
    """ 
    Filter available actions based on immediate collisions by checking image overlap.
    """
    mask = torch.zeros(len(available_actions), dtype=bool)
    reduced_available_actions= []
    for i, action in enumerate(available_actions):
        if torch.sum(action_features[i] * block_features) == 0 and torch.sum(action_features[i] * obstacle_features) == 0:
            mask[i] = True
            reduced_available_actions.append(action)
    
    return reduced_available_actions, action_features[mask]
