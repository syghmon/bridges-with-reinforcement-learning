import os
from assembly_gym.utils.stability import is_stable_pybullet, is_stable_rbe
import numpy as np
import pybullet_data
import pybullet as p
from importlib.resources import files

from compas.geometry import Frame, Translation, Rotation, Quaternion, Box, Scale
from compas_robots import RobotModel
from compas_robots.resources import LocalPackageMeshLoader
from compas_assembly.datastructures import Block as CRA_Block

from compas_cra.datastructures import CRA_Assembly
from compas_cra.algorithms import assembly_interfaces_numpy

import assembly_gym
from assembly_gym.envs.compas_bullet import CompasClient
from assembly_gym.utils.geometry import bounding_box_collision, quaternion_distance, merge_coplanar_faces, maximum_tension


class Shape:
    """
    Describes a base block shape.
    The main use is to extract the mesh from a urdf file.

    Additionally, 2d information can be extracted.
    """

    def __init__(self, mesh=None, urdf_file=None, name="", receiving_faces_2d=None, target_faces_2d=None):
        self.urdf_file = None
        self.mesh = None
        self.name = name

        if mesh is not None:
            self.from_mesh(mesh)
        elif urdf_file is not None:
            self.from_urdf(urdf_file)

        self._target_faces_2d = target_faces_2d
        self._receiving_faces_2d = receiving_faces_2d

    def scale(self, factor):
        self.mesh.transform(Scale.from_factors([factor, factor, factor]))

    def from_mesh(self, mesh, merge_faces=True):
        if merge_faces:
            merge_coplanar_faces(mesh)
        self.mesh = mesh.copy(cls=CRA_Block)
        self.bounding_box = mesh.aabb()
        self._2d_faces = [face for face in mesh.faces() if np.abs(mesh.face_normal(face)[1]) < 1e-6]
        self._all_faces = self._2d_faces + [face for face in mesh.faces() if face not in self._2d_faces]


    def from_urdf(self, urdf_file, package='blocks', merge_faces=True):
        if not os.path.exists(urdf_file):
            urdf_file = str(files(assembly_gym).joinpath('../').joinpath(urdf_file))
            
            if not os.path.exists(urdf_file):
                raise FileNotFoundError(f"URDF file not found: {urdf_file}")

        self.urdf_file = urdf_file

        # compute vertices from urdf file
        base_path = os.path.split(urdf_file)[0]
        robot = RobotModel.from_urdf_file(self.urdf_file)
        robot.load_geometry(LocalPackageMeshLoader(base_path, package))
        mesh = robot.links[0].collision[0].geometry.shape.meshes[0]
        self.from_mesh(mesh, merge_faces=merge_faces)

    @property
    def num_faces(self):
        return len(self._all_faces)
    
    @property
    def faces(self):
        return self._all_faces
    
    @property
    def faces_2d(self):
        return range(self.num_faces_2d)
    
    @property
    def target_faces_2d(self):
        return self._target_faces_2d or self.faces_2d
    
    @property
    def receiving_faces_2d(self):
        return self._receiving_faces_2d or self.faces_2d

    @property
    def num_faces_2d(self):
        return len(self._2d_faces)

    @property
    def vertices(self):
        for vertex in self.mesh.vertices():
            vertex = self.mesh.vertex_coordinates(vertex)
            yield vertex

    @property
    def vertices_2d(self):
        # find forward looking face

        for face, vertices in self.mesh.face.items():
            n = self.mesh.face_normal(face)
            if np.abs(n[1] - 1) < 1e-3:
                break
        
        for i in vertices:
            vertex = self.mesh.vertex_coordinates(i)
            # if vertex[1] > 0:
            yield [vertex[0], vertex[2]]


    def get_face_frame(self, face):
        return Frame.from_plane(self.mesh.face_plane(self._all_faces[face]))

    def get_face_frame_2d(self, face):
        # construct a frame for the face where the first coordinate corresponds to the x-axis
        normal = self.mesh.face_normal(self._2d_faces[face])
        yaxis = [0, 1, 0]
        return Frame(point=self.mesh.face_center(self._2d_faces[face]),
                     xaxis=-np.cross(normal, yaxis),
                     yaxis=yaxis)

    def contains_2d(self, points):
        contains = np.ones(len(points), dtype=bool)

        for i in self.faces_2d:
            frame = self.get_face_frame_2d(i)

            offset = np.array([frame.point[0], frame.point[2]])
            normal = np.array([frame.normal[0], frame.normal[2]])

            contains = contains & (np.dot(points - offset, normal) <= 0)

        return contains
    

class Block(Shape):
    """
    A block is a shape with a position and orientation.
    Also stores an object_id to identify the block in the pybullet environment.
    """

    def __init__(self, shape, position, orientation=None, object_id=None):
        self.shape = shape
        self.position = position
        self.orientation = orientation if orientation is not None else Quaternion(1., 0., 0., 0.)
        self.object_id = object_id
        self.is_static = False
        self._transformation = Translation.from_vector(self.position) * Rotation.from_quaternion(self.orientation)
        super().__init__(mesh=self.shape.mesh.transformed(self._transformation))

    def __repr__(self):
        return f"Block ({self.object_id})"


class AssemblyEnv:
    """
    Assembly environment
    """

    def __init__(self, render=False, bounds=None, stability='rbe', mu=0.8, density=1.0, cra_env=True, pybullet_env=False):
        self.obstacles = []
        self.blocks = []
        if bounds is None:
            bounds = np.array([[-3., -3., -1], [7., 7., 9.]])
        self.bounds = bounds
        self._state_info = None
        self.mu = mu
        self.density = density

        # create a pybullet client
        self.client = None

        if pybullet_env:
            self.client = CompasClient(connection_type='gui' if render else 'direct')
            self.client.connect()

        # stability function
        if stability == 'pybullet':
            self.stability_fct = is_stable_pybullet
        elif stability == 'rbe':
            self.stability_fct = is_stable_rbe
        elif stability is None:
            self.stability_fct = lambda x: (None, None)
        else:
            self.stability_fct = stability
        
        # create cra assembly
        self.cra_assembly = None
        if cra_env:
            self.cra_assembly = CRA_Assembly()
        
        self.reset()

    def reset(self):
        self.obstacles = []
        self.blocks = []
        self.is_block_frozen = False
        self.frozen_block_index = None
        self._reset_cra_assembly()
        self._reset_pybullet()
        self._update_state_info()
        

    def _reset_pybullet(self, time_step=1/240):
        if self.client is None:
            return
        self.client.resetSimulation()

        # Import Ground URDF
        self._floor_object_id = self.client.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0], [0, 0, 0, 1])
        # adjust friction
        self.client.changeDynamics(self._floor_object_id, -1, lateralFriction=self.mu)

        # Optionally, adjust spinningFriction and rollingFriction as needed
        #self.client.changeDynamics(self._floor_object_id, -1, spinningFriction=100.0)
        #self.client.changeDynamics(self._floor_object_id, -1, rollingFriction=100.0)

        # Set Gravity Simulation
        self.client.setGravity(0, 0, -9.8)
        self.client.setTimeStep(time_step)
        self.client.setRealTimeSimulation(0)  # if it is "1" it will be locked

    def disconnect_client(self):
        """
        Disconnect the pybullet client
        """
        if self.client is not None:
            self.client.disconnect()

    def restore(self):
        """
        restore the blocks in the scene
        """
        self._reset_pybullet()
        for i, block in enumerate(self.blocks):
            self._add_block_to_pybullet(block)
            if block.is_static:
                self.freeze_block(i)

        for i, obs in enumerate(self.obstacles):
            self._add_block_to_pybullet(obs, use_fixed_base=True)
    
    def _clear_blocks_pybullet(self):
        """
        clear all the blocks in the scene
        """
        if self.client is not None:
            for block in self.blocks:
                self.client.removeBody(block.object_id)

    def _clear_obstacles_pybullet(self):
        """
        clear all the blocks in the scene
        """
        if self.client is not None:
            for block in self.obstacles:
                self.client.removeBody(block.object_id)

    def _add_block_to_pybullet(self, block, use_fixed_base=False):
        if self.client is None:
            return
        # add to pybullet environment
        orientation = block.orientation
        if isinstance(block.orientation, Quaternion):
            orientation = (block.orientation.x, block.orientation.y, block.orientation.z, block.orientation.w)

        block.object_id = self.client.loadURDF(block.shape.urdf_file, block.position, orientation,
                                               useFixedBase=use_fixed_base) #, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        # Set friction for the block
        self.client.changeDynamics(block.object_id, -1, lateralFriction=self.mu)
        # set mass according to the density
        self.client.changeDynamics(block.object_id, -1, mass=self.density * block.mesh.volume())
        # Optionally, adjust spinningFriction and rollingFriction as needed
        #self.client.changeDynamics(block.object_id, -1, spinningFriction=100.0)
        #self.client.changeDynamics(block.object_id, -1, rollingFriction=.1)

    def _reset_cra_assembly(self):
        """
        Reset the CRA assembly by adding the support block and all the blocks in the scene.
        Note (Johannes, 29.5.2024): I tried to incrementally add blocks to the assembly, but this seems to create artifacts.
        If the full reset becomes too slow (unlikely), we can revisit this.
        """
        if self.cra_assembly is None:
            return
        self.cra_assembly = CRA_Assembly()
        width = self.bounds[1][0] - self.bounds[0][0]
        depth = self.bounds[1][1] - self.bounds[0][1]
        thickness = 0.05 * width
        frame = Frame.worldXY().transformed(Translation.from_vector([0.0, 0.0, -thickness/2]))
        support = Box(width, depth, thickness, frame=frame)
        self.cra_assembly.add_block(CRA_Block.from_shape(support), node=-1)
        self.cra_assembly.set_boundary_condition(-1)

        for i, block in enumerate(self.blocks):
            self.cra_assembly.add_block_from_mesh(block.mesh, node=i)
            if block.is_static:
                self.cra_assembly.set_boundary_condition(i)

        if len(self.blocks) > 0:
            assembly_interfaces_numpy(self.cra_assembly, amin=0.001)


    def _update_state_info(self):
        if self.client is not None:
            collision, collision_info = self._check_collision()
        else:
            collision = False
            collision_info = {'obstacles': [], 'blocks': [], 'floor': False, 'bounding_box': False}
            
        self._state_info = {
            "last_block": self.blocks[-1] if self.blocks else None,
            "collision": collision,
            "collision_info": collision_info,
            # "frozen": self.is_frozen(),
            "frozen_block": self.frozen_block_index
        }

        is_stable, stability_info = self.stability_fct(self)
        self._state_info["stable"] = is_stable
        self._state_info["stability_info"] = stability_info


    def add_block(self, block):
        # add block
        self._add_block_to_pybullet(block)
        self.blocks.append(block)
        self._reset_cra_assembly()
        self._update_state_info()
        return self._state_info

    @property
    def state_info(self):
        return self._state_info

    def get_floor_frame(self):
        return Frame.worldXY()

    def add_obstacle(self, obstacle):
        self._add_block_to_pybullet(obstacle, use_fixed_base=True)
        self.obstacles.append(obstacle)

    def _check_collision(self, tol=0.005):
        collision_info = {
            'obstacles': [],
            'blocks': [],
            'floor': False,
            'bounding_box': False
        }
        if len(self.blocks) == 0:
            return False, collision_info
        
        # check collision for last block only
        block = self.blocks[-1]

        # check bounding box
        if np.any(block.position < self.bounds[0]) or np.any(block.position > self.bounds[1]):
            collision_info['bounding_box'] = True

        # perform the pybullet collision detection
        #self.client.performCollisionDetection()

        # check collision with other blocks
        for b in self.blocks:
            if b.object_id == block.object_id:
                continue

            # check penetration depth
            contact_points = self.client.getContactPoints(b.object_id, block.object_id)
            if any([p[8] < -tol for p in contact_points]):
                collision_info['blocks'].append(b.object_id)

        # check collision with floor
        contact_points = self.client.getContactPoints(self._floor_object_id, block.object_id)
        if any([p[8] < -tol for p in contact_points]):
            collision_info['floor'] = True

        for obs in self.obstacles:
            # # based on bounding box
            # if bounding_box_collision(block, obs):
            #     collision_info['obstacles'].append(obs.object_id)

            # pybullet collision detection
            contact_points = self.client.getContactPoints(obs.object_id, block.object_id)
            if any([p[8] < - tol for p in contact_points]):
                collision_info['obstacles'].append(obs.object_id)

        return any(collision_info.values()), collision_info
   

    def is_stable(self):
        return self._state_info['stable']
        
    def simulate(self, steps=240):
        for i in range(steps):
            self.client.step_simulation()

    def realtime(self):
        self.client.setRealTimeSimulation(1)

    def freeze_block(self, block_index, freeze_color=(1, 1, 0, 1)):
        block = self.blocks[block_index]
        block.is_static = True

        # freeze in cra assembly
        if self.cra_assembly is not None:
            self.cra_assembly.set_boundary_condition(block_index)

        # freeze in pybullet by setting mass to zero
        if self.client is not None:
            self.client.changeDynamics(block.object_id, -1, mass=0.0)
            self.client.change_object_color(block.object_id, freeze_color)


    def unfreeze_block(self, block_index, default_color=(0, 0, 1, 1)):
        block = self.blocks[block_index]
        block.is_static = False
        
        # unfreeze in pybullet by restoring mass
        if self.client is not None:
            self.client.changeDynamics(block.object_id, -1, mass=self.density * block.mesh.volume())
            self.client.change_object_color(block.object_id, default_color)

        # unfreeze in cra assembly (unfortunately, we need to reset all boundary conditions)
        if self.cra_assembly is not None:
            self.cra_assembly.unset_boundary_conditions()
            self.cra_assembly.set_boundary_condition(-1)  # floor is fixed

            # Johannes: resetting fully is potentially slow, but currently this is the only way to make sure the interfaces are detected correctly
            self._reset_cra_assembly()

        
            for i, block in enumerate(self.blocks):
                if block.is_static:
                    self.cra_assembly.set_boundary_condition(i)
