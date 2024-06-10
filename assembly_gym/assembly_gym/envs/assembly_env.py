import os
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
from compas_cra.equilibrium import cra_solve, cra_penalty_solve, rbe_solve

import assembly_gym
from assembly_gym.envs.compas_bullet import CompasClient
from assembly_gym.utils.geometry import quaternion_distance, merge_coplanar_faces, maximum_tension


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
        self.mesh = mesh
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
        for vertex in self.mesh.vertices():
            vertex = self.mesh.vertex_coordinates(vertex)
            if vertex[1] > 0:
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
        self._transformation = Translation.from_vector(self.position) * Rotation.from_quaternion(self.orientation)
        super().__init__(mesh=self.shape.mesh.transformed(self._transformation))

    def __repr__(self):
        return f"Block ({self.object_id})"


class AssemblyEnv:
    """
    Assembly environment
    """

    def __init__(self, render=False, bounds=None, stability_checks=('rbe')):
        self.obstacles = []
        self.blocks = []
        if bounds is None:
            bounds = np.array([[0., -1., 0], [1., 1., 1.]])
        self.bounds = bounds
        self._state_info = None
        self.is_block_frozen = False
        self.frozen_block_index = None 
        self.stability_checks = stability_checks

        # create a pybullet client
        self.client = CompasClient(connection_type='gui' if render else 'direct')
        self.client.connect()

        # create cra assembly
        self.assembly = CRA_Assembly()
        self.reset()

    def reset(self, time_step=1/240):
        self.obstacles = []
        self.blocks = []
        self.is_block_frozen = False
        self.frozen_block_index = None  
        self._update_state_info()

        self.client.resetSimulation()

        # Import Ground URDF
        self._floor_object_id = self.client.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0], [0, 0, 0, 1])
       
        #self.client.changeDynamics(self._floor_object_id, -1, lateralFriction=100.0)

        # Optionally, adjust spinningFriction and rollingFriction as needed
        #self.client.changeDynamics(self._floor_object_id, -1, spinningFriction=100.0)
        #self.client.changeDynamics(self._floor_object_id, -1, rollingFriction=100.0)

        # Set Gravity Simulation
        self.client.setGravity(0, 0, -9.8)
        self.client.setTimeStep(time_step)
        self.client.setRealTimeSimulation(0)  # if it is "1" it will be locked

        # reset the cra assembly
        self._reset_cra_assembly()

    def disconnect_client(self):
        """
        Disconnect the pybullet client
        """
        self.client.disconnect()

    def restore(self):
        """
        restore the blocks in the scene
        """
        self._clear_blocks()
        for block in self.blocks:
            self._add_block_to_client(block)
        if self.is_block_frozen:
            self.freeze_block(self.frozen_block_index)
    
    def _clear_blocks(self):
        """
        clear all the blocks in the scene
        """
        for block in self.blocks:
            self.client.removeBody(block.object_id)

    def _clear_obstacles(self):
        """
        clear all the blocks in the scene
        """
        for block in self.obstacles:
            self.client.removeBody(block.object_id)

    def _add_block_to_client(self, block, use_fixed_base=False):
        # add to pybullet environment
        orientation = block.orientation
        if isinstance(block.orientation, Quaternion):
            orientation = (block.orientation.x, block.orientation.y, block.orientation.z, block.orientation.w)
        block.object_id = self.client.loadURDF(block.shape.urdf_file, block.position, orientation,
                                               useFixedBase=use_fixed_base) #, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        # Set high friction for the block
        #self.client.changeDynamics(block.object_id, -1, lateralFriction=100.0)
        # Optionally, adjust spinningFriction and rollingFriction as needed
        #self.client.changeDynamics(block.object_id, -1, spinningFriction=100.0)
        #self.client.changeDynamics(block.object_id, -1, rollingFriction=.1)

    def _reset_cra_assembly(self):
        """
        Reset the CRA assembly by adding the support block and all the blocks in the scene.
        Note (Johannes, 29.5.2024): I tried to incrementally add blocks to the assembly, but this seems to create artifacts.
        If the full reset becomes too slow (unlikely), we can revisit this.
        """
        self.assembly = CRA_Assembly()
        width = self.bounds[1][0] - self.bounds[0][0]
        depth = self.bounds[1][1] - self.bounds[0][1]
        thickness = 0.05 * width
        frame = Frame.worldXY().transformed(Translation.from_vector([0.0, 0.0, -thickness/2]))
        support = Box(width, depth, thickness, frame=frame)
        self.assembly.add_block(CRA_Block.from_shape(support), node=-1)
        self.assembly.set_boundary_condition(-1)

        for i, block in enumerate(self.blocks):
            self.assembly.add_block_from_mesh(block.mesh, node=i)

        if self.is_block_frozen:
            self.assembly.set_boundary_condition(self.frozen_block_index)
        
        if len(self.blocks) > 0:
            assembly_interfaces_numpy(self.assembly, amin=0.001)


    def _update_state_info(self):
        collision, collision_info = self._check_collision()
        self._state_info = {
            "last_block": self.blocks[-1] if self.blocks else None,
            "collision": collision,
            "collision_info": collision_info,
            # "frozen": self.is_frozen(),
            "frozen_block": self.frozen_block_index
        }


        # compute stability information
        if 'pybullet' in self.stability_checks:
            pybullet_stable, pybullet_initial_state, pybullet_final_state = self.is_stable_pybullet()
            self._state_info["pybullet_stable"] = pybullet_stable
            self._state_info["pybullet_initial_state"] = pybullet_initial_state
            self._state_info["pybullet_final_state"] = pybullet_final_state

            if self.stability_checks[0] == 'pybullet':
                self._state_info['stable'] = pybullet_stable
        
        if 'cra' in self.stability_checks:
            cra_stable = self.is_stable_cra()
            self._state_info["cra_stable"] = cra_stable

            if self.stability_checks[0] == 'cra':
                self._state_info['stable'] = cra_stable

        if 'cra_penalty' in self.stability_checks:
            cra_penalty_stable = self.is_stable_cra_penalty()
            self._state_info["cra_penalty_stable"] = cra_penalty_stable

            if self.stability_checks[0] == 'cra_penalty':
                self._state_info['stable'] = cra_penalty_stable

        if 'rbe' in self.stability_checks:
            rbe_stable = self.is_stable_rbe()
            self._state_info["rbe_stable"] = rbe_stable

            if self.stability_checks[0] == 'rbe':
                self._state_info['stable'] = rbe_stable

        if 'rbe_penalty' in self.stability_checks:
            rbe_penalty_stable = self.is_stable_rbe_penalty()
            self._state_info["rbe_penalty_stable"] = rbe_penalty_stable

            if self.stability_checks[0] == 'rbe_penalty':
                self._state_info['stable'] = rbe_penalty_stable

        #MESSY CODE
        rbe_stable = self.is_stable_rbe()
        self._state_info["rbe_stable"] = rbe_stable

        self._state_info['stable'] = rbe_stable
        #MESSY CODE


    def add_block(self, block):
        # add block
        self._add_block_to_client(block)
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
        self._add_block_to_client(obstacle, use_fixed_base=True)
        self.obstacles.append(obstacle)

    def _check_collision(self, threshold=0.005):
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
        self.client.performCollisionDetection()

        # check collision with obstacles
        for obs in self.obstacles:
            contact_points = self.client.getContactPoints(obs.object_id, block.object_id)
            if any([p[8] < -threshold for p in contact_points]):
                collision_info['obstacles'].append(obs.object_id)

        # check collision with other blocks
        for b in self.blocks:
            if b.object_id == block.object_id:
                continue

            # check penetration depth
            contact_points = self.client.getContactPoints(b.object_id, block.object_id)
            if any([p[8] < -threshold for p in contact_points]):
                collision_info['blocks'].append(b.object_id)

        # check collision with floor
        contact_points = self.client.getContactPoints(self._floor_object_id, block.object_id)
        if any([p[8] < -threshold for p in contact_points]):
            collision_info['floor'] = True

        return any(collision_info.values()), collision_info
    
    def is_stable_cra(self, density=0.1, d_bnd=0.05):
        if self.assembly.graph.number_of_edges() == 0:
            return True
        
        try:
            cra_solve(self.assembly, density=density, d_bnd=d_bnd)
        except ValueError as e:
            if e.args[0] == "infeasible":
                return False
            else:
                raise ValueError(*e.args, 'Error in CRA solve')
        
        self._cra_solution_graph = self.assembly.graph.copy()
        return True
    
    def is_stable_cra_penalty(self, density=0.1, d_bnd=0.05, tension_threshold=0.001):
        if self.assembly.graph.number_of_edges() == 0:
            return True
        
        self._cra_penalty_solution_graph = self.assembly.graph.copy()
        cra_penalty_solve(self.assembly, density=density, d_bnd=d_bnd)
        if maximum_tension(self.assembly) > tension_threshold:
            return False
        return True
    
    def is_stable_rbe(self, density=0.1):
        if self.assembly.graph.number_of_edges() == 0:
            return True
        
        try:
            rbe_solve(self.assembly, density=density, penalty=False)
        except ValueError as e:
            if e.args[0] == "infeasible":
                return False
            else:
                raise ValueError(*e.args, 'Error in RBE solve')
        self._rbe_solution_graph = self.assembly.graph.copy()
        return True
    
    def is_stable_rbe_penalty(self, density=0.1, tension_threshold=0.001):
        if self.assembly.graph.number_of_edges() == 0:
            return True

        rbe_solve(self.assembly, density=density, penalty=True)
        if maximum_tension(self.assembly) > tension_threshold:
            return False
        self._rbe_penalty_solution_graph = self.assembly.graph.copy()
        return True

    def is_stable(self):
        return self._state_info['stable']

    def is_stable_pybullet(self, distance_threshold=0.02, angle_threshold=0.2):
        initial_states = []
        final_states = []
        
        # Store initial states for all blocks
        for block in self.blocks:
            position, orientation = self.client.getBasePositionAndOrientation(block.object_id)
            initial_states.append((position, Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])))
        
        # Pybullet simulation
        self.simulate(steps=3000)
        is_stable = True
        # Compare initial and final states for all blocks
        for index, block in enumerate(self.blocks):
            new_position, new_orientation = self.client.getBasePositionAndOrientation(block.object_id)
            final_states.append((new_position, Quaternion(x=new_orientation[0], y=new_orientation[1], z=new_orientation[2], w=new_orientation[3])))
            initial_position, initial_orientation = initial_states[index]

            dist = np.linalg.norm(np.array(initial_position) - np.array(new_position))
            new_orientation = Quaternion(x=new_orientation[0], y=new_orientation[1], z=new_orientation[2], w=new_orientation[3])
            angle = quaternion_distance(initial_orientation, new_orientation) # gives Nan on non-zero angles...

            collision_obs = False
            for obs in self.obstacles:
                contact_points = self.client.getContactPoints(obs.object_id, block.object_id)
                if len(contact_points) > 0:
                    #print("Collision with block")
                    collision_obs = True
                
            # Update is_stable based on the thresholds
            is_stable = is_stable and (dist < distance_threshold) and (angle < angle_threshold) and (not collision_obs)
            
        self.restore()
        return is_stable, initial_states, final_states
        
    def simulate(self, steps=240):
        for i in range(steps):
            self.client.step_simulation()

    def realtime(self):
        self.client.setRealTimeSimulation(1)

    def freeze_block(self, block_index, freeze_color=(1, 1, 0, 1)):
        if not (0 <= block_index < len(self.blocks)):
            raise ValueError("Invalid block index: {}".format(block_index))

        # Proceed to freeze the block
        self.frozen_block_index = block_index
        self.is_block_frozen = True

        fblock = self.blocks[block_index]

        # Check if the block is already frozen to avoid redundant constraints
        if hasattr(fblock, 'constraint_id'):
            return fblock.constraint_id

        orientation_list = [fblock.orientation.x, fblock.orientation.y, fblock.orientation.z, fblock.orientation.w]
        fixed_constraint = self.client.createConstraint(
            parentBodyUniqueId=fblock.object_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=self.client.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=fblock.position,
            parentFrameOrientation=orientation_list
        )
        fblock.constraint_id = fixed_constraint
        self._state_info["frozen"] = block_index
        self.client.change_object_color(fblock.object_id, freeze_color)

        self.assembly.set_boundary_condition(block_index)

        return fixed_constraint


    def unfreeze_block(self):
        #unfreezes the currently frozen block
        if self.is_block_frozen:
            frozen_block = self.blocks[self.frozen_block_index]
            if hasattr(frozen_block, 'constraint_id'):
                self.client.removeConstraint(frozen_block.constraint_id)
                del frozen_block.constraint_id

            # remove all boundary conditions and add the boundary condition for the support block back
            self.assembly.unset_boundary_conditions()
            self.assembly.set_boundary_condition(-1)

        self.is_block_frozen = False
        self.frozen_block_index = None
        self._state_info["frozen_block"] = None

 