
import numpy as np
import time

from assembly_gym.utils.geometry import quaternion_distance, maximum_tension
from compas.geometry import Frame, Translation, Rotation, Quaternion, Box, Scale
from compas_cra.equilibrium.rbe_pyomo import rbe_solve
from compas_cra.equilibrium.cra_pyomo import cra_solve
from compas_cra.equilibrium.cra_penalty_pyomo import cra_penalty_solve
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning


def is_stable_pybullet(assembly_env, tol_distance=1e-2, tol_angle=1e-2, steps=3000, return_states=True):
    initial_states = []
    final_states = []
    
    # Store initial states for all blocks
    for block in assembly_env.blocks:
        position, orientation = assembly_env.client.getBasePositionAndOrientation(block.object_id)
        initial_states.append((position, Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])))
    
    # Pybullet simulation
    assembly_env.simulate(steps=steps)
    is_stable = True
    # Compare initial and final states for all blocks
    for index, block in enumerate(assembly_env.blocks):
        new_position, new_orientation = assembly_env.client.getBasePositionAndOrientation(block.object_id)
        final_states.append((new_position, Quaternion(x=new_orientation[0], y=new_orientation[1], z=new_orientation[2], w=new_orientation[3])))
        initial_position, initial_orientation = initial_states[index]

        dist = np.linalg.norm(np.array(initial_position) - np.array(new_position))
        new_orientation = Quaternion(x=new_orientation[0], y=new_orientation[1], z=new_orientation[2], w=new_orientation[3])
        angle = quaternion_distance(initial_orientation, new_orientation) # gives Nan on non-zero angles...

        collision_obs = False
        for obs in assembly_env.obstacles:
            contact_points = assembly_env.client.getContactPoints(obs.object_id, block.object_id)
            if len(contact_points) > 0:
                #print("Collision with block")
                collision_obs = True
            
        # Update is_stable based on the thresholds
        is_stable = is_stable and bool(dist < tol_distance) and bool(angle < tol_angle) and (not collision_obs)
        
    assembly_env.restore()
    return is_stable, (dict(initial_states=initial_states, final_states=final_states) if return_states else None)


def is_stable_rbe(assembly_env):
    # print("calling rbe")
    t0 = time.time()
    # the solver fails if there are no edges, so we are handling this case here
    if assembly_env.cra_assembly.graph.number_of_edges() == 0:
        free_nodes = [node for node in assembly_env.cra_assembly.graph.node.values() if not node.get('is_support')]
        # no edges and more than 1 free node means the structure is unstable
        return len(free_nodes) == 0, None
        

    res, res_dict = True, None
    try:
        with SuppressInfeasibleWarning():
            rbe_solve(assembly_env.cra_assembly, mu=assembly_env.mu, density=assembly_env.density, penalty=False)
    except ValueError as e:
        if e.args[0] == "infeasible":
            res, res_dict = False, None
        
        else:
            res, res_dict = None, dict(error=str(e))
        
    # print(f"Took {time.time() - t0:.2f} seconds ({res}, {res_dict})")
    return res, res_dict



def is_stable_rbe_penalty(assembly_env, tol=1e-3):
    # the solver fails if there are no edges, so we are handling this case here
    if assembly_env.cra_assembly.graph.number_of_edges() == 0:
        free_nodes = [node for node in assembly_env.cra_assembly.graph.node.values() if not node.get('is_support')]
        # no edges and more than 1 free node means the structure is unstable
        return len(free_nodes) == 0, None
    
    try:
        rbe_solve(assembly_env.cra_assembly, density=assembly_env.density, mu=assembly_env.mu, penalty=True)
    except ValueError as e:
        return None, dict(error=str(e))
    max_tension = maximum_tension(assembly_env.cra_assembly)
    return max_tension <= tol, {'max_tension': max_tension}


def is_stable_cra(assembly_env):
    # the solver fails if there are no edges, so we are handling this case here
    if assembly_env.cra_assembly.graph.number_of_edges() == 0:
        free_nodes = [node for node in assembly_env.cra_assembly.graph.node.values() if not node.get('is_support')]
        # no edges and more than 1 free node means the structure is unstable
        return len(free_nodes) == 0, None
    
    try:
        cra_solve(assembly_env.cra_assembly, mu=assembly_env.mu, density=assembly_env.density)
    except ValueError as e:
        if e.args[0] == "infeasible":
            return False, None
        else:
            return None, dict(error=str(e))
        
    return True, None

def is_stable_cra_penalty(assembly_env, tol=1e-3):
    # the solver fails if there are no edges, so we are handling this case here
    if assembly_env.cra_assembly.graph.number_of_edges() == 0:
        free_nodes = [node for node in assembly_env.cra_assembly.graph.node.values() if not node.get('is_support')]
        # no edges and more than 1 free node means the structure is unstable
        return len(free_nodes) == 0, None
    
    try:
        cra_penalty_solve(assembly_env.cra_assembly, mu=assembly_env.mu, density=assembly_env.density)
    except ValueError as e:
        return None, dict(error=str(e))
    max_tension = maximum_tension(assembly_env.cra_assembly)
    return max_tension <= tol, {'max_tension': max_tension}


def is_action_stable_rbe(gym_env, action):
    block = gym_env.create_block(action)
    gym_env.assembly_env.blocks.append(block)
    gym_env.assembly_env._reset_cra_assembly()
    
    stable, _ = is_stable_rbe(gym_env.assembly_env)
    gym_env.assembly_env.blocks.pop() # reset the environment
    
    return stable
 