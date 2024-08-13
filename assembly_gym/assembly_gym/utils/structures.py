from assembly_gym.envs.assembly_env import AssemblyEnv, Shape
from assembly_gym.envs.gym_env import Action, AssemblyGym, sparse_reward
import matplotlib.pyplot as plt
import numpy as np
import os


def create_env(mu, density, shapes, freeze_last=False, cra_env=True, pybullet_env=True):
    env = AssemblyGym(shapes=shapes, targets=[], obstacles=[],
                      reward_fct=sparse_reward,
                      restrict_2d=True,
                      assembly_env=AssemblyEnv(render=False, 
                                             mu=mu,
                                             density=density,
                                             stability=None,
                                             cra_env=cra_env,
                                             pybullet_env=pybullet_env))
    env.reset()
    return env


def hexagon(mu=0.8, density=1):
    env = create_env(mu=mu, density=density, shapes=[Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")])

    actions = [
        (Action(target_block=-1, target_face=0, shape=0, face=0, offset_x=0, offset_y=0), True),
        (Action(target_block=0, target_face=3, shape=0, face=3, offset_x=0., offset_y=0), mu > 1.732),
    ]
    
    return env, actions


def trapezoid_bridge(mu=0.8, density=1., freeze_last=True):
    env = create_env(mu=mu, density=density, shapes=[Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")])

    actions = [
        (Action(target_block=-1, target_face=0, shape=0, face=0, offset_x=-3, offset_y=0, frozen=freeze_last), True),
        (Action(target_block=0, target_face=3, shape=0, face=3, offset_x=0., offset_y=0, frozen=freeze_last), freeze_last or mu > 1.732),
        (Action(target_block=1, target_face=1, shape=0, face=1, offset_x=0, offset_y=0, frozen=freeze_last), freeze_last and mu > 0.5),
        (Action(target_block=2, target_face=3, shape=0, face=3, offset_x=0, offset_y=0, frozen=freeze_last), freeze_last and mu > 0.5),
        (Action(target_block=3, target_face=1, shape=0, face=2, offset_x=0, offset_y=0, frozen=freeze_last), freeze_last and mu > 0.5),
        (Action(target_block=4, target_face=0, shape=0, face=1, offset_x=0, offset_y=0, frozen=freeze_last), freeze_last and mu > 0.5),
        (Action(target_block=5, target_face=3, shape=0, face=3, offset_x=0, offset_y=0, frozen=freeze_last), freeze_last and mu > 0.5),
        (Action(target_block=6, target_face=1, shape=0, face=1, offset_x=0, offset_y=0, frozen=freeze_last), freeze_last and mu > 0.5),
        (Action(target_block=7, target_face=3, shape=0, face=3, offset_x=0, offset_y=0), mu > 0.5)
    ]

    return env, actions

def hexagon_bridge_3(mu=0.8, density=1., freeze_last=True):
    env = create_env(mu=mu, density=density, shapes=[Shape(urdf_file='shapes/hexagon.urdf', name="hexagon")])

    actions = [
        (Action(target_block=-1, target_face=0, shape=0, face=0, offset_x=-3, offset_y=0, frozen=freeze_last), True),
        (Action(target_block=0, target_face=5, shape=0, face=0, offset_x=0., offset_y=0, frozen=freeze_last), freeze_last),
        (Action(target_block=1, target_face=5, shape=0, face=0, offset_x=0., offset_y=0, frozen=False), freeze_last),
    ]

    return env, actions

def hexagon_bridge_5(mu=0.8, density=1., freeze_last=True):
    env = create_env(mu=mu, density=density, shapes=[Shape(urdf_file='shapes/hexagon.urdf', name="hexagon")])

    actions = [
        (Action(target_block=-1, target_face=0, shape=0, face=0, offset_x=-3, offset_y=0, frozen=freeze_last), True),
        (Action(target_block=0, target_face=5, shape=0, face=0, offset_x=0., offset_y=0, frozen=freeze_last), freeze_last),
        (Action(target_block=1, target_face=4, shape=0, face=0, offset_x=0., offset_y=0, frozen=freeze_last), freeze_last),
        (Action(target_block=2, target_face=5, shape=0, face=0, offset_x=0., offset_y=0, frozen=freeze_last), freeze_last),
        (Action(target_block=3, target_face=4, shape=0, face=0, offset_x=0., offset_y=0, frozen=False), freeze_last)
    ]
    return env, actions


def horizontal_bridge(mu=0.8, density=1., freeze_last=True):
    shapes = [Shape(urdf_file='shapes/trapezoid.urdf', name="trapezoid")]
 
    # obstacles = [(i * square_size, 0, square_size / 2) for i in range(1, num_obstacles + 1)]
    env = create_env(mu=mu, density=density, shapes=shapes)

    actions = [
        (Action(target_block=-1, target_face=0, shape=0, face=2, offset_x=-0.9, offset_y=0, frozen=freeze_last), True),
        (Action(target_block=0, target_face=0, shape=0, face=2, offset_x=0, offset_y=0, frozen=freeze_last), freeze_last),
        (Action(target_block=1, target_face=0, shape=0, face=2, offset_x=0, offset_y=0, frozen=False), True)
    ]

    return env, actions


def tower(num_blocks=3, mu=0.8, density=1):
    cube = Shape(urdf_file='shapes/cube.urdf', name="cube")
    env = create_env(mu, density, [cube])

    # Tower
    actions = []
    for i in range(num_blocks):
        actions.append((Action(target_block=i -1, target_face=0, shape=0, face=3, offset_x=0, offset_y=0), True))

    return env, actions



def levitating_block(mu=0.8, density=1, freeze_last=False, offset_y=0.5):
    cube = Shape(urdf_file='shapes/cube.urdf', name="cube")
    env = create_env(mu, density, shapes=[cube])
    actions = [
        (Action(target_block=-1, target_face=0, shape=0, face=0, offset_x=0, offset_y=offset_y, frozen=freeze_last), freeze_last or offset_y < 1e-4),
        (Action(target_block=0, target_face=3, shape=0, face=0, offset_x=0, offset_y=0, frozen=freeze_last), offset_y < 1e-4)]
    return env, actions