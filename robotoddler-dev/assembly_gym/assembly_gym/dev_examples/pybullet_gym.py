# import pybullet as p
# import pybullet_envs
# from time import sleep
# import gym

# cid = p.connect(p.DIRECT)
# env = gym.make("CartPoleContinuousBulletEnv-v0")
# env.render()
# env.reset()

# for _ in range(10000):
#     sleep(1 / 60)
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)

# p.disconnect(cid)

import pybullet as p
from time import sleep
from pybullet_envs.bullet import CartPoleBulletEnv

cid = p.connect(p.DIRECT)
env = CartPoleBulletEnv(renders=True, discrete_actions=False)

env.render()
env.reset()

for _ in range(10000):
    sleep(1 / 60)
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

p.disconnect(cid)