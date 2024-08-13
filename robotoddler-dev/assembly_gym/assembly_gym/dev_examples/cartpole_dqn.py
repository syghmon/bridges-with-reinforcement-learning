# encoding:utf-8
# train.py
from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines.deepq import DQN
from time import sleep
import pybullet as p

import sys

# def callback(*params):
#     print(params[0])
#     print("-" * 20)
#     print(params[1])
#     sys.exit(-1)

def callback(*params):
    info_dict = params[0]
    episode_rewards = info_dict['episode_rewards']
    print(f"episode total reward: {sum(episode_rewards)}")

env = CartPoleBulletEnv(renders=False, discrete_actions=True)

model = DQN(policy="MlpPolicy", env=env)

print("开始训练，稍等片刻")
model.learn(total_timesteps=10000, callback=callback)
model.save("./model")