import math
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

from assembly_gym.envs.assembly_env import AssemblyEnv, Shape, Block
from assembly_gym.envs.gym_env import AssemblyGym, sparse_reward, tower_setup, bridge_setup, hard_tower_setup

from assembly_gym.utils import align_frames_2d
from assembly_gym.utils.rendering import plot_assembly_env, render_assembly_env

from robotoddler.DDQ import *
from robotoddler.policy import *

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

batch_size = 64
num_updates = 10
#target_update_freq = 200 # replaced by target_tau
target_tau = 0.01
threshold_lr = 100
lr_schedule = lambda i_episode: 1e-4 # 1e-3 * (i_episode < threshold_lr) + 1e-4 * (i_episode >= threshold_lr)
gamma = 0.99
intial_policy = 'best_policy.pkl'
intial_target_policy = 'best_target_policy.pkl'
num_episodes = 1001
test_freq = 500
memory_file = 'memory.pkl'
memory_size = 30000
alpha = 0.6
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 300
eps_schedule = lambda t: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)

block_dim = 7
obstacle_dim = 3
target_dim = 3

num_layers = 2
num_heads = 2
hidden_dim = 64

def tower_setup(num_targets = 3, x_min = 0.2, x_max = 0.8, z_min = 0., z_max = 0.2):
    rectangle = Shape(urdf_file='../assembly_gym/shapes/block.urdf', name="rectangle")
    #rectangle = Shape(urdf_file=os.path.join(DATA_PATH,'block.urdf'), name="rectangle")
    shapes = [rectangle]
    # We can only place the cubes on top of each other:
    receiving_faces = [[3]]
    coming_faces = [[0]]
    targets = [(np.random.uniform(x_min, x_max), 0, np.random.uniform(z_min, z_max)) for _ in range(num_targets)]
    obstacles = []
    return shapes, receiving_faces, coming_faces, obstacles, targets

def mysetup():
    rectangle = Shape(urdf_file='../assembly_gym/shapes/block.urdf', name="rectangle")
    shapes = [rectangle]
    receiving_faces = [[3]] # [[0,1,2,3]]
    coming_faces = [[0]] #[[0,1]] # either horizontal or vertical
    x_target = np.random.uniform(0.45, 0.55)
    stories = random.randint(2,4)
    targets = [[x_target, 0, 0.05*(stories-1) + np.random.uniform(0.0, 0.05)]]
    #targets = [[x_target, 0, np.random.uniform(0.05, 0.1)]]
    #obstacles = [[x_target + np.random.uniform(-0.05, 0.05), 0, 0.025]]
    obstacles = [[0.5, 0, 0.025]]
    
    return shapes, receiving_faces, coming_faces, obstacles, targets

def hard_setup():
    rectangle = Shape(urdf_file='../assembly_gym/shapes/block.urdf', name="rectangle")
    shapes = [rectangle]
    receiving_faces = [[3]] # [[0,1,2,3]]
    coming_faces = [[0]] #[[0,1]] # either horizontal or vertical
    targets = [[0.5 + np.random.uniform(-0.02, 0.02), 0, 0.075], [0.5 + np.random.uniform(-0.02, 0.02), 0, 0.175]]
    obstacles = [[0.5, 0, 0.025], [0.5, 0, 0.125]]
    #obstacles = [[x_target + np.random.uniform(-0.03, 0.03), 0, 0.025]]
    
    return shapes, receiving_faces, coming_faces, obstacles, targets

def connecting_setup(): # Similar as the connecting setup in Deepmind's paper
    rectangle = Shape(urdf_file='../assembly_gym/shapes/block.urdf', name="rectangle")
    cube = Shape(urdf_file='../assembly_gym/shapes/cube.urdf', name="cube")
    #shapes = [rectangle, cube]
    shapes = [cube]
    receiving_faces = [[3]] # [[3], [3]] # [[0,1,2,3]]
    coming_faces = [[1]] # [[0], [1]] #[[0,1]] # either horizontal or vertical
    #targets = [[np.random.uniform(0.4, 0.6), 0, 0.175], [np.random.uniform(0.4, 0.6), 0, 0.175], [np.random.uniform(0.4, 0.6), 0, 0.175]]
    #obstacles = [[np.random.uniform(0.4, 0.47), 0, np.random.uniform(0.025, 0.125)], [np.random.uniform(0.53, 0.6), 0, np.random.uniform(0.025, 0.125)]]
    targets = [[np.random.uniform(0.4, 0.6), 0, 0.125], [np.random.uniform(0.4, 0.6), 0, 0.125], [np.random.uniform(0.4, 0.6), 0, 0.125]]
    obstacles = [[np.random.uniform(0.4, 0.45), 0, np.random.uniform(0.025, 0.075)], [np.random.uniform(0.55, 0.6), 0, np.random.uniform(0.025, 0.075)]]

    #obstacles = [[x_target + np.random.uniform(-0.03, 0.03), 0, 0.025]]
    
    return shapes, receiving_faces, coming_faces, obstacles, targets

#num_targets = 1
#x_min = 0.3
#x_max = 0.7
#z_min = 0.1
#z_max = 0.24
#setup = lambda: tower_setup(num_targets, x_min, x_max, z_min, z_max)
setup = connecting_setup
reward_fct = sparse_reward
max_blocks = 9

env = AssemblyGym(setup=setup, 
                  reward_fct=reward_fct,
                  restrict_2d=True, 
                  assembly_env=AssemblyEnv(render=False),
                  max_blocks = max_blocks)

shapes, receiving_faces, coming_faces, _, _ = setup()

#x_discr_ground = [round(val, 3) for val in list(np.linspace(0.3, 0.7, 9))]
x_discr_ground = [round(val, 3) for val in list(np.linspace(0.35, 0.65, 13))]
y_discr_ground = [0]
#x_block_offset = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04]
x_block_offset = [-0.02, -0.01, 0, 0.01, 0.02]
y_block_offset = [0]
z_max = 0.2

agent = Agent(shapes, receiving_faces, coming_faces, block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, x_discr_ground, y_discr_ground, x_block_offset, y_block_offset, z_max)
target_agent = Agent(shapes, receiving_faces, coming_faces, block_dim, obstacle_dim, target_dim, hidden_dim, num_layers, num_heads, x_discr_ground, y_discr_ground, x_block_offset, y_block_offset, z_max)

# Load an initial policy if wanted
if len(intial_policy) > 0:
    with open(intial_policy, 'rb') as file:
        best_policy_param = pickle.load(file)
    agent.policy.load_state_dict(best_policy_param)

target_agent.policy.load_state_dict(agent.policy.state_dict())

HER = True
PER = True
if PER:
    memory = PrioritizedReplayBuffer(memory_size, alpha)
else:
    memory = ReplayMemory(memory_size)

if len(memory_file) > 0:
    with open(memory_file, 'rb') as file:
        memory_dict = pickle.load(file)
    memory.memory = memory_dict["memory"]
    memory.episode = memory_dict["episode"]
    memory.priorities = memory_dict["priorities"]
    


optimizer = torch.optim.AdamW(agent.policy.parameters(), lr=lr_schedule(1), amsgrad=False)



policy_net, target_net, best_policy_param, test_success, test_rewards, episode_rewards, episode_values, Q_losses, memory, gradient_mags, max_weights = \
train(env, agent, target_agent, num_episodes, num_updates, optimizer, batch_size, gamma, eps_schedule, \
    lr_schedule, target_tau, test_freq, memory, HER)

dir = ''

with open(dir+'best_policy.pkl', 'wb') as file:
    pickle.dump(policy_net.state_dict(), file)

with open(dir+'best_target_policy.pkl', 'wb') as file:
    pickle.dump(target_net.state_dict(), file)


if type(memory) is PrioritizedReplayBuffer:
    memory_dict = {"episode":memory.episode, "memory":memory.memory, "priorities":memory.priorities}
else:
    memory_dict = {"episode":memory.episode, "memory":memory.memory}
with open(dir+'memory.pkl', 'wb') as file:
    pickle.dump(memory_dict, file)



logs = {"rewards":episode_rewards, "value":episode_values, "gradient_mags":gradient_mags, "max_weights":max_weights}
with open(dir+'logs.pkl', 'wb') as file:
    pickle.dump(logs, file)

plt.figure()
plt.plot(test_success)
plt.ylabel("Success probability")
plt.savefig(dir+'Test_success.png')

plt.figure()
plt.plot(episode_rewards)
plt.plot(np.convolve(episode_rewards, np.ones(200), 'valid') / 200, 'r')
plt.ylabel("Episodic reward")
plt.savefig(dir+'Episodic_reward.png')

plt.figure()
plt.plot(episode_rewards)
plt.plot(episode_values)
plt.plot(np.convolve(episode_rewards, np.ones(200), 'valid') / 200, 'r')
plt.plot(np.convolve(episode_values, np.ones(200), 'valid') / 200, 'r')
plt.ylabel("Episodic value")
plt.savefig(dir+'Episodic_value.png')

plt.figure()
plt.plot(Q_losses, '+')
plt.plot(np.convolve(Q_losses, np.ones(200), 'valid') / 200, 'r')
plt.ylabel("DDQ loss")
plt.savefig(dir+'DDQ_loss.png')

