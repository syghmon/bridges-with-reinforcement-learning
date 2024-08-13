from robotoddler.policy import *
from robotoddler.DDQ import *
from assembly_gym.envs.gym_env import *
import pickle
import matplotlib.pyplot as plt

num_targets = 1
x_min = 0.3
x_max = 0.7
z_min = 0.21
z_max = 0.24
setup = lambda: tower_setup(num_targets, x_min, x_max, z_min, z_max)

shapes, _, _ = setup()
receiving_faces = [[1]]
coming_faces = [[3]]
block_dim = 7
obstacle_dim = 3
target_dim = 3
hidden_dim = 64
num_rounds = 2
num_heads = 2
x_discr_ground = [round(val, 3) for val in list(np.linspace(0.2, 0.8, 13))]
y_discr_ground = [0]
x_block_offset = [-0.02, 0, 0.02]
y_block_offset = [0]

agent = Agent(shapes, receiving_faces, coming_faces, block_dim, obstacle_dim, target_dim, hidden_dim, num_rounds, num_heads, x_discr_ground, y_discr_ground, x_block_offset, y_block_offset)
with open('best_policy.pkl', 'rb') as file:
#with open('../robotoddler/Results/04-03/best_policy.pkl', 'rb') as file:
    best_policy_param = pickle.load(file)
agent.policy.load_state_dict(best_policy_param)

reward_fct = sparse_reward

env = AssemblyGym(setup=setup, 
                  reward_fct=reward_fct,
                  restrict_2d=True, 
                  assembly_env=AssemblyEnv(render=False))


success = np.zeros([41, 24])
for i, t_x in enumerate(np.linspace(0.3, 0.7, 41)):
    for j, t_z in enumerate(np.linspace(0.01, 0.24, 24)):
        r, s = run_episode(env, agent, eps=0, target=[float(t_x), 0, float(t_z)], allow_collisions=False)
        success[i,j] = s


plt.imshow(success.T, interpolation='nearest')
plt.gca().invert_yaxis()
plt.colorbar(ticks=[0, 1], label='Values')
plt.savefig("Success_colormap.png")