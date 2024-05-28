# import torch
# from typing import Any
# import random
# from collections import namedtuple


# from robotoddler.models.image_models import ConvNet
# from robotoddler.utils.utils import init_weights
# from robotoddler.utils.action_space import generate_actions, filter_collision_actions
# from robotoddler.utils.features import get_image_features, get_task_features
# from robotoddler.utils.replay_memory import ReplayBuffer



# Transition = namedtuple('Transition',
#                         ('state_features',
#                          'binary_features',
#                          'action',
#                          'action_features', 
#                          'reward',
#                          'next_state_features', 
#                          'next_binary_features',
#                          'next_available_actions',
#                          'next_actions_features',
#                          'task_features',
#                          'obstacle_features',
#                          'done'))

# def train():
#     pass

# def select_action(env, policy):
#     pass

# def rollout(env, policy, policy_net):
#     done = False
#     transitions = []

#     # environment reset
#     observation, info = env.reset()

#     # initial features and actions
#     task_features, obstacle_features = get_task_features(info)
#     available_actions = [*generate_actions(env)]
#     img_state_features, add_state_features, action_features = get_image_features(observation)
#     available_actions, action_features = filter_collision_actions(image_state_featers, tasks_features, available_actions, action_features)

#     while not done:
#         # action and environment step
#         action = policy(*policy_net(img_state_features, add_state_features, action_features, task_features))
#         next_observation, reward, done, info = env.step(action)

#         # get next features and actions
#         next_img_state_features, next_add_state_features, next_action_features = get_image_features(next_observation)
#         available_actions = [*generate_actions(env)]
#         next_available_actions, next_action_features = filter_collision_actions(next_img_state_features, task_features, available_actions, next_action_features)

#         # add transition
#         transitions.append(Transition(
#             state_features=img_state_features,
#             binary_features=add_state_features,
#             action_features=action_features,
#             task_features=task_features,
#             next_state_features=next_img_state_features,
#             next_binary_features=next_add_state_features,
#             next_actions_features=next_action_features,
#         ))

#         # update features and actions
#         img_state_features = next_img_state_features
#         add_state_features = next_add_state_features
#         action_features = next_action_features
#         available_actions = next_available_actions

#     return transitions

# class DDQ_CNN(Experiment):
    

#     def sample_environment(self):
#         pass


# def __main__():
    

#     exp = Experiment()
#     exp.add_parameter()


#     img_size = (64, 64)
#     exp.train

#     greedy = lambda q_values: torch.argmax(q_values)

#     if load_checkpoint:
#         # load checkpoint
#         replay_buffer.load_checkpoint()
#         policy_net.load_checkpoint()
#         target_net.load_checkpoint()
#         optimizer.load_checkpoint()
#         policy.load_checkpoint()
#     else:
#         policy_net = ConvNet(img_size=img_size)
#         target_net = ConvNet(img_size=img_size)

#         policy_net.apply(init_weights)
#         target_net.load_state_dict(policy_net.state_dict())

#         replay_buffer = ReplayBuffer(capacity=10000)
#         eps_greedy = EpsilonGreedy(eps_start=0.5, gamma=0.999)


#     for i in range(num_episodes):
#         transitions = rollout(env=env, policy=eps_greedy, policy_net=policy_net)

#         replay_buffer.push(transitions)
#         reward = sum([gamma ** t * t.reward for t in transitions])

#         losses = train(policy_net, target_net, transitions, optimizer, scheduler)

#         exp.log(
#             step=i, 
#             reward=reward, 
#             loss=torch.avg(losses)
#             context='training'
#         )

#         if i % 10:
#             # evaluate
#             transitions = rollout(env=env, policy=greedy, policy_net=policy_net)
#             reward = sum([gamma ** t * t.reward for t in transitions])

#             exp.log(
#                 step=i, 
#                 reward=reward, 
#                 context='evaluation'
#             )

#             # visualize transitions
#             fig = plot_transitions(transitions, policy_net)
#             exp.log_figure(fig, step=i, context='evaluation')
#             fig.close()

#         # save checkpoint
#         if i % 100 == 0:
#             policy_net.save_checkpoint()
#             target_net.save_checkpoint()
#             optimizer.save_checkpoint()
#             replay_buffer.save_checkpoint()
#             print("Checkpoint saved.")