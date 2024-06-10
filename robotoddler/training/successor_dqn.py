from collections import deque, namedtuple
import gc
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import argparse
import os
import aim

from assembly_gym.envs.gym_env import AssemblyGym, sparse_reward, tower_setup, hard_tower_setup, bridge_setup
from assembly_gym.envs.assembly_env import AssemblyEnv, Block, Shape
from assembly_gym.utils.rendering import get_rgb_array, render_assembly_env, render_blocks_2d

from robotoddler.utils.replay_memory import ReplayBuffer
from robotoddler.utils.utils import init_weights, parse_img_size, load_checkpoint, save_checkpoint, optimizer_to, convolve_with_gaussian
from robotoddler.models.cv import ConvNet, SuccessorMLP, UNet, Policy
from robotoddler.utils.actions import generate_actions, filter_actions

import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('block_features',
                         'binary_features',
                         'action',
                         'action_features', 
                         'reward',
                         'lin_reward',
                         'done',
                         'reward_features',
                         'obstacle_features',
                         'next_block_features', 
                         'next_binary_features',
                         'next_available_actions',
                         'next_actions_features',
                         'next_reward_features',  # duplicated for convenience
                         'next_obstacle_features',  # duplicated for convenience
                         ))


def get_state_features(observation, xlim=(0, 1), ylim=(0, 1), img_size=(512,512), device=None):
    """
    Get the state features for the observation.
    Returns the image state features and additional binary features.
    """
    binary_features = np.array([
        observation['stable'],
        observation['collision'], # ToDo obstacle vs block collision
        observation['collision_block'],
        observation['collision_obstacle'],
        observation['collision_floor'],
        observation['collision_boundary'],
    ])
    #if not (np.array(binary_features) == np.array([1,0,0,0,0,0])).all(): # unstable or collision
    #    image_features = None
    image_features =  torch.Tensor(render_blocks_2d(observation['blocks'], xlim=xlim, ylim=ylim, img_size=img_size)).unsqueeze(0).to(device)
    return image_features, torch.Tensor(binary_features).to(device)


def get_task_features(obs, xlim=(0, 1), ylim=(0, 1), img_size=(512,512), device=None):
    """
    Get the task features for the environment.
    Returns the image features for the obstacles and the targets.
    """
    # reward features
    cube = Shape(urdf_file='shapes/cube.urdf')
    target_blocks = [Block(shape=cube, position=target) for target in obs['targets']]
    reward_features = render_blocks_2d(target_blocks, xlim=xlim, ylim=ylim, img_size=img_size).astype(np.float32)
    reward_features /= np.sum(reward_features)
    kernel_size = 31 # Needs to be odd !!!
    sigma = 4
    reward_features = convolve_with_gaussian(torch.Tensor(reward_features), kernel_size, sigma)
    # obstacle features
    obstacle_features = render_blocks_2d(obs['obstacle_blocks'], xlim=xlim, ylim=ylim, img_size=img_size)
    return reward_features.unsqueeze(0).to(device), torch.Tensor(obstacle_features).unsqueeze(0).to(device)


def get_action_features(env, actions, xlim=(0,1), ylim=(0, 1), img_size=(512,512), device=None):
    """
    Get the action feature images.
    """
    blocks = [env.create_block(action) for action in actions]
    return torch.Tensor(np.array([render_blocks_2d([block], xlim=xlim, ylim=ylim, img_size=img_size) for block in blocks])).unsqueeze(1).to(device)


class EpsilonGreedy:
    def __init__(self, eps_start=0.5, eps_end=0.05, gamma=0.99, episode=0):
        self.epsilon = (eps_start - eps_end) * (gamma ** episode) + eps_end
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.gamma = gamma

    def step(self):
        self.epsilon = (self.epsilon - self.eps_end) * self.gamma + self.eps_end
        return self

    def __call__(self, q_values, *args, **kwargs):
        if random.random() > self.epsilon:
            return torch.argmax(q_values).item()
        return random.randint(0, len(q_values)-1)


def train_policy_net(policy_net, target_net, optimizer, replay_buffer, gamma, loss_fct='mse_q_values', scheduler=None, n_steps=10, batch_size=16, verbose=False, device='cuda'):
    """
    Train the policy net.
    """
    if len(replay_buffer) < batch_size:
        return

    loss_fct = loss_fct.split('+')

    policy_net.train()
    target_net.eval()

    losses = []
    it = tqdm(range(n_steps), disable=not verbose)
    for i in it:
        # prepare batch
        transitions, batch = replay_buffer.sample(batch_size=batch_size, stack_tensors=True, device=device)

        # policy net prediction for current state
        q_values, succ_block_features, succ_binary_features = policy_net(
            batch.block_features, 
            batch.binary_features, 
            batch.action_features, 
            batch.reward_features, 
            batch.obstacle_features
        )
        
        with torch.no_grad():
            # We compute all next states and actions in a single forward pass
            # Note: this may cause memory issues if there are lot's of actions and large batch size

            #TODO: Avoid forward computation for done states
            with torch.no_grad():
                next_q_values, next_succ_block_features, next_succ_binary_features = target_net(
                    batch.next_block_features, 
                    batch.next_binary_features, 
                    batch.next_actions_features, 
                    batch.next_reward_features, 
                    batch.next_obstacle_features
                )
            # compute argmax actions
            num_actions = [len(actions) for actions in batch.next_available_actions]
            offsets = np.cumsum([0] + num_actions)
            selected_actions = [chunk.argmax().item() + offset for chunk, offset in zip(next_q_values.split(num_actions), offsets)]

            # reduce next state information to selected actions
            next_q_values = next_q_values[selected_actions]
            next_succ_binary_features = next_succ_binary_features[selected_actions][:,0] #.softmax(dim=1)[:,1]
            if succ_block_features is not None:
                next_succ_block_features = next_succ_block_features[selected_actions][:,0] #.softmax(dim=1)[:,1]

            # handle done states
            done_mask = torch.tensor(batch.done, dtype=bool)
            next_q_values[done_mask] = 0
            next_succ_binary_features[done_mask] = 0 # batch.next_binary_features[selected_actions][done_mask].squeeze()
            if succ_block_features is not None:
                next_succ_block_features[done_mask] = 0 # batch.next_block_features[selected_actions][done_mask].squeeze()


        # define loss functions
        mse = torch.nn.MSELoss()
        loss = 0.
        
        # reward loss
        if 'mse_q_values' in loss_fct:
            loss += mse(q_values, batch.lin_reward + gamma * next_q_values)
            

        # state feature mse loss
        if 'mse_block_features' in loss_fct:
            if succ_block_features is None:
                raise ValueError("No successor block features available from the chosen policy net.")
            # state_target = (1-gamma) * batch.block_features.squeeze(1) + gamma * next_succ_block_features
            state_target = batch.action_features.squeeze(1) + gamma * next_succ_block_features
            # state_target = torch.stack([1 - state_target, state_target], dim=1)
            #plt.figure()
            #plt.imshow(state_target[0])
            loss += mse(succ_block_features[:,0], state_target)
            #loss += mse(succ_block_features.softmax(dim=1)[:,1], state_target)

            loss += F.binary_cross_entropy(succ_binary_features[:,0], batch.next_binary_features[selected_actions,0]) # only assesses immediate stability (gamma=0) 
        
        # cross_entropy = torch.nn.CrossEntropyLoss()
        # state feature cross-entropy loss
        # state_target = (1-gamma) * torch.stack(batch.state_features).squeeze(1) + gamma * next_state_successor_features
        # state_target = torch.stack([1 - state_target, state_target], dim=1)
        # loss = cross_entropy(successor_state_features, state_target) 

        # additional features loss
        # add_features_target = (1-gamma) * torch.stack(batch.binary_features) + gamma * next_state_binary_features
        # loss += cross_entropy(successor_binary_features, torch.stack([1 - add_features_target, add_features_target], dim=1)) 

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        it.set_postfix(loss=loss.item())
        losses.append(loss.item())

    return losses


def update_target_net(policy_net, target_net, tau=0.01):
    """
    Update the target net using the policy net with a soft update.
    """
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key]* (1 - tau)
    target_net.load_state_dict(target_net_state_dict)


def rollout_episode(env, policy, policy_net, x_discr_ground, setup_fct, offset_values=[0.], img_size=(64, 64), xlim=(0,1), ylim=(0,1), log_images=False, device=None):
    """
    Roll out a single episode using the policy and the policy network.
    """
    done = False
    transitions = []
    images = [] if log_images else None
    policy_net.eval()

    # environment reset
    obs, info = env.reset(**setup_fct())

    # initial features and actions
    reward_features, obstacle_features = get_task_features(obs, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    block_features, binary_features = get_state_features(obs, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    available_actions = [*generate_actions(env, x_discr_ground=x_discr_ground, offset_values=offset_values)]
    action_features = get_action_features(env, available_actions, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    available_actions, action_features = filter_actions(available_actions, action_features, block_features, obstacle_features)
    num_actions = len(available_actions)


    print("block_features:", block_features.shape)
    print("binary_features:", binary_features.shape)
    print("action_features:", action_features.shape)
    print("reward_features:", reward_features.shape)
    print("obstacle_features:", obstacle_features.shape)

    while not done:
        # action and environment step
        with torch.no_grad():
            q_values, succ_block_features, succ_binary_features = policy_net(block_features.expand(num_actions, -1, -1, -1),
                                                   binary_features.expand(num_actions, -1),
                                                   action_features, 
                                                   reward_features.expand(num_actions, -1, -1, -1),
                                                   obstacle_features.expand(num_actions, -1, -1, -1))
        selected_action_index = policy(q_values, succ_block_features, succ_binary_features)
        action = available_actions[selected_action_index]
        selected_action_features = action_features[selected_action_index]
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # note: we compute our own (linear) reward function here
        lin_reward = torch.sum(selected_action_features * reward_features)

        # get next features and actions
        next_block_features, next_binary_features = get_state_features(next_observation, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
        next_available_actions = [*generate_actions(env, x_discr_ground=x_discr_ground, offset_values=offset_values)]
        next_action_features = get_action_features(env, next_available_actions, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
        next_available_actions, next_action_features = filter_actions(next_available_actions, next_action_features, next_block_features, obstacle_features)
        num_actions = len(next_available_actions)
        
        # add transition
        transitions.append(Transition(
            block_features=block_features.unsqueeze(0),  # add batch dimension
            binary_features=binary_features.unsqueeze(0),
            action_features=action_features[selected_action_index].unsqueeze(0),
            reward_features=reward_features.unsqueeze(0),
            obstacle_features=obstacle_features.unsqueeze(0),
            action=action,
            lin_reward=lin_reward.unsqueeze(0),
            reward=torch.Tensor([reward]),
            done=done,
            next_block_features=next_block_features.expand(num_actions, -1, -1, -1),
            next_binary_features=next_binary_features.expand(num_actions, -1),
            next_actions_features=next_action_features,
            next_reward_features=reward_features.expand(num_actions, -1, -1, -1),  # duplicated for convenience
            next_obstacle_features=obstacle_features.expand(num_actions, -1, -1, -1),  # duplicated for convenience
            next_available_actions=next_available_actions,
        ))

        # update features and actions
        block_features = next_block_features
        binary_features = next_binary_features
        action_features = next_action_features
        available_actions = next_available_actions

        # record additional information
        if log_images:
            img_dict = dict()
            if succ_block_features is None:
                raise ValueError("No successor block features available from the chosen policy net. Disable image logging or use a different model.")
            img_dict['succ_block_features'] = succ_block_features[selected_action_index][0].cpu().numpy()
            env.assembly_env.simulate(steps=240)
            img_dict['env_render'] = get_rgb_array(near=0.001, fov=80, far=10, target=[0.5, 0, 0.1], pitch=-20)
            env.assembly_env.restore()
            images.append(img_dict)

    return transitions, images






def log_episode(episode, transitions, losses, gamma, context='training', policy=None, images=None, log_images=False, wandb_run=None, aim_run=None, verbose=False):
    """
    Log the episode information.
    """

    reward = sum([gamma ** i * t.reward for i,t in enumerate(transitions)]).item()
    lin_reward = sum([gamma ** i * t.lin_reward for i,t in enumerate(transitions)]).item()
    avg_loss = sum(losses)/len(losses) if losses else None
    num_steps = len(transitions)
    stable = transitions[-1].next_binary_features[0, 0].item()
    collision = transitions[-1].next_binary_features[0, 1].item()
    fig = None

    log_info = {
        'reward': reward, 
        'lin_reward' : lin_reward, 
        'avg_loss': avg_loss, 
        'num_steps': num_steps, 
        'stable': stable, 
        'collision': collision
    }

    if policy is not None:
        log_info['epsilon'] = policy.epsilon

    if log_images:
        num_transitions = len(transitions)
        num_plots = 6
        fig, axes = plt.subplots(num_transitions, num_plots, figsize=(2 * num_plots, 2 * num_transitions))
        
        # Ensure axes is a 2D array
        if num_transitions == 1:
            axes = np.expand_dims(axes, 0)
        elif num_plots == 1:
            axes = np.expand_dims(axes, -1)
        
        print(f"Figure and Axes structure: fig type: {type(fig)}, axes type: {type(axes)}")
        if isinstance(axes, np.ndarray):
            print(f"Axes shape: {axes.shape}")
            print(f"First axes element type: {type(axes[0])}")
            if isinstance(axes[0], np.ndarray):
                print(f"First axes element shape: {axes[0].shape}")

        for i, t in enumerate(transitions):
            if verbose:
                print(f"Transition {i}, reward: {t.reward[0]}")
                print(f"Block features shape: {t.block_features.shape}")
                print(f"Action features shape: {t.action_features.shape}")
                print(f"Reward features shape: {t.reward_features.shape}")
                print(f"Obstacle features shape: {t.obstacle_features.shape}")
                print(f"Block features data: {t.block_features}")
                print(f"Action features data: {t.action_features}")
                print(f"Reward features data: {t.reward_features}")
                print(f"Obstacle features data: {t.obstacle_features}")
            
            # Ensure the features are 2D
            block_features = t.block_features.squeeze().cpu().numpy()
            if block_features.ndim == 1:
                block_features = block_features.reshape(1, -1)
            
            action_features = t.action_features.squeeze().cpu().numpy()
            if action_features.ndim == 1:
                action_features = action_features.reshape(1, -1)
            
            reward_features = t.reward_features.squeeze().cpu().numpy()
            if reward_features.ndim == 1:
                reward_features = reward_features.reshape(1, -1)
            
            obstacle_features = t.obstacle_features.squeeze().cpu().numpy()
            if obstacle_features.ndim == 1:
                obstacle_features = obstacle_features.reshape(1, -1)
            
            # Debug prints immediately before imshow
            print(f"Plotting transition {i}, block features shape: {block_features.shape}")
            print(f"Plotting transition {i}, action features shape: {action_features.shape}")
            print(f"Plotting transition {i}, reward features shape: {reward_features.shape}")
            print(f"Plotting transition {i}, obstacle features shape: {obstacle_features.shape}")
            try:
                axes[i, 0].imshow(block_features, cmap='gray', vmin=0, vmax=1)
                axes[i, 1].imshow(block_features + action_features, cmap='gray', vmin=0, vmax=1)
                axes[i, 2].imshow(reward_features, cmap='gray')
                axes[i, 3].imshow(obstacle_features, cmap='gray', vmin=0, vmax=1)
                if images:
                    axes[i, 4].imshow(images[i]['succ_block_features'], vmin=0, vmax=1, cmap='gray')
                    axes[i, 5].imshow(images[i]['env_render'])
            except IndexError as e:
                print(f"IndexError occurred at transition {i}: {e}")
                print(f"axes shape: {axes.shape} | accessing axes[{i}, 0]")

        axes[0, 0].set_title('Block Features')
        axes[0, 1].set_title('Next state Features')
        axes[0, 2].set_title('Reward Features')
        axes[0, 3].set_title('Obstacle Features')
        if images:
            axes[0, 4].set_title('Successor Features')
            axes[0, 5].set_title('Environment Render')
        
        for ax in axes.flatten():
            ax.axis('off')

        fig.subplots_adjust()
        fig.tight_layout()
        plt.show()

    # aim tracking
    if aim_run is not None:
        for k, v in log_info.items():
            if v is not None:
                aim_run.track(v, name=k, step=episode, context=dict(context=context))
        if log_images:
            aim_figure = aim.Image(fig)
            aim_run.track(aim_figure, name=f"Images", step=episode, context=dict(context=context))  
        
    if wandb_run is not None:
        raise NotImplementedError("Wandb logging not implemented.")
        wandb_run.log(context=context, episode=episode, **log_info)

    return log_info, fig













if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes for training.")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--num_training_steps", type=int, default=20, help="Number of training steps in each episode.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Adam learning rate.")
    loss_functions = ['mse_q_values',  'mse_block_features',  'mse_q_values+mse_block_features']
    parser.add_argument("--loss_function", choices=loss_functions, default='mse_q_values', help="Loss function for training")
    parser.add_argument("--tau", type=float, default=0.01, help="Step size for updating target net.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor.")
    parser.add_argument("--model", choices=['SuccessorMLP', 'ConvNet', 'UNet'], default='UNet', help="Model type.")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cuda', help="Device to use.")
    parser.add_argument("--image_size", type=parse_img_size, default="64x64", help="Size of image features {width}x{height}.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a checkpoint to load.")
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save the checkpoint.")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="")
    parser.add_argument("--evaluate_every", type=int, default=100, help="")
    parser.add_argument("--aim", action='store_true', help="Use aim logging.")
    parser.add_argument("--aim_repo", type=str, default='aim-data/', help="Path to aim repository.")
    parser.add_argument("--tower_height", type=int, default=2, help="Height of the tower.")
    parser.add_argument("--verbose", action='store_true', help="Verbose output.")
    parser.add_argument("--log_images", action='store_true', help="Log images.")
    parser.add_argument("--replay_buffer_capacity", type=int, default=2000, help="Replay buffer capacity.")

    # parser.add_argument("--wandb", type=bool, default=False, help="Use wandb logging.")

    # parse arguments
    args = vars(parser.parse_args())
    device = args['device']
    gamma = args['gamma']
    verbose = args['verbose']

    # random seed
    if args['seed'] is not None:
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])

    # initialize environment
    x_discr_ground = np.linspace(0.4, 0.6, 9)
    offset_values = [0]
    
    xlim = (0.25, 0.75)
    ylim = (0., 0.5)
    def setup_fct():
        tower_height = 0.02 + 0.05 * args['tower_height']
        return bridge_setup(num_stories=1)
        #return tower_setup(targets=[(random.choice(x_discr_ground), 0, tower_height)])
    
    env = AssemblyGym(reward_fct=sparse_reward, max_steps=args['max_steps'], restrict_2d=True, assembly_env=AssemblyEnv(render=False))
    
    # initialize models and optimizer
    if args['model'] == 'SuccessorMLP':
        hidden_dims = [256, 128, 64, 128, 256]
        policy_net = SuccessorMLP(img_size=args['image_size'], hidden_dims=hidden_dims).to(device)
        target_net = SuccessorMLP(img_size=args['image_size'], hidden_dims=hidden_dims).to(device)
    elif args['model'] == 'ConvNet':
        policy_net = ConvNet(img_size=args['image_size']).to(device)
        target_net = ConvNet(img_size=args['image_size']).to(device)
    elif args['model'] == 'UNet':
        policy_net = Policy().to(device)
        target_net = Policy().to(device)
    else:
        raise ValueError(f"Unknown model type {args['model']}.")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args['learning_rate'])
    policy_net.apply(init_weights)
    target_net.load_state_dict(policy_net.state_dict())

    # initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=args['replay_buffer_capacity'])
    
    episode = 0
    aim_run=None
    if args['load_checkpoint']:
        raise NotImplementedError("Loading checkpoints is not tested.")
        print("loading checkpoint...")
        meta = load_checkpoint(
            path=args['load_checkpoint'], 
            policy_net=policy_net, 
            target_net=target_net, 
            replay_buffer=replay_buffer,
            optimizer=optimizer, 
        )
        optimizer_to(optimizer=optimizer, device=device)
        episode = meta['episode']

        aim_hash = meta.get('aim_hash', None)
        if aim_hash and args['aim']:
            aim_run = aim.Run(hash=aim_hash)
    else:
        if args['aim']:
            aim_run = aim.Run(experiment="SuccessorQLearning", repo=args['aim_repo'])

    # define policies
    eps_greedy = EpsilonGreedy(eps_start=0.5, gamma=0.999, eps_end=0.05, episode=episode)
    greedy = lambda q, *args, **kwargs: torch.argmax(q)


    # main training loop
    it = tqdm(range(episode + 1, episode + args['num_episodes'] + 1), disable=not verbose)
    for i in it:
        eval_round = (i % args['evaluate_every'] == 0)

        # rollout episde
        transitions, images = rollout_episode(env=env, 
                                      policy=eps_greedy.step(), 
                                      policy_net=policy_net,
                                      setup_fct=setup_fct, 
                                      x_discr_ground=x_discr_ground,
                                      offset_values=offset_values,
                                      img_size=args['image_size'],
                                      device=device,
                                      log_images=args['log_images'] and eval_round)
        # add transistions to replay buffer
        replay_buffer.push(transitions)
      
        # train policy net for n steps
        losses = train_policy_net(policy_net=policy_net, 
                    target_net=target_net, 
                    optimizer=optimizer, 
                    loss_fct=args['loss_function'],
                    replay_buffer=replay_buffer, 
                    gamma=gamma, 
                    batch_size=args['batch_size'],
                    n_steps=args['num_training_steps'],
                    device=device,
                    verbose=False)

        # update the target net
        update_target_net(policy_net=policy_net, target_net=target_net, tau=args['tau'])

        # logging
        log_info, fig = log_episode(
            episode=i, 
            transitions=transitions,
            images=images,
            policy=eps_greedy,
            log_images=args['log_images'] and eval_round, 
            losses=losses,
            context='training',
            gamma=gamma,
            aim_run=aim_run
        )
        plt.close(fig)
        it.set_postfix(episode=i, **log_info)

        # evaluate using greedy policy
        if eval_round:
            # evaluate
            transitions, images = rollout_episode(env=env, 
                                      policy=greedy, 
                                      policy_net=policy_net, 
                                      setup_fct=setup_fct, 
                                      x_discr_ground=x_discr_ground, 
                                      img_size=args['image_size'],
                                      device=device,
                                      log_images=args['log_images'])

            log_info, fig = log_episode(
                episode=i,
                transitions=transitions,
                images=images,
                log_images=args['log_images'],
                losses=None,
                context='evaluation',
                gamma=gamma,
                aim_run=aim_run
            )
            plt.close(fig)

            # # visualize transitions
            # fig = plot_transitions(transitions, policy_net)
            # exp.log_figure(fig, step=i, context='evaluation')
            # fig.close()

            it.set_postfix(episode=episode, **log_info)

        # save checkpoint
        if args['save_checkpoint'] and i % args['checkpoint_every'] == 0:
            save_checkpoint(path=args['save_checkpoint'], 
                            policy_net=policy_net, 
                            target_net=target_net, 
                            replay_buffer=replay_buffer, 
                            optimizer=optimizer, 
                            episode=i,
                            config=args)
            if args['verbose']:
                print(f"Checkpoint saved on episode {i}.")

       