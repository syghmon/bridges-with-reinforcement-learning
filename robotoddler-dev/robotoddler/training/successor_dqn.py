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

from assembly_gym.envs.gym_env import AssemblyGym, sparse_reward, tower_setup, hard_tower_setup, bridge_setup, horizontal_bridge_setup
from assembly_gym.envs.assembly_env import AssemblyEnv, Block, Shape
from assembly_gym.utils.rendering import get_rgb_array, plot_cra_assembly, render_assembly_env, render_blocks_2d

from robotoddler.utils.replay_memory import ReplayBuffer,PrioritizedReplayBuffer
from robotoddler.utils.utils import init_weights, parse_img_size, load_checkpoint, save_checkpoint, optimizer_to, convolve_with_gaussian
from robotoddler.models.cv import ConvNet, SuccessorMLP, UNet, Policy
from robotoddler.utils.actions import generate_actions, filter_actions

import torch.nn.functional as F
import wandb




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
                         'td_error'  # Add TD error
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
    cube = Shape(urdf_file='shapes/cube06.urdf')
    target_blocks = [Block(shape=cube, position=target) for target in obs['targets']]
    

    reward_features = render_blocks_2d(target_blocks, xlim=xlim, ylim=ylim, img_size=img_size).astype(np.float32)

    #reward_features /= np.sum(reward_features)
    kernel_size = 101 # Needs to be odd !!!
    sigma = 16
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
    def __init__(self, eps_start=0.5, eps_end=0.05, gamma=0.99, episode=0, max_steps=10, device=torch.device('cpu')):
        self.epsilon = (eps_start - eps_end) * (gamma ** episode) + eps_end
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.gamma = gamma
        self.max_steps = max_steps
        self.device = device
        self.step_images = [torch.zeros(64, 64, device=device) for _ in range(max_steps)]  # Assuming image size is (64, 64)

    def step(self):
        self.epsilon = (self.epsilon - self.eps_end) * self.gamma + self.eps_end
        return self

    def __call__(self, q_values, step_index, action_features, *args, **kwargs):
        if random.random() > self.epsilon:
            selected_action_index = torch.argmax(q_values).item()
        else:
            # Calculate the join score for each action
            join_scores = []
            for i in range(len(action_features)):
                action_feature = action_features[i].squeeze(0).to(self.device)
                join_score = torch.sum(self.step_images[step_index] * action_feature).item()
                join_scores.append(join_score)
                #print(f"[DEBUG] Action {i}: join score = {join_score}, action_feature sum = {action_feature.sum().item()}")

            # Choose the action with the smallest join score
            selected_action_index = torch.argmin(torch.tensor(join_scores)).item()
            #print(f"[DEBUG] Exploring: Selected action {selected_action_index} with min join score {join_scores[selected_action_index]}")

            # Update the step image with the selected action features
            self.step_images[step_index] += action_features[selected_action_index].squeeze(0).to(self.device)
            #print(f"[DEBUG] Updated step image for step {step_index}")

        return selected_action_index



    
    
class Softmax:
    def __init__(self, temp_start=1, temp_end=0.1, decay=0.99, episode=0):
        self.temp = (temp_start - temp_end) * (decay ** episode) + temp_end
        self.eps_start = temp_start
        self.eps_end = temp_end
        self.decay = decay

    def step(self):
        self.temp = (self.temp - self.temp_end) * self.decay + self.temp_end
        return self

    def __call__(self, q_values, *args, **kwargs): # Should be normalize the q values before sampling ?
        exp_q = torch.exp(q_values / self.temp)
        prob = exp_q / exp_q.sum()

        
        return np.random.choice(prob)


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
            num_actions = [max(1, len(actions)) for actions in batch.next_available_actions]
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
            
            if verbose:
                num_transitions = 3
                num_plots = 6
                fig, axes = plt.subplots(num_transitions, num_plots, figsize=(2 * num_plots, 2 * num_transitions))
                for i in range(num_transitions):
                    axes[i,0].imshow(batch.block_features.squeeze(1)[i]) # s_t
                    axes[i,1].imshow(batch.action_features.squeeze(1)[i]) # a_t
                    axes[i,2].imshow(batch.next_block_features.squeeze(1)[selected_actions[i]]) # s_t+1
                    axes[i,3].imshow(batch.next_actions_features.squeeze(1)[selected_actions[i]]) # a_t+1^*
                    axes[i,4].imshow(succ_block_features[i,0].detach().cpu())
                    axes[i,5].imshow(state_target[i].detach().cpu())
                    
                plt.show()
            
            
            #loss += mse(succ_block_features.softmax(dim=1)[:,1], state_target)

            #loss += F.binary_cross_entropy(succ_binary_features[:,0], batch.next_binary_features[selected_actions,0]) # only assesses immediate stability (gamma=0) 
        
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

def rollout_episode_scripted(env, predefined_actions, setup_fct, x_discr_ground, offset_values=[0.], img_size=(64, 64), xlim = (-3, 7), ylim = (0., 10), log_images=False, device=None):
    """
    Roll out a single episode using predefined actions.
    """
    done = False
    transitions = []
    images = [] if log_images else None

    # Environment reset
    obs, info = env.reset(**setup_fct())

    # Initial features and actions
    reward_features, obstacle_features = get_task_features(obs, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    block_features, binary_features = get_state_features(obs, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    available_actions = [*generate_actions(env, x_discr_ground=x_discr_ground, offset_values=offset_values)]
    action_features = get_action_features(env, available_actions, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    available_actions, action_features = filter_actions(env,available_actions, action_features, block_features=block_features, obstacle_features=obstacle_features,xlim=xlim, ylim=ylim)

    action_index = 0  # Track predefined actions

    while not done and action_index < len(predefined_actions):
        action = predefined_actions[action_index]
        action_index += 1

        selected_action_features = get_action_features(env, [action], img_size=img_size, device=device, xlim=xlim, ylim=ylim)[0]
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Compute linear reward function
        lin_reward = torch.sum(selected_action_features * reward_features)

        # Get next features and actions
        next_block_features, next_binary_features = get_state_features(next_observation, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
        next_available_actions = [*generate_actions(env, x_discr_ground=x_discr_ground, offset_values=offset_values)]
        next_action_features = get_action_features(env, next_available_actions, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
        next_available_actions, next_action_features = filter_actions(env,next_available_actions, next_action_features, next_block_features, obstacle_features=obstacle_features, xlim=xlim, ylim=ylim)
        num_actions = len(next_available_actions)

        # Add transition
        transitions.append(Transition(
            block_features=block_features.unsqueeze(0),  # Add batch dimension
            binary_features=binary_features.unsqueeze(0),
            action_features=selected_action_features.unsqueeze(0),
            reward_features=reward_features.unsqueeze(0),
            obstacle_features=obstacle_features.unsqueeze(0),
            action=action,
            lin_reward=lin_reward.unsqueeze(0),
            reward=torch.Tensor([reward]),
            done=done,
            next_block_features=next_block_features.expand(num_actions, -1, -1, -1),
            next_binary_features=next_binary_features.expand(num_actions, -1),
            next_actions_features=next_action_features,
            next_reward_features=reward_features.expand(num_actions, -1, -1, -1),  # Duplicated for convenience
            next_obstacle_features=obstacle_features.expand(num_actions, -1, -1, -1),  # Duplicated for convenience
            next_available_actions=next_available_actions,
            td_error=0
        ))

        # Update features and actions
        block_features = next_block_features
        binary_features = next_binary_features
        action_features = next_action_features
        available_actions = next_available_actions

        # Record additional information
        if log_images:
            img_dict = dict()
            img_dict['succ_block_features'] = next_block_features[0].cpu().numpy()
            env.assembly_env.simulate(steps=240)
            img_dict['env_render'] = get_rgb_array(near=0.001, fov=80, far=10, target=[0.5, 0, 0.1], pitch=-20)
            env.assembly_env.restore()
            images.append(img_dict)

    return transitions, images


def rollout_episode(env, policy, policy_net, x_discr_ground, setup_fct, offset_values=[0.], img_size=(64, 64), xlim=(0,1), ylim=(0,1), log_images=False, device=None):
    done = False
    transitions = []
    images = [] if log_images else None
    policy_net.eval()

    obs, info = env.reset(**setup_fct())

    reward_features, obstacle_features = get_task_features(obs, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    block_features, binary_features = get_state_features(obs, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    available_actions = [*generate_actions(env, x_discr_ground=x_discr_ground, offset_values=offset_values)]
    action_features = get_action_features(env, available_actions, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
    available_actions, action_features = filter_actions(env, available_actions, action_features, block_features=block_features, obstacle_features=obstacle_features, xlim=xlim, ylim=ylim)
    num_actions = len(available_actions)
    
    step_index = 0

    while not done:
        with torch.no_grad():
            q_values, succ_block_features, succ_binary_features = policy_net(block_features.expand(num_actions, -1, -1, -1),
                                                   binary_features.expand(num_actions, -1),
                                                   action_features, 
                                                   reward_features.expand(num_actions, -1, -1, -1),
                                                   obstacle_features.expand(num_actions, -1, -1, -1))
        selected_action_index = policy(q_values, step_index, action_features, succ_block_features, succ_binary_features)
        action = available_actions[selected_action_index]
        selected_action_features = action_features[selected_action_index]
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frozen_stable, unfrozen_stable = env.stabilities_freezing()

        lin_reward = torch.zeros(1, device=device).view(-1)
        if frozen_stable:
            lin_reward = torch.sum(selected_action_features * reward_features).view(-1) / 100
        if unfrozen_stable:
            lin_reward = torch.sum(selected_action_features * reward_features).view(-1)

        next_block_features, next_binary_features = get_state_features(next_observation, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
        next_available_actions = [*generate_actions(env, x_discr_ground=x_discr_ground, offset_values=offset_values)]
        next_action_features = get_action_features(env, next_available_actions, img_size=img_size, device=device, xlim=xlim, ylim=ylim)
        next_available_actions, next_action_features = filter_actions(env, next_available_actions, next_action_features, block_features=next_block_features, obstacle_features=obstacle_features, xlim=xlim, ylim=ylim)
        num_actions = len(next_available_actions)

        if len(next_available_actions) == 0:
            done = True
            next_action_features = torch.zeros([1, 1, *img_size], device=device)

        with torch.no_grad():
            q_value = q_values[selected_action_index].item()
            if not done:
                next_q_values, _, _ = policy_net(next_block_features.expand(num_actions, -1, -1, -1),
                                                 next_binary_features.expand(num_actions, -1),
                                                 next_action_features, 
                                                 reward_features.expand(num_actions, -1, -1, -1),
                                                 obstacle_features.expand(num_actions, -1, -1, -1))
                next_q_value = next_q_values.max().item()
            else:
                next_q_value = 0

            expected_q_value = reward + 0.95 * next_q_value
            td_error = abs(q_value - expected_q_value)

        transitions.append(Transition(
            block_features=block_features.unsqueeze(0),
            binary_features=binary_features.unsqueeze(0),
            action_features=action_features[selected_action_index].unsqueeze(0),
            reward_features=reward_features.unsqueeze(0),
            obstacle_features=obstacle_features.unsqueeze(0),
            action=action,
            lin_reward=lin_reward.unsqueeze(0),
            reward=torch.Tensor([reward]),
            done=done,
            next_block_features=next_block_features.expand(max(1,num_actions), -1, -1, -1),
            next_binary_features=next_binary_features.expand(max(1,num_actions), -1),
            next_actions_features=next_action_features,
            next_reward_features=reward_features.expand(max(1,num_actions), -1, -1, -1),
            next_obstacle_features=obstacle_features.expand(max(1,num_actions), -1, -1, -1),
            next_available_actions=next_available_actions,
            td_error=td_error,
        ))

        block_features = next_block_features
        binary_features = next_binary_features
        action_features = next_action_features
        available_actions = next_available_actions

        if log_images:
            img_dict = dict()
            if succ_block_features is None:
                raise ValueError("No successor block features available from the chosen policy net. Disable image logging or use a different model.")
            img_dict['succ_block_features'] = succ_block_features[selected_action_index][0].cpu().numpy()

            if env.assembly_env.client is not None:
                env.assembly_env.simulate(steps=240)
                img_dict['env_render'] = get_rgb_array(near=0.001, fov=80, far=10, target=[0.5, 0, 0.1], pitch=-20)
                env.assembly_env.restore()
            else:
                fig, ax = plt.subplots(figsize=(5, 5), frameon=False)
                ax.set_axis_off()
                fig, ax = plot_cra_assembly(env, plot_forces=False, fig=fig, ax=ax)
                fig.subplots_adjust(bottom=0, left=0, right=1, top=1)
                fig.canvas.draw()
                rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plt.close(fig)
                img_dict['env_render'] = rgb_array
            images.append(img_dict)

        step_index += 1

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
        num_plots = 7  # Increased by 1 for annotations
        fig, axes = plt.subplots(num_transitions + 1, num_plots, figsize=(2 * num_plots, 2 * (num_transitions + 1)))
        for i, t in enumerate(transitions):
            if verbose:
                print(t.reward[0])
            axes[i, 0].imshow(t.block_features.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 1].imshow(t.block_features.squeeze().cpu().numpy() + t.action_features.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 2].imshow(t.reward_features.squeeze().cpu().numpy(), cmap='gray')
            axes[i, 3].imshow(t.obstacle_features.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            if images:
                axes[i, 4].imshow(images[i]['succ_block_features'], vmin=0, vmax=1, cmap='gray')
                if 'env_render' in images[i]:
                    axes[i, 5].imshow(images[i]['env_render'])

            # Add text annotations for reward, lin_reward, and stability in a separate column
            axes[i, 6].text(0.5, 0.5, f'Reward: {t.reward.item():.2f}\nLin Reward: {t.lin_reward.item():.2f}\nStable: {t.next_binary_features[0, 0].item()}', 
                            fontsize=8, ha='center', va='center', transform=axes[i, 6].transAxes)

        # Add episode-level reward and lin_reward at the bottom
        axes[num_transitions, 6].text(0.5, 0.5, f'Total Reward: {reward:.2f}\nTotal Lin Reward: {lin_reward:.2f}',
                                      fontsize=10, ha='center', va='center', transform=axes[num_transitions, 6].transAxes)
        
        axes[0, 0].set_title('Block Features')
        axes[0, 1].set_title('Next state Features')
        axes[0, 2].set_title('Reward Features')
        axes[0, 3].set_title('Obstacle Features')
        if images:
            axes[0, 4].set_title('Successor Features')
            # axes[0, 5].set_title('Environment Render')
        axes[0, 6].set_title('Annotations')

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
        log_name = f"episode_{str(episode).zfill(5)}"
        wandb.log({
            "episode": episode,
            "reward": reward,
            "lin_reward": lin_reward,
            "avg_loss": avg_loss,
            "num_steps": num_steps,
            "stable": stable,
            "collision": collision,
            f"{log_name}_combined_image": wandb.Image(fig) if log_images else None,
            "eval_reward": lin_reward if context == 'evaluation' else None
        })

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
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cpu', help="Device to use.")
    parser.add_argument("--image_size", type=parse_img_size, default="64x64", help="Size of image features {width}x{height}.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a checkpoint to load.")
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save the checkpoint.")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="")
    parser.add_argument("--evaluate_every", type=int, default=100, help="")
    parser.add_argument("--aim", action='store_true', help="Use aim logging.")
    parser.add_argument("--aim_repo", type=str, default='aim-data/', help="Path to aim repository.")
    parser.add_argument("--bridge_length", type=int, default=1, help="Length of the bridge in blocks")
    parser.add_argument("--verbose", action='store_true', help="Verbose output.")
    parser.add_argument("--log_images", action='store_true', help="Log images.")
    parser.add_argument("--replay_buffer_capacity", type=int, default=2000, help="Replay buffer capacity.")
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb logging.")

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
    x_discr_ground = np.linspace(-2, 0, 10)
    #x_discr_ground = [-0.9]
    offset_values = [0]
    
    xlim = (-3, 7)
    ylim = (0., 10)
    
    #will be defined later
    """def setup_fct():
        return horizontal_bridge_setup(num_obstacles=args['bridge_length'])
    
    env = AssemblyGym(reward_fct=sparse_reward, max_steps=args['max_steps'], restrict_2d=True, assembly_env=AssemblyEnv(render=False))
    """
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
    replay_buffer = ReplayBuffer(
        capacity=args['replay_buffer_capacity'],
    )

    # replay_buffer = ReplayBuffer(
    #     capacity=args['replay_buffer_capacity']
    # )

    episode = 0
    aim_run=None
    wandb_run=None
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
        if args['wandb']:
            wandb_run = wandb.init(project="dual_arm", config=args)

    # define policies
    eps_greedy = EpsilonGreedy(eps_start=0.5, gamma=0.999, eps_end=0.05, episode=episode)
    greedy = lambda q, *args, **kwargs: torch.argmax(q)


    # main training loop
    #for j in range(1,4):



    # ths part can go into the function

    def setup_fct():
        return horizontal_bridge_setup(num_obstacles=args['bridge_length'])
    

    j = 1
    #temporary test:
    curr_lr = args['learning_rate']
    env = AssemblyGym(reward_fct=sparse_reward, max_steps=args['max_steps'], restrict_2d=True, assembly_env=AssemblyEnv(render=False))
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=curr_lr)
    it = tqdm(range(episode + 1, episode + args['num_episodes'] + 1), disable=not verbose)
    for i in it:
        eval_round = (i % args['evaluate_every'] == 0)

        # rollout episde
        transitions, images = rollout_episode(env=env, 
                                    policy=eps_greedy.step(), 
                                    policy_net=policy_net,
                                    setup_fct=setup_fct, 
                                    x_discr_ground=x_discr_ground,
                                    xlim=xlim,
                                    ylim=ylim,
                                    offset_values=offset_values,
                                    img_size=args['image_size'],
                                    device=device,
                                    log_images=args['log_images'])
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
            log_images=args['log_images'], 
            losses=losses,
            context='training',
            gamma=gamma,
            aim_run=aim_run,
            wandb_run=wandb_run
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
                                        xlim=xlim,
                                        ylim=ylim,
                                        offset_values=offset_values,
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
                aim_run=aim_run,
                wandb_run=wandb_run
            )
            plt.close(fig)

            # # visualize transitions
            # fig = plot_transitions(transitions, policy_net)
            # exp.log_figure(fig, step=i, context='evaluation')
            # fig.close()

            it.set_postfix(episode=episode, **log_info)


    save = False
    if(save):
        torch.save(policy_net.state_dict(), os.path.join('', 'policy_net.pth'))
        print("saved.")




    