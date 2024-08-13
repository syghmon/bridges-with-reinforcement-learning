import torch
import numpy as np
import torch.nn as nn
import random
from itertools import count
from robotoddler.ER import *
import time
import copy
from assembly_gym.utils.geometry import collision_rectangles



def optimize_model(agent, target_agent, batch_size, optimizer, gamma, memory, transitions_to_optimize = None):
    # Implementation with batchsize=1 !
    if transitions_to_optimize is None:
        if len(memory.memory) < batch_size:
            return None, None, 0
        if type(memory) is PrioritizedReplayBuffer:
            beta = 0.4
            batch_indices, transitions, batch_weights = memory.sample(batch_size, beta)
        else:
            transitions = memory.sample(batch_size)
    else:
        transitions = transitions_to_optimize
        
    time_Q_value = 0
    loss = 0
    for i, t in enumerate(transitions):
        s = t.state
        a = t.action
        r = t.reward
        next_s = t.next_state
        t = time.time()
        state_action_value = agent.get_Q_value(s, a)
        
        
        # Compute the optimal action a' selected by the target net in next_state and evaluate Q(s', a')
        next_state_value = 0
        with torch.no_grad():
            if next_s is not None:
                # Double Q Learning (should the next state value be considered fixed during optimization ?)
                #next_a, _ = target_agent.select_best_action(next_s)
                #next_state_value = agent.get_Q_value(next_s, next_a)

                # Classical Q learning with target network
                next_a, _ = target_agent.select_best_action(next_s)
                next_state_value = target_agent.get_Q_value(next_s, next_a)
        time_Q_value += time.time() - t
        expected_state_action_value = next_state_value * gamma + r
        #expected_state_action_value = max(0, (next_state_value * gamma)) + r
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        
        td_error = criterion(state_action_value, expected_state_action_value)

        if type(memory) is PrioritizedReplayBuffer and transitions_to_optimize is None:
            loss += batch_weights[i] * td_error
            memory.update_priorities([batch_indices[i]], [td_error.item()])
        else:
            loss += td_error
    
    loss /= batch_size
    
    t = time.time()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    gradient_mag = max([p.grad.data.norm() if p.grad is not None else 0 for p in agent.policy.parameters()])
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(agent.policy.parameters(), 100)
    optimizer.step()
    #print("Q-value computation: {}, gradient step: {}".format(time_Q_value, time.time() - t))
    return loss, transitions, gradient_mag

def select_action(agent, state, eps, allow_collisions=True):
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            return agent.select_best_action(state, allow_collisions)
    else:
        print("Random action")
        num_blocks = len(state["blocks"])
        collision = True
        while collision:
            block_index = np.random.randint(num_blocks+1) - 1 # adding the floor
            #print("Block index: {}".format(block_index))
            if block_index == -1:
                target_shape = -1
                target_shape_index = -1
                block_face = -1
                shape = np.random.randint(len(agent.shapes))
                shape_face = random.choice(agent.coming_faces[shape])
                offset_x = random.choice(agent.x_discr_ground)
                offset_y = random.choice(agent.y_discr_ground)

                # Only works for horizontal rectanlge (as in block.urdf)
                collision = agent.check_collision(state, [offset_x,offset_y,0.025], shape)
                #print("Checked collision for pos {} with shape {}: {}".format([offset_x,0.025], shape, collision))
            else:
                target_shape = state['blocks'][block_index][-1]
                target_shape_index = len(np.where(np.array(state['blocks'])[:block_index,-1] == target_shape)[0])
                block_face = random.choice(agent.receiving_faces[target_shape])
                shape =  np.random.randint(len(agent.shapes))
                shape_face = random.choice(agent.coming_faces[shape])
                offset_x = random.choice(agent.x_block_offset)
                offset_y = random.choice(agent.y_block_offset)

                # Only works for horizontal rectanlge (as in block.urdf)
                x = state['blocks'][block_index][0] + offset_x
                y = state['blocks'][block_index][1] + offset_y
                z = state['blocks'][block_index][2] + 0.05
                collision = agent.check_collision(state, [x,y,z], shape) or (z > max([s[2] for s in state["targets"]]) + 0.03)
                #print("Checked collision for pos {} with shape {}: {}".format([x,z], shape, collision))

        return [target_shape, target_shape_index, block_face, shape, shape_face, offset_x, offset_y], []



def run_episode(env, agent, eps, target=None, allow_collisions=True):
    state = env.reset()
    if target is not None:
        env.targets = [target]
        state = env._get_obs()
    
    print("Target at {}".format(state["targets"]))
    tot_reward = 0
    terminated = False
    while not terminated:
        action, _ = select_action(agent, state, eps, allow_collisions)
        state, reward, terminated, _ = env.step(action)
        tot_reward += reward
    
    success = 0
    if reward > 0: # means the episode was a success
        success = 1
    
    return tot_reward, success

        
def test(env, agent, num_episodes=5, allow_collisions=True):
    print("TEST")
    avg_reward = 0
    success_rate = 0
    for n in range(num_episodes):
        tot_reward, success = run_episode(env, agent, eps=0, allow_collisions=allow_collisions)
        avg_reward += tot_reward
        success_rate += success
    return avg_reward / num_episodes, success_rate / num_episodes


def train(env, agent, target_agent, num_episodes, num_updates, optimizer, batch_size, gamma, eps_schedule, lr_schedule, target_tau, test_freq, memory, HER=False):
    test_rewards = []
    test_success = []
    episode_values = [] # evaluate the value function along the episode    
    episode_rewards = []
    Q_losses = []
    gradient_mags = []
    max_weights = []

    tot_reward = 0
    tot_value = 0
    best_policy_param = agent.policy.state_dict()
    best_test_acc = 0

    for i_episode in range(num_episodes):
        print("Episode {}".format(memory.episode))

        optimizer.param_groups[0]["lr"] = lr_schedule(memory.episode)
            
        if memory.episode > 5:
            for j in range(num_updates):
                loss, transitions, gradient_mag = optimize_model(agent, target_agent, batch_size, optimizer, gamma, memory)
                gradient_mags.append(gradient_mag)
                if loss is not None:
                    Q_losses.append(loss.item())
            max_weights.append(max([p.data.norm() for p in agent.policy.parameters()]))
            
        keys = [k for k in agent.policy.state_dict().keys()]
        target_agent.policy.load_state_dict({k:(1-target_tau) * target_agent.policy.state_dict()[k] + target_tau * agent.policy.state_dict()[k] for k in keys})
        #for p_t, p in zip(target_agent.policy.parameters(), agent.policy.parameters()):
        #    p_t = (1-target_tau) * p_t + target_tau * p
        #target_agent.policy.load_state_dict(
        #if memory.episode % target_update_freq == 0 and memory.episode > 1:
        #    target_agent.policy.load_state_dict(agent.policy.state_dict())
            
        if memory.episode % test_freq == 0 and memory.episode > 1:
            r, s = test(env, agent, num_episodes=20)
            test_rewards.append(r)
            test_success.append(s)
            if True: #s >= best_test_acc:
                best_test_acc = s
                best_policy_param = agent.policy.state_dict()
            
        # Initialize the environment and get its state
        state = copy.deepcopy(env.reset())
        print("Targets: {}, Obstacles: {}".format(str(state["targets"]), str(state["obstacles"])))
        memory.episode += 1
        obstacles = state["obstacles"] # to use in HER
        heighest_block_pos = [0, 0, 0]
        action_list = []
        for t in count():
            eps_threshold = eps_schedule(memory.episode)
            action, colliding_actions = select_action(agent, state, eps_threshold, allow_collisions=False)
            next_state, reward, terminated, info = env.step(action)
            tot_reward += reward
            tot_value += agent.get_Q_value(state, action)
            reward = torch.tensor(reward)
            print("New block position: {}, shape: {}".format(info["new_block_pos"], action[3]))

            if terminated:
                next_state = None
                episode_values.append(tot_value.detach())
                episode_rewards.append(tot_reward)
                print("Episode reward: {}, episode value: {}, unstable: {}, collision: {}".format(tot_reward, tot_value.detach(), not info["stable"], info["collision"]))
                tot_reward = 0
                tot_value = 0
            else:
                #next_state = observation.copy()
                # pos = info["new_block_pos"]
                pos = next_state["blocks"][-1].position
                if pos[2] > heighest_block_pos[2]:
                    heighest_block_pos = pos
                action_list.append(action)

            # Store the transition in memory
            if type(memory) is PrioritizedReplayBuffer:
                for coll_action in colliding_actions:
                    memory.push(copy.deepcopy(state), coll_action, None, torch.tensor(-1), 1000)
                memory.push(copy.deepcopy(state), action, copy.deepcopy(next_state) if next_state is not None else None, reward, 1000) # push first with high priority
            else:
                for coll_action in colliding_actions:
                    memory.push(copy.deepcopy(state), coll_action, None, torch.tensor(-1))
                memory.push(copy.deepcopy(state), action, copy.deepcopy(next_state) if next_state is not None else None, reward)

            if terminated:
                break

            # Move to the next state
            state = copy.deepcopy(next_state)


        if HER and reward <= 0: # Hindsight Experience Replay (in case the episode failed)
            HER_strategy = "future"
            if HER_strategy == "final":
                setup = {"shapes":env.shapes, "targets":[heighest_block_pos], "obstacles":obstacles}
                state = env.reset(setup)
                for action in action_list:
                    next_state, reward, terminated, _ = env.step(action)
                    if terminated:
                        next_state = None
                    if type(memory) is PrioritizedReplayBuffer:
                        memory.push(copy.deepcopy(state), action, copy.deepcopy(next_state), torch.tensor(reward), 1000) # push first with high priority
                    else:
                        memory.push(copy.deepcopy(state), action, copy.deepcopy(next_state), torch.tensor(reward))
                    state = next_state

                    if terminated:
                        break
            elif HER_strategy == "future":
                #state = env._get_obs()
                K = min(3, len(state['blocks']))
                setup = {"shapes":env.shapes, "targets":[], "obstacles":obstacles}
                for _ in range(K):
                    env.reset(setup)
                    for i, action in enumerate(action_list):
                        num_targets_HER = min(random.randint(1,3), len(state["blocks"][i:]))
                        targets = []
                        for _ in range(num_targets_HER):
                            b = random.choice(state["blocks"][i:])
                            dx = 0.05 * random.random() - 0.025
                            dz = 0.05 * random.random() - 0.025
                            targets.append([b[0] + dx, b[1], b[2] + dz])

                        env.targets = targets
                        state_HER = copy.deepcopy(env._get_obs())

                        next_state_HER, reward, terminated, _ = env.step(action)
                        if next_state_HER is not None and len(state_HER["targets"]) < len(next_state_HER["targets"]):
                            raise NotImplementedError
                        if terminated:
                            next_state_HER = None
                        if type(memory) is PrioritizedReplayBuffer:
                            memory.push(copy.deepcopy(state_HER), action, copy.deepcopy(next_state_HER), torch.tensor(reward), 1000) # push first with high priority
                        else:
                            memory.push(copy.deepcopy(state_HER), action, copy.deepcopy(next_state_HER), torch.tensor(reward))
                        state_HER = next_state_HER

                        #if terminated:
                        #    break
            else:
                raise NotImplementedError

        
    print('Complete')

    return agent.policy, target_agent.policy, best_policy_param, test_success, test_rewards, episode_rewards, episode_values, Q_losses, memory, gradient_mags, max_weights





