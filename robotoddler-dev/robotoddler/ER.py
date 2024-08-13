import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.episode = 0
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    
class PrioritizedReplayBuffer(object):
    def __init__(self, max_size, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.idx = 0
        self.episode = 0

    def push(self, state, action, next_state, reward, priority):
        if len(self.memory) < self.max_size:
            self.memory.append(Transition(state, action, next_state, reward))
        else:
            self.memory[self.idx] = Transition(state, action, next_state, reward)
        self.priorities[self.idx] = priority
        self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:len(self.memory)]
        priorities = priorities ** self.alpha
        prob = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=prob)
        weights = (len(self.memory) * prob[indices]) ** (-beta)
        weights /= weights.max()

        experiences = [self.memory[idx] for idx in indices]
        return indices, experiences, weights

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority