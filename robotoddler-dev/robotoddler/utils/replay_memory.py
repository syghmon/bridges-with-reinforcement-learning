import random
from collections import deque, namedtuple
import torch
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb


class ReplayBuffer:

    def __init__(self, capacity=None):
        if capacity is None:
            self.memory = []
        else:
            self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        if not isinstance(transition, list):
            transition = [transition]
        for t in transition:
            self.memory.append(t)

    def save(self, path):
        torch.save(self.memory, path)

    def load(self, path):
        self.memory = torch.load(path)

    def sample(self, batch_size=None, stack_tensors=False, device=None):
        if batch_size is None:
            batch = self.memory
        else:
            batch = random.sample(self.memory, batch_size)

        dtype = batch[0].__class__
        if stack_tensors:
            return batch, dtype(*map(lambda x : torch.cat(x).to(device=device) if torch.is_tensor(x[0]) else x, zip(*batch)))
        else:
            return batch, dtype(*zip(*batch))
    
    def __len__(self):
        return len(self.memory)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity=None, gamma=0.99, policy_net=None, target_net=None, device=None):
        if capacity is None:
            self.memory = []
            self.priorities = []
        else:
            self.memory = deque([], maxlen=capacity)
            self.priorities = deque([], maxlen=capacity)
        self.gamma = gamma
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device

    def push(self, transition):
        if not isinstance(transition, list):
            transition = [transition]
        for t in transition:
            self.memory.append(t)
            self.priorities.append(t.td_error + 1e-5)  # Small positive constant to avoid zero priority

    def save(self, path):
        torch.save((self.memory, self.priorities), path)

    def load(self, path):
        self.memory, self.priorities = torch.load(path)

    def sample(self, batch_size=None, stack_tensors=False, device=None):
        if batch_size is None:
            batch = self.memory
        else:
            priorities_cpu = [p.cpu().item() if isinstance(p, torch.Tensor) else p for p in self.priorities]

            total_priority = sum(priorities_cpu)
            if total_priority < 1e-10:
                sampling_probabilities = [1 / len(priorities_cpu)] * len(priorities_cpu)
            else:
                sampling_probabilities = [p / total_priority for p in priorities_cpu]

            sampled_indices = np.random.choice(len(self.memory), batch_size, p=sampling_probabilities)
            batch = [self.memory[i] for i in sampled_indices]

        dtype = batch[0].__class__
        if stack_tensors:
            return batch, dtype(*map(lambda x: torch.cat(x).to(device=device) if torch.is_tensor(x[0]) else x, zip(*batch)))
        else:
            return batch, dtype(*zip(*batch))
    
    def __len__(self):
        return len(self.memory)
