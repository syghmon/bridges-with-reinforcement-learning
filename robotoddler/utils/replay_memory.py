import random
from collections import deque, namedtuple
import torch

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


# class Memory(object):
#     def __init__(self, dtype=None):
#         self.memory = []
#         if dtype is None:
#             self.dtype = Transition
#         self.dtype = dtype

#     def push(self, *args):
#         """Saves a transition."""
#         self.memory.append(self.dtype(*args))

#     def sample(self, batch_size=None):
#         if batch_size is None:
#             return Transition(*zip(*self.memory))
#         else:
#             random_batch = random.sample(self.memory, batch_size)
#             return Transition(*zip(*random_batch))

#     def append(self, new_memory):
#         self.memory += new_memory.memory

#     def __len__(self):
#         return len(self.memory)


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

# def tensor_size_MB(a):
#     return a.element_size() * a.nelement() / 1024 / 1024

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