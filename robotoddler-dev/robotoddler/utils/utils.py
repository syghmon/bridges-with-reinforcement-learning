from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
import random
import argparse
import os

import json


def init_weights(m):
    # weights initialization
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def parse_img_size(s):
    """
    Argument parser for image size.
    """
    try:
        return tuple(map(int, s.split('x')))
    except:
        raise argparse.ArgumentTypeError("Image size must be a tuple of integers.")
        

def load_checkpoint(path, policy_net, target_net, replay_buffer, optimizer, devices=None):
    """
    Load a checkpoint from a given path.
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        raise FileNotFoundError(f"Path {path} does not exist or is not a directory.")
    
    # read meta information
    with open(os.path.join(path, 'meta.json')) as f:
        meta = json.load(f)

    if devices is None:
        devices = dict()
     
    policy_net.load_state_dict(torch.load(os.path.join(path, 'policy_net.pt'), map_location=devices.get('policy_net')))
    target_net.load_state_dict(torch.load(os.path.join(path, 'target_net.pt'), map_location=devices.get('target_net')))
    optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location=devices.get('optimizer')))
    replay_buffer.load(os.path.join(path, 'replay_buffer.pt'))

    return meta


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def save_checkpoint(path, policy_net, target_net, replay_buffer, optimizer, episode, config, aim_run=None, wandb_run=None):
    """
    Save a checkpoint to a given path.
    """
    current = os.path.join(path, str(episode))
    os.makedirs(current, exist_ok=True)
    torch.save(policy_net.state_dict(), os.path.join(current, 'policy_net.pt'))
    torch.save(target_net.state_dict(), os.path.join(current, 'target_net.pt'))
    torch.save(optimizer.state_dict(), os.path.join(current, 'optimizer.pt'))
    replay_buffer.save(os.path.join(current, 'replay_buffer.pt'))


    meta = dict(
        episode=episode,
        timestamp=str(datetime.now()),
        config=config,
    )

    if aim_run is not None:
        meta['aim_hash'] = aim_run.hash

    # save meta information
    with open(os.path.join(current, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


    # symlink the latest checkpoint
    latest = os.path.join(path, 'latest')
    if os.path.exists(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(current), latest)



def gaussian_kernel(kernel_size, sigma):
    # Create a 1D Gaussian kernel
    coords = torch.arange(kernel_size) - kernel_size // 2
    kernel1d = torch.exp(-(coords.float() ** 2) / (2 * sigma ** 2))
    
    # Normalize the kernel
    kernel1d /= kernel1d.sum()
    
    # Convert 1D kernel to 2D
    kernel2d = kernel1d.unsqueeze(0) * kernel1d.unsqueeze(1)
    
    return kernel2d

def convolve_with_gaussian(input_tensor, kernel_size, sigma):
    # Generate Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Apply convolution
    output_tensor = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), 
                             kernel.unsqueeze(0).unsqueeze(0), 
                             padding=kernel_size//2).squeeze(0).squeeze(0)
    
    return output_tensor