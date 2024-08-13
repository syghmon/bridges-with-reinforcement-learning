# Building Bridges with Reinforcement Learning

This repository is a fork of the original project hosted on GitLab and has been actively maintained for the last 1.5 years by Johannes Kirschner and Paul Rolland at SDSC. I had the privilege of contributing to this project as part of my bachelor thesis, which focused on leveraging reinforcement learning for autonomous robotic construction.

## Project Overview

The construction industry has traditionally relied on labor-intensive and time-consuming processes, often susceptible to human error and inefficiencies. With advancements in robotics and artificial intelligence, new possibilities for automating construction tasks have emerged. However, traditional construction robots typically rely on precise, preprogrammed movements, which can be impractical in dynamic real-world scenarios.

This project aims to address these challenges by developing a dual robotic arm system capable of autonomously constructing spanning structures, such as bridges, without the need for supporting pillars. By utilizing reinforcement learning (RL), the system learns optimal behaviors through trial-and-error interactions with the environment, enabling it to adapt to dynamic, unpredictable conditions and design structures by fulfilling abstract functional goals.

## Installation

The project consists of two main packages:

- **`assembly_gym`**: Contains the environments for the construction tasks.
- **`robotoddler`**: Includes the reinforcement learning algorithms used to train the robotic system.

To install the packages, run the following commands:

```bash
pip install -e assembly_gym
pip install -e .
```

## Getting Started

To see how to use the Assembly environment, check out the `notebooks/AssemblyGym.ipynb` notebook. This provides an introduction to setting up and interacting with the construction environment.

## Reinforcement Learning with Q-Learning and Successor Features

The core of this project is the application of reinforcement learning to teach the dual robotic arm system to autonomously construct bridges. Specifically, the project explores the use of Q-learning and Successor Features to achieve this.

### Running Standard DQN with ConvNet

To train a standard Deep Q-Network (DQN) with a convolutional neural network (ConvNet) as the Q-network and epsilon-greedy exploration, use the following command:

```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=200 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=2 --loss_function=mse_q_values --model=ConvNet
```

### Q-learning with SuccessorMLP

For a simple task using Q-learning with SuccessorMLP, run:

```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=200 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=2 --log_images --loss_function=mse_q_values --model=SuccessorMLP
```

### Learning Successor Features Directly

To solve the same task by directly learning the successor features, use:

```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=500 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=2 --log_images --loss_function=mse_block_features --model=SuccessorMLP
```

### Complex Task with Combined Loss Function

For a more complex task involving a tower height of 4, where the loss function combines Q-learning with learning successor features, use:

```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=2000 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=4 --log_images --max_steps=15 --loss_function=mse_q_values+mse_block_features --model=SuccessorMLP
```

## Logging with Aim Stack

To use [Aim Stack](https://aimstack.io) for logging, first install Aim with:

```bash
pip install aim
```

All data will be stored locally by default in the `aim-data/` directory. To access the dashboard, navigate to the `aim-data` directory and run:

```bash
cd aim-data && aim up
```

Then, open the provided URL in your browser. The repository also supports other logging mechanisms, such as Weights and Biases (wandb).
