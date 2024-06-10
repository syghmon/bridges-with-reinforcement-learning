# Robotoddler

## Installation

There are two packages: `assembly_gym` which contains the environments, 
and `robotoddler` which contains reinforcement learning algorithms.

To install the packages, run the following commands:

```bash
pip install -e assembly_gym
pip install -e .
```

## Getting started

Take a look at the `notebooks/AssemblyGym.ipynb` notebook to see how to use the Assembly environment.



## Image Based Q-Learning and Successor Features

The image features and training routines are defined in `robotoddler/training/successor_dqn.py`. 

To run standard DQN with a ConvNet as a Q-network and epsilon greedy for exploration, use:

```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=200 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=2 --loss_function=mse_q_values --model=ConvNet
```
This is a very simple task and the training should converge to the optimal policy almost immmediately. Q-learning with SucessorMLP:
```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=200 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=2 --log_images --loss_function=mse_q_values --model=SuccessorMLP
```



To solve the same task learning the successor features directly, use

```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=500 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=2 --log_images --loss_function=mse_block_features --model=SuccessorMLP
```

Note that the loss is directly defined on the successor features, not the q values. This should also converge within 500 episodes and nicely predict the successor images.

Lastly, a slightly more complex task with tower height 4. Note that here we use a loss that combines q-learning with learning sucessor features, which helps to find the optimal policy faster:

```bash
python robotoddler/training/successor_dqn.py --aim --verbose --batch_size=32 --num_training_steps=25 --evaluate_every=10 --num_episodes=2000 --device=cuda --learning_rate=0.0001 --tau=0.01 --gamma=0.95 --seed=2 --tower_height=4 --log_images --max_steps=15 --loss_function=mse_q_values+mse_block_features --model=SuccessorMLP
```

### Use Aim Stack for Logging

To use [Aim Stack](https://aimstack.io) for logging first install `pip install aim`. All data is stored locally, by default in the `aim-data/` directory. 
To access the dashboard, run `cd aim-data && aim up`, and then open the url in the browser. We can also easily add wandb or some other logging mechanism.