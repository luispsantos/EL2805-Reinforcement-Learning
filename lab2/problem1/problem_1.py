# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code authors: [Simon Mello - smello@kth.se]
#               [Luis Santos - lmpss@kth.se]

# Load packages
import time
import copy
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_utils import Experience, ExperienceReplayBuffer, NeuralNet
from DQN_agent import RandomAgent


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def linear_decay(epsilon_min, epsilon_max, Z, k):
    epsilon_k = max(epsilon_min, epsilon_max - ((epsilon_max-epsilon_min)*(k-1)) / (Z-1))
    return epsilon_k


def exponential_decay(epsilon_min, epsilon_max, Z, k):
    epsilon_k = max(epsilon_min, epsilon_max * (epsilon_min/epsilon_max) ** ((k-1)/(Z-1)))
    return epsilon_k


def plot_q_values(network, steps=50):
    y = np.linspace(0, 1.5, steps)
    w = np.linspace(-np.pi, np.pi, steps)
    
    Y, W = np.meshgrid(y, w)
    states = np.zeros((steps**2, 8))
    
    states[:, 1] = Y.flatten()
    states[:, 4] = W.flatten()
    
    values = network(torch.tensor(states, dtype=torch.float32, requires_grad=False))
    max_values, argmax_values = values.max(1)

    max_Q = max_values.detach().numpy().reshape((steps, steps))
    argmax_Q = argmax_values.detach().numpy().reshape((steps, steps))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, W, max_Q, cmap='viridis', edgecolor='none')
    
    ax.set_title('Maximum Q values')
    ax.set_xlabel('y')
    ax.set_ylabel('$\omega$')
    ax.set_zlabel('Max Q')
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, W, argmax_Q, cmap='viridis', edgecolor='none')
    
    ax.set_title('Optimal Q actions')
    ax.set_xlabel('y')
    ax.set_ylabel('$\omega$')
    ax.set_zlabel('Action')
    ax.set_zticks([0, 1, 2, 3])
    ax.set_zticklabels(['Stay', 'Left E.', 'Main E.', 'Right E.'])


def random_agent(env, episodes):
    agent = RandomAgent(n_actions)
    episode_reward_list = []
    for i in range(episodes):
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    return running_average(episode_reward_list, n_ep_running_average)


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 400                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
learning_rate = 5e-4                         # Learning rate
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
buffer_length = 30000                        # Replay buffer size
N_batch = 128                                # Training batch
buffer_fill = 15000                          # Initial buffer fill
C = int(buffer_length / N_batch)             # Target network update frequency
epsilon_min = 0.05                           # Final epsilon value after decay
epsilon_max = 0.99                           # Initial epsilon value before decay
Z = 0.9*N_episodes                           # Episode range epsilon decreases over
hidden_layer_size = 64                       # Number of neurons per hidden layer
render_env = False                           # Show Lunar Lander animation or not

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Create the neural networks
network = NeuralNet(dim_state, hidden_layer_size, n_actions)
target_network = copy.deepcopy(network)
optimizer = optim.Adam(network.parameters(), learning_rate)
step_count = 0

# Create the experience replay buffer
buffer = ExperienceReplayBuffer(maximum_length=buffer_length)
buffer.random_fill(buffer_fill)

### Training process
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    # Perform linear decay of epsilon across the episodes
    epsilon = linear_decay(epsilon_min, epsilon_max, Z, i)

    while not done:
        # Environment rendering
        if render_env:
            env.render()
            #time.sleep(0.05)

        # Compute output of the network (one value per action) given the state tensor
        value = network(torch.tensor([state], requires_grad=False, dtype=torch.float32))

        # Follow an epsilon-greedy policy for action selection
        if random.random() > epsilon:
            # Take a greedy action
            action = value.max(1)[1].item()
        else:
            # Take a random action
            action = env.action_space.sample()

        # Get next state and reward, as well as done check
        next_state, reward, done, _ = env.step(action)
        
        # Append experience to the buffer
        buffer.append(Experience(state, action, reward, next_state, done))

        # Sample N experiences from buffer
        states, actions, rewards, next_states, dones = buffer.sample_batch(N_batch)

        # Compute output of the network given the states batch
        target_values = target_network(torch.tensor(next_states, requires_grad=False, dtype=torch.float32))

        not_done = 1 - torch.tensor(dones, dtype=torch.int32)
        y = torch.tensor(rewards, dtype=torch.float32) + not_done * discount_factor * target_values.max(1)[0]
        y = y.view(-1, 1)
        
        # Compute output of the network given the states batch
        values = network(torch.tensor(states, requires_grad=True, dtype=torch.float32))
        values = values.gather(1, torch.tensor(actions).view(-1,1))

        # Training process, set gradients to 0
        optimizer.zero_grad()

        # Compute loss function
        loss = nn.functional.mse_loss(values, y)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        optimizer.step()

        # Update episode reward
        total_episode_reward += reward
        state = next_state
        t += 1
        step_count += 1

        # Update the target network
        if step_count == C:
            target_network = copy.deepcopy(network)
            step_count = 0

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - Loss: {:.2f}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1], loss.item()))

# Saves the network
torch.save(network, 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), 'r--', label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), 'r--', label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)

# Creates 3D plots for max and argmax of Q-function
plot_q_values(network)

# Compares Q-network with random agent
random_rewards = random_agent(env, N_episodes)
fig = plt.figure()
plt.plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward, Q-network agent')
plt.plot([i for i in range(1, N_episodes+1)], random_rewards, 'r--', label='Avg. episode reward, Random agent')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Deep Q-Network vs Random Agent')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
